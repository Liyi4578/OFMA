#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --Based on YOLOX made by Megvii, Inc. and its affiliates.--
import sys
import copy

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent).replace('/','\\'))
sys.path.append(str(Path(__file__).parent).replace('/','\\'))
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.functions import meshgrid
from utils.boxes import bboxes_iou
from nets.net_componets import TBLFAM

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
        
        
class YOLOXLoss(nn.Module):
    
    def __init__(
            self,
            num_classes,
            strides=[8, 16, 32],
            n_anchor = 1,
            pre_nms = 0.75, # pre_nms
            memory_num = 100,
            pre_num = 750,
            ):
        '''
        n_anchor 每个网格点的锚框数量
        '''
        super(YOLOXLoss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.use_l1 = False
        self.nms_thresh = pre_nms
        self.memory_num = memory_num
        self.pre_num = pre_num

        
        # 这些变量似乎可以常态化存储在类中？
        self.x_shifts = None
        self.y_shifts = None
        self.expanded_strides = None
        self.total_anchors_num = None
        self.n_anchor = n_anchor
        
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

        self.tem_fam = TBLFAM()
    
    def forward(self,outputs_from_net, labels):
        '''
        outputs_from_net: [[B,4+1+num_classes,H1,W2],[B,4+1+num_classes,H2,W2],[B,4+1+num_classes,H3,W3]]
        '''
        
        self.x_shifts = []
        self.y_shifts = []
        self.expanded_strides = []

        # type = outputs_from_net[0].type()
        outputs = []
        origin_preds = [] if self.use_l1 else None
        
        for k,(output,stride_this_level) in enumerate(zip(outputs_from_net,self.strides)):

            # l1 loss 需要 未经 处理的 reg_output
            # origin_preds 不是 outputs[:, :, :4]  
            # output 经过 get_output_and_grid  -> 左上角+宽高 <-stride
            if self.use_l1:
                reg_output = output[:,:4,:,:]
                batch_size = output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                # [B,4,H,W] -> [B,1,4,H,W]
                reg_output = reg_output.view(
                    batch_size, self.n_anchor, 4, hsize, wsize
                )
                # [B,1,4,H,W] -> [B,1,H,W,4] -> [B,1*H*W,4]
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())
                
            # output: [B,1*H*W,(5+80)]  grid:   [1,hsize*wsize,2]
            output, grid = self._get_output_and_grid(output, stride_this_level)

            self.x_shifts.append(grid[:, :, 0]) # x_shift:[1,hsize*wsize]
            self.y_shifts.append(grid[:, :, 1]) # y_shift:[1,hsize*wsize]
            self.expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level) # 8 or 16 or 32
                .type_as(outputs_from_net[0])
            ) # expanded_strides: [1,H*W] fill with 8/16/32

            outputs.append(output)
        if self.use_l1:
            # [B,1*H*W,4] s -> [B,total_anchors_num,4] 
            origin_preds = torch.cat(origin_preds, 1)
            
        # [B,1*H*W,(5+80)] + [B,1*H*W/4,(5+80)]+ [B,1*H*W/16,(5+80)] --> [8,num_all_anchors,85]
        outputs = torch.cat(outputs, dim=1)
        # 网络输出的预测框数量(=锚框数量)
        self.total_anchors_num = outputs.shape[1]

        (
            total_loss,
            iou_loss,
            conf_loss,
            cls_loss,
            l1_loss,
            pr, # 不是召回率，只是有匹配真实框的锚框与真实框的比值
        )  = self._cal_losses(labels,outputs,origin_preds)
        outputs = {
            "total_loss": total_loss,
            "iou_loss": iou_loss,
            "l1_loss": l1_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "num_fg": pr,
        }
            
        # logger.info('num_fg/num_gt {}',pr)
        
        return outputs
    
    def decode_outputs(self, outputs, flevel=0):
        # outputs: [2,H1*W1+H2*W2+H3*W3,4+1+CLS_NUM]
        dtype = outputs.type()
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    
    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None):
        '''
        features [b,H1*W1+H2*W2+H3*W3,channels] 
        idxs:list(idxs[topK] for img1,idxs[topK] for img2 ...)
        reg_features [b,H1*W1+H2*W2+H3*W3,channels] 
        imgs
        predictions: list(res[TopK,17] for img1,res[TopK,17] for img2 ...)

        return:
            features_cls [b*self.simN,channel_num]
            features_reg [b*self.simN,channel_num]
            cls_scores [b*self.simN]
            fg_scores [b*self.simN]
        '''
        features_cls = []
        features_reg = []
        cls_scores = []
        fg_scores = []
        for i, feature in enumerate(features):
            features_cls.append(feature[idxs[i][:self.simN]])
            features_reg.append(reg_features[i, idxs[i][:self.simN]])
            cls_scores.append(predictions[i][:self.simN, 5])
            fg_scores.append(predictions[i][:self.simN, 4])
        features_cls = torch.cat(features_cls)
        features_reg = torch.cat(features_reg)
        cls_scores = torch.cat(cls_scores)
        fg_scores = torch.cat(fg_scores)
        return features_cls, features_reg, cls_scores, fg_scores

    def postpro_woclass(self, prediction, nms_thre=0.75, topK=75, features=None):
        # find topK predictions, play the same role as RPN
        '''
        Args:
            prediction: [batch,feature_num,5+clsnum] = [2,H1*W1+H2*W2+H3*W3,4+1+CLS_NUM]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''

        self.topK = topK
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4] # 中心宽高 转换为 左上右下
        output = [None for _ in range(len(prediction))]
        output_index = [None for _ in range(len(prediction))]
        features_list = []
        for i, image_pred in enumerate(prediction):
            # 每张图片 image_pred [H1*W1+H2*W2+H3*W3,4+1+CLS_NUM]
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            # class_conf [H1*W1+H2*W2+H3*W3,1]    class_pred [H1*W1+H2*W2+H3*W3,1]
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)

            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            # [H1*W1+H2*W2+H3*W3,4+1 +1++ +CLS_NUM]
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + self.num_classes]), 1)

            conf_score = image_pred[:, 4] # [H1*W1+H2*W2+H3*W3]
            # 可以改成 mean + var ?
            top_pre = torch.topk(conf_score, k=self.pre_num)
            sort_idx = top_pre.indices[:self.pre_num]
            detections_temp = detections[sort_idx, :]

            # 预选出 750 个候选框然后进行 NMS
            nms_out_index = torchvision.ops.batched_nms(
                detections_temp[:, :4], # box
                detections_temp[:, 4] * detections_temp[:, 5], # score
                detections_temp[:, 6], # 
                nms_thre,
            )
            # 过滤后的 pre_num 个
            topk_idx = sort_idx[nms_out_index[:self.topK]]
            output[i] = detections[topk_idx, :]
            output_index[i] = topk_idx

        return output, output_index

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):

        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        for i, detections in enumerate(prediction):

            if not detections.size(0):
                continue

            detections_ori = prediction_ori[i]

            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]
            nms_out_index = torchvision.ops.batched_nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                detections_ori[:, 6],
                nms_thre,
            )
            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori
        # print(output)
        return output_ori, output_ori
    

    def _get_output_and_grid(self,output, stride):
        # output:[B,85,H,W]  k:0/1/2 stride=8/16/32 
        
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        
        # yv,xv = 0,0 0,1 0,2....
        # yv.shape = xv.shape = [hsize,wsize]
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])

        # ([ny,nx],[ny,nx]) -> [hsize,wsize,2]->[1,1,hsize,wsize,2]
        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(output.type())

        # output:[B,85,H,W] -> [B,1,(5+80),H,W]
        output = output.view(batch_size, self.n_anchor, n_ch, hsize, wsize)
        # output:[B,1,(5+80),H,W] -> [B,1,H,W,(5+80)] -> [B,1*H*W,(5+80)]
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchor * hsize * wsize, -1
        )
        # grid: [1,1,hsize,wsize,2] -> [1,hsize*wsize,2] : [[[0,0],[0,1],...],...]
        grid = grid.view(1, -1, 2)
        
        # 预测的是 anchor 的左上角偏移 + grid )* stride 就是左上角偏移
        output[..., :2] = (output[..., :2] + grid) * stride 
        # 预测的是目标的大小比例的ln， exp(·)*grid 就是预测的实际大小
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride 
        return output, grid
        
    def _cal_losses(self,labels,outputs,origin_preds):
        '''
        outputs 是网络输出 [B,all_anchors_num,85]
        '''
        dtype = outputs.type()
        
        bbox_preds = outputs[:, :, :4]  # [batch, total_anchors_num, 4]
        # 也可以 obj_preds = outputs[:, :, 4:5]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, total_anchors_num, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, total_anchors_num, num_classes]
        
        # 首先的搞懂 target/labels 里面是什么
        # torch.Size([8, 3, 640, 640])
        # torch.Size([8, 50, 5])  5=4+1
        # torch.Size([8, 2]) info h,w  被 prefetcher 扔了
        # torch.Size([8, 1]) id 被 prefetcher 扔了
        
        # 计算这个batch中每张图片共有多少个目标
        # e.g. tensor([ 3,  6,  1,  2,  8,  3, 16,  4])
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
       
        
        # 所有预测框的偏移
        self.x_shifts = torch.cat(self.x_shifts, 1)  # [1, total_anchors_num]
        self.y_shifts = torch.cat(self.y_shifts, 1)  # [1, total_anchors_num]
        # 所有预测框的 stride
        # expanded_strides: [[1,H*W],[1,H*W],[1,H*W]]-> [1,total_anchors_num]
        self.expanded_strides = torch.cat(self.expanded_strides, 1) 

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                # 第一个维度为空，结果为 tensor([],shape=(0...))
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((self.total_anchors_num, 1))
                fg_mask = outputs.new_zeros(self.total_anchors_num).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :4] #[num_gt,4]
                gt_classes_per_image = labels[batch_idx, :num_gt, 4] #[num_gt]
                try:
                # fg_mask 所有锚框有匹配到真实框的有哪些 [total_anchors_num]
                # num_fg_per_image :有匹配的真实框的锚框的数量
                # gt_matched_classes 每个匹配成功锚框的类别 [num_fg_per_image]
                # pred_ious_this_matching 锚框与真实框的 IoU [num_fg_per_image]
                # matched_gt_inds每个元素表示该锚框匹配的真实框序号 [num_fg_per_image] 
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_per_image,
                    ) = self._label_assignments(batch_idx,num_gt,labels,bbox_preds,cls_preds,obj_preds)
                except RuntimeError as e:
                
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_per_image,
                    ) = self._label_assignments(batch_idx,num_gt,labels,bbox_preds, cls_preds,obj_preds,mode='cpu')
                    
                torch.cuda.empty_cache()
                num_fg += num_fg_per_image # 所有匹配到真实框的锚框
                # num_fg 应该小于等于 num_gt，可能有的真实框没找到 锚框？
                
                # cls_target [num_fg_per_image,num_classes]
                cls_target = (F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) 
                                     * pred_ious_this_matching.unsqueeze(-1))
                # obj_target [num_fg_per_image,1]
                obj_target = fg_mask.unsqueeze(-1)
                # reg_target [num_fg_per_image,4]
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_per_image, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        self.expanded_strides[0][fg_mask],
                        x_shifts=self.x_shifts[0][fg_mask],
                        y_shifts=self.y_shifts[0][fg_mask],
                    )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)   
        
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
            
        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg
        if self.use_l1:
            loss_l1 = self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1), # 召回率？
        )        
                    
                    
    def _label_assignments(self,batch_idx,num_gt,labels,bbox_preds,cls_preds,obj_preds,mode='gpu'):
        
        gt_bboxes_per_image = labels[batch_idx, :num_gt, :4]
        gt_classes_per_image = labels[batch_idx, :num_gt, 4] #[num_gt]
        
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes_per_image = gt_classes_per_image.cpu().float()
            self.expanded_strides = self.expanded_strides.cpu().float()
            self.x_shifts = self.x_shifts.cpu()
            self.y_shifts = self.y_shifts.cpu()
        # is_in_boxes_and_center [gt_num,fg_mask为真个数:num_in_boxes_anchor]
        # fg_mask [total_anchors_num]
        fg_mask, is_in_boxes_and_center = self._get_in_boxes_info(gt_bboxes_per_image)
        # 在中心或中心区域(中心区域可能比目标整个还大)的锚框
        # !!! fixed for:one of the variables needed for gradient computation has been modified by an inplace operation
        bboxes_preds_per_image = bbox_preds[batch_idx][fg_mask]
        # [batch, total_anchors_num, num_classes] -> [num_in_boxes_anchor, num_classes]
        cls_preds_per_image = cls_preds[batch_idx][fg_mask] 
        # [batch, total_anchors_num, 1] -> [num_in_boxes_anchor, 1]
        obj_preds_per_image = obj_preds[batch_idx][fg_mask]
        
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        
        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # # !!! fixed for:one of the variables needed for gradient computation has been modified by an inplace operation
        # pair_wise_ious 只参与 _dynamic_k_matching 不需要梯度？！后面把它del了
        pair_wise_ious = pair_wise_ious.detach()
        
        # [num_gt] -> [num_gt,num_classes] -> [num_gt,1,num_classes] -> [num_gt,num_in_boxes_anchor,num_classes]
        onehot_gt_cls_per_image = (
            F.one_hot(gt_classes_per_image.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        if mode == "cpu":
            cls_preds_per_image, obj_preds_per_image = cls_preds_per_image.cpu(), obj_preds_per_image.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            # 通过 sigmoid 将值压缩到了 0-1
            # cls_pred_per_imag = cls_pred_per_imag * obj_pred_per_imag
            # [num_in_boxes_anchor, num_classes] -> [1,num_in_boxes_anchor, num_classes] 
            # -> [num_gt,num_in_boxes_anchor,num_classes]
            # [num_in_boxes_anchor,1] -...> [num_gt,num_in_boxes_anchor,1]
            cls_preds_per_image = (
                cls_preds_per_image.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_per_image.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            # 开根后所有真实框与预测框的分类做 BCE Loss
            # 这里 reduction 不可以是 "sum"，这 真实框与锚框还未匹配 ！
            # pair_wise_cls_loss [num_gt,num_in_boxes_anchor]
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_per_image.sqrt_(), onehot_gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_per_image
        
        # 不是在真实框中且不在中心区域的，其 cost 极大
        # cost [num_gt,num_in_boxes_anchor]
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )
        
        # fg_mask 传进去后也做了更新[total_anchors_num]
        # num_fg :有匹配的真实框的锚框的数量
        # gt_matched_classes 每个匹配成功锚框的类别 [num_fg]
        # pred_ious_this_matching 锚框与真实框的 IoU [num_fg]
        # matched_gt_inds每个元素表示该锚框匹配的真实框序号 [num_fg] 
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self._dynamic_k_matching(cost, pair_wise_ious, gt_classes_per_image, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        
        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
        
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target 
        
    def _get_in_boxes_info(self,gt_bboxes_per_image):
        '''
        其中心在真实框 或 在真实框中心区域的锚框有哪些 is_in_boxes_anchor
        （is_in_boxes_anchor 的 长度是 total_anchors_num）
        中心在真实框 且 在真实框中心区域的锚框有哪些 is_in_boxes_and_center
        （is_in_boxes_and_center 的长度是 is_in_boxes_anchor 为真的数量）
        '''
        num_gt = gt_bboxes_per_image.shape[0]
        expanded_strides_per_image = self.expanded_strides[0]
        x_shifts_per_image = self.x_shifts[0] * expanded_strides_per_image
        # [total_anchors_num]
        y_shifts_per_image = self.y_shifts[0] * expanded_strides_per_image
        
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )   # [total_anchors_num] -> [num_gt, total_anchors_num]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, self.total_anchors_num)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, self.total_anchors_num)
        )#[num_gt] -> [num_gt,1] -> [num_gt,total_anchors_num] 
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, self.total_anchors_num)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, self.total_anchors_num)
        )
         # 理解这部分 可以画个表，打勾来理解
        #       anchor1 anchor2 anchor3 anchor4 ...
        # gt1     √
        # gt2              √
        # gt3                               √
        
        
        # 所有锚框中心点到所有真实框边界的距离
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        #  [num_gt,total_anchors_num,4] 
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        
        # is_in_boxes [num_gt,total_anchors_num]
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_radius = 2.5
        
        # 所有真实框中心 周围半径 center_radius*stride 的区域
        # [num_gt] -> [num_gt,1] -> [num_gt,total_anchors_num]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, self.total_anchors_num
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, self.total_anchors_num
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, self.total_anchors_num
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, self.total_anchors_num
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        
        # 其实可能有中心区域半径比真实框范围还要大的情况
        # 所有锚框中心点到所有真实框中心 周围半径 center_radius*stride 的区域的距离
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0

        # 不，是在gt box 中心区域的锚框有哪些 
        # is_in_centers_all [total_anchors_num] 
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        
        # 其中心在真实框 或 在真实框中心区域的锚框有哪些
        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        
        # 其中心在真实框 且 在真实框中心区域的锚框有哪些
        # is_in_boxes_and_center 的长度是 is_in_boxes_anchor 为真的数量
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        #  matching_matrix [num_gt,num_in_boxes_anchor]
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        # n_candidate_k 为 在真实框中的锚框数量 且不超过10
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # topk_ious [num_gt,n_candidate_k]
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        # 根据 IoU 动态决定每个 真实框 分配几个 锚框，但至少一个
        # 分配依据是 cost
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx
        
        # 每个锚框 对应 几个真实框
        # anchor_matching_gt [num_in_boxes_anchor]
        anchor_matching_gt = matching_matrix.sum(0)
        # 存在 有一个锚框对应多个真实框的情况
        if (anchor_matching_gt > 1).sum() > 0:
            # 选择 cost 最低的 那个真实框，让他去匹配
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
            
        # ? 似乎没有保证真实框一定有锚框对应
        
        # 有匹配的真实框的锚框有哪些
        # fg_mask_inboxes [num_in_boxes_anchor]
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        
        # fg_mask [total_anchors_num]
        # 匹配一个真实框后的锚框有哪些
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        
        # matched_gt_inds [num_fg] 每个元素表示该锚框匹配的真实框序号
        # [num_gt,num_in_boxes_anchor] ->[num_gt,num_fg] -> [num_fg]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 表示每个匹配到真实框的锚框的类别
        # gt_matched_classes [num_fg]
        gt_matched_classes = gt_classes[matched_gt_inds]
        
        # debug for 'one of the variables needed for gradient computation has been modified by an inplace operation'
        # pair_wise_ious = torch.randn_like(pair_wise_ious)
        # (matching_matrix * pair_wise_ious) [num_gt,num_in_boxes_anchor]
        # -> [num_in_boxes_anchor] -> [num_fg]
        # 匹配到真实框的锚框与真实框的 IoU
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds    





def show_get_in_boxes_info():
    import cv2
    import numpy as np
    from utils.functions import xyxy2cxcywh,cxcywh2xyxy
    
    hsize = wsize = 20
    x_shifts,y_shifts = [],[]
    expanded_strides = []
    # yv,xv = 0,0 0,1 0,2....
    # yv.shape = xv.shape = [hsize,wsize]
    yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])

    # ([ny,nx],[ny,nx]) -> [hsize,wsize,2]->[1,1,hsize,wsize,2]
    grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(torch.float32)
    grid = grid.view(1, -1, 2)
    
    x_shifts = grid[:, :, 0] # x_shift:[1,hsize*wsize]
    y_shifts = grid[:, :, 1] # y_shift:[1,hsize*wsize]
    expanded_strides = (
        torch.zeros(1, grid.shape[1])
        .fill_(32) # 8 or 16 or 32
        .type(torch.float32)
    ) # expanded_strides: [1,H*W] fill with 8/16/32
    # print(x_shifts.shape)
    # print(expanded_strides.shape)
    
    total_anchors_num = hsize*wsize

    gt_bboxes_per_image = [[120, 96, 64, 58],
                           [400,500,128, 96],
                           [400,200,250, 300],
                           [200,200,120,180]]
    num_gt = len(gt_bboxes_per_image)
    gt_bboxes_per_image = torch.tensor(gt_bboxes_per_image)
    
    canvas = np.zeros((640,640,3),dtype=np.uint8)
    draw_gt = cxcywh2xyxy(gt_bboxes_per_image.clone())
    for box in draw_gt:
        # print(box)
        box = box.tolist()
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (204, 204, 51), 1)
        
    # ----------- get in boxes info 可视化-----------
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    # [total_anchors_num]
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    
    x_centers_per_image = (
        (x_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
        .repeat(num_gt, 1)
    )  # [total_anchors_num] -> [num_gt, total_anchors_num]
    y_centers_per_image = (
        (y_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
        .repeat(num_gt, 1)
    )
    
    for x,y in zip((x_shifts_per_image + 0.5 * expanded_strides_per_image).tolist(),
                    (y_shifts_per_image + 0.5 * expanded_strides_per_image).tolist()):
        cv2.circle(canvas,(int(x),int(y)),2,(255,0,0),2)
        
    gt_bboxes_per_image_l = (
        (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_anchors_num)
    ) #[num_gt] -> [num_gt,1] -> [num_gt,total_anchors_num] 
    gt_bboxes_per_image_r = (
        (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_anchors_num)
    )
    gt_bboxes_per_image_t = (
        (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_anchors_num)
    )
    gt_bboxes_per_image_b = (
        (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_anchors_num)
    )
    # 所有锚框中心点到所有真实框边界的距离
    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image
    #  [num_gt,total_anchors_num,4] 
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
    
    # is_in_boxes [num_gt,total_anchors_num]
    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    
    # 显示所有在 gt box 中的锚框中心点
    for x,y in zip(
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)[is_in_boxes_all].tolist(),
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)[is_in_boxes_all].tolist()):
        cv2.circle(canvas,(int(x),int(y)),3,(255,0,255),2)
        
    # in fixed center
    center_radius = 2.5
    
    # 所有真实框中心 周围半径 center_radius*stride 的区域
    # [num_gt] -> [num_gt,1] -> [num_gt,total_anchors_num]
    gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        1, total_anchors_num
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        1, total_anchors_num
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        1, total_anchors_num
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        1, total_anchors_num
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    
    # 其实可能有中心区域半径比真实框范围还要大的情况
    # 所有锚框中心点到所有真实框中心 周围半径 center_radius*stride 的区域的距离
    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    # 每个锚框在几个 gt box 的中心区域里
    # 不，是在gt box 中心区域的锚框有哪些 
    # is_in_centers_all [total_anchors_num] 
    is_in_centers_all = is_in_centers.sum(dim=0) > 0
    
    for x,y in zip(
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)[is_in_centers_all].tolist(),
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)[is_in_centers_all].tolist()):
        cv2.circle(canvas,(int(x),int(y)),5,(0,255,255),2)
        
    
    # 其中心在真实框 或 在真实框中心区域的锚框有哪些
    # in boxes and in centers
    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
    fg_mask = is_in_boxes_anchor
    # print(is_in_boxes_anchor.shape)
    # print(is_in_boxes_anchor.sum())
    # 其中心在真实框 且 在真实框中心区域的锚框有哪些
    # [gt_num,num_in_boxes_anchor]
    # is_in_boxes_and_center 的长度是 is_in_boxes_anchor 为真的数量
    is_in_boxes_and_center = (
        is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    )
    print(is_in_boxes_and_center.shape)
    cv2.imshow('test',canvas)
    cv2.waitKey(0)

    for gt_anchors in is_in_boxes_and_center:
        
        cavs = canvas.copy()
        for x,y in zip(
                (x_shifts_per_image + 0.5 * expanded_strides_per_image)[is_in_boxes_anchor][gt_anchors].tolist(),
                (y_shifts_per_image + 0.5 * expanded_strides_per_image)[is_in_boxes_anchor][gt_anchors].tolist()
                ):
            cv2.circle(cavs,(int(x),int(y)),5,(255,255,255),5)
        cv2.imshow('test',cavs)
        cv2.waitKey(0)
        
    # print(is_in_boxes_anchor == is_in_boxes_all)
    # print(is_in_centers_all & is_in_boxes_all == is_in_centers_all)
    
@logger.catch
def test():
    inputs = [torch.randn([2, 20, 40, 40]), torch.randn([2, 20, 20, 20]), torch.randn([2, 20, 10, 10])]
    loss_func = YOLOXLoss(15)
    labels = torch.randint(0,15,[2, 50, 5])

    
    logger.info(loss_func(inputs,labels))
    
    
if __name__ == '__main__':
    # show_get_in_boxes_info()
    test()
    
    
    