# -*- coding: utf-8 -*-
# --Based on YOLOX made by Megvii, Inc. and its affiliates.--

import math
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent).replace('/','\\'))
sys.path.append(str(Path(__file__).parent).replace('/','\\'))

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from nets.darknet import CSPDarknet
from nets.net_componets import BaseConv, CSPLayer, DWConv, TBLFAM

from utils.functions import meshgrid,bboxes_iou,postprocess,cxcywh2xyxy,generalized_box_iou

from utils.losses import IOUloss


# YOLOX-L

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self,depth=1.0,width=1.0,in_features=("dark3", "dark4", "dark5"),in_channels=[256, 512, 1024],depthwise=False,act=True):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # [8,1024,H,W] -> [8,512,H,W]
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), kernel_size=1, stride=1, act=act)
        # [8,1024,H,W] -> [8,512,H,W]
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )  # cat
        
        # [8,512,H,W] -> [8,256,H,W]
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), kernel_size=1, stride=1, act=act)
        # [8,512,H,W] -> [8,256,H,W]
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            n=round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        # [8,256,H,W] -> [8,256,H/2,W/2]
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), kernel_size=3, stride=2, act=act)
        # [8,512,H,W] -> [8,512,H,W]
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        # [8,512,H,W] -> [8,512,H/2,W/2]
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=3, stride=2, act=act)
        # [8,1024,H,W] -> [8,1024,H,W]
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            n=round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        # x2: [8,256,80,80]
        # x1: [8,512,40,40]
        # x0: [8,1024,20,20]

        # [8,1024,20,20] -> [8,512,20,20]
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # [8,512,20,20] -> [8,512,40,40]
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # [8,512,40,40] + [8,512,40,40]  -> [8,1024,40,40]
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        # [8,1024,40,40] -> [8,512,40,40]
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        
        # [8,512,40,40] -> [8,256,40,40]
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # [8,256,40,40] -> [8,256,80,80]
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # [8,256,80,80] + [8,256,80,80] -> [8,512,80,80]
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        # [8,512,80,80] -> [8,256,80,80]
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        
        # [8,256,80,80] -> [8,256,40,40]
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # [8,256,40,40] + [8,256,40,40] -> [8,512,40,40]
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # [8,512,40,40] -> [8,512,40,40]
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        
        # [8,512,40,40] -> [8,512,20,20]
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # [8,512,20,20] + [8,512,20,20] -> [8,1024,20,20]
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # [8,1024,20,20] -> [8,1024,20,20]
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        #  [8,256,80,80]  [8,512,40,40]  [8,1024,20,20]
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs




class YOLOXHead(nn.Module):
    '''
    输出并未经过 sigmoid 只是单纯的网络
    '''
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act=True,
            depthwise=False,
            n_anchor = 1,
            device = 'cuda',
            memory = False,
            heads = 4,
            drop=0.0,
            conf_thre = 0.05,
            sim_thresh = 0.75,
            pre_nms = 0.75, # pre_nms
            memory_num = 100,
            pre_num = 750,
            ave = True,
            use_mask = False,
            use_score=True,
            ):
        super().__init__()

        self.n_anchors = 1 # eiyi: anchor-free?
        self.num_classes = num_classes
        self.strides = strides
        
        self.use_l1 = False

        self.memory = memory
        self.nms_thresh = pre_nms
        self.memory_num = memory_num
        self.pre_num = pre_num
        self.ave = ave
        self.use_mask = use_mask
        self.use_score = use_score

        self.conf_thre = conf_thre
        self.x_shifts = None
        self.y_shifts = None
        self.expanded_strides = None
        self.total_anchors_num = None
        self.n_anchor = n_anchor
        self.width = int(256 * width)
        self.sim_thresh = sim_thresh

        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        
        self.device = device
        
        if self.memory:
            self.tem_fam = TBLFAM(num_heads=heads,num_channels=self.width,out_channels=4*self.width,attn_drop=drop).to(device)
            self.linear_pred = nn.Linear(int(4 * self.width),
                                     num_classes ) # + 1  # Mlp(in_features=1024,hidden_features=self.num_classes+1)
            self.memory_cls = None
            self.memory_reg = None
            self.ref_loss = nn.BCEWithLogitsLoss(reduction="none")

        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv


        for i in range(len(in_channels)):
            # stem: [8,256/512/1024,H,W] -> [8,256,H,W]
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width),out_channels=int(256 * width),kernel_size=1,stride=1,act=act)
            )
            # [8,256,H,W] -> [8,256,H,W]
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(in_channels=int(256 * width),out_channels=int(256 * width),kernel_size=3,stride=1,act=act),
                        Conv(in_channels=int(256 * width),out_channels=int(256 * width),kernel_size=3,stride=1,act=act),
                    ]
                )
            )
            # [8,256,H,W] -> [8,256,H,W]
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(in_channels=int(256 * width),out_channels=int(256 * width),kernel_size=3,stride=1,act=act),
                        Conv(in_channels=int(256 * width),out_channels=int(256 * width),kernel_size=3,stride=1,act=act),
                    ]
                )
            )
            # [8,256,H,W] -> [8,n_anchors * num_classes,H,W]
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width),out_channels=self.n_anchors * self.num_classes,kernel_size=1,stride=1,padding=0)
            )
            # [8,256,H,W] -> [8,4,H,W]
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width),out_channels=4,kernel_size=1,stride=1,padding=0)
            )
            # [8,256,H,W] -> [8,n_anchors * 1,H,W]
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width),out_channels=self.n_anchors * 1,kernel_size=1,stride=1,padding=0)
            )

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feat, labels=None, imgs=None, nms_thresh=0.5):
        '''
        feat :[[B,C1,H1,W1],[B,C2,H2,W2],[B,C3,H3,W3]]
        '''
        outputs = []
        outputs_decode = []
        origin_preds = []
        self.x_shifts = []
        self.y_shifts = []
        self.expanded_strides = []
        
        if self.memory:
            before_nms_cls_feats = []
            before_nms_reg_feats = []
        self.device = feat[0].device
        
        for k, (cls_conv, reg_conv,stride_this_level, x) in enumerate(zip(self.cls_convs, self.reg_convs,self.strides, feat)):
            x = self.stems[k](x)
            # [B,chanel_in,H,W] -> [B,256,H,W]
            cls_x = x 
            reg_x = x

            cls_feat = cls_conv(cls_x) # [B,256,H,W]
            cls_output = self.cls_preds[k](cls_feat) # [B,80*1,H,W]

            reg_feat = reg_conv(reg_x) # [B,256,H,W]
            reg_output = self.reg_preds[k](reg_feat).clamp(min=0.) # [B,4,H,W]
            obj_output = self.obj_preds[k](reg_feat) # [B,1*1,H,W]
            
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1) # [2,4+1+CLS_NUM,80,80]
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                ) # [2,4+1+CLS_NUM,80,80]
                # # output: [B,1*H*W,(4+1+CLS_NUM)]  grid:   [1,H*W,2]
                output, grid = self._get_output_and_grid(
                    output, stride_this_level
                )
                self.x_shifts.append(grid[:, :, 0])
                self.y_shifts.append(grid[:, :, 1])
                self.expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                        .fill_(stride_this_level)
                        .type_as(feat[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize # n_anchors = 1
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                if self.memory:
                    before_nms_cls_feats.append(cls_feat)
                    before_nms_reg_feats.append(reg_feat)
                outputs.append(output)
            else:
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                # which features to choose
                if self.memory:
                    before_nms_cls_feats.append(cls_feat)
                    before_nms_reg_feats.append(reg_feat)
            outputs_decode.append(output_decode)

        if self.use_l1 and self.training:
            # [B,1*H*W,4] s -> [B,total_anchors_num,4] 
            origin_preds = torch.cat(origin_preds, dim=1)
            
        # [B,1*H*W,(5+80)] + [B,1*H*W/4,(5+80)]+ [B,1*H*W/16,(5+80)] --> [8,num_all_anchors,85]

        self.hw = [x.shape[-2:] for x in outputs_decode] # [[80, 80], [40, 40], [20, 20]]
        # [[2,4+1+CLS_NUM,H1,W1],[2,4+1+CLS_NUM,H2,W2],[2,4+1+CLS_NUM,H3,W3]] -> [2,H1*W1+H2*W2+H3*W3,4+1+CLS_NUM]
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2).permute(0, 2, 1) 
        self.total_anchors_num = outputs_decode.shape[1]
        # 将回归映射到真实位置 [2,H1*W1+H2*W2+H3*W3,4+1+CLS_NUM]

        decode_res = self.decode_outputs(outputs_decode) 

        # 就是 NMS 了
        # pred_result:list([self.memory_num,4+1+1(cls_score)+1(cls_pred)+num_cls])
        pred_result, pred_idx = self.postpro_woclass(decode_res, nms_thre=self.nms_thresh,topK=self.memory_num)   # postprocess(decode_res,num_classes=30)
        
        # return pred_result
        # if not self.training and imgs.shape[0] == 1:
        #     return self.postprocess_single_img(pred_result, self.num_classes)
        
        if self.memory:
            # cls_feat_flatten [b,H1*W1+H2*W2+H3*W3,channels]
            cls_feat_flatten = torch.cat(
                [x.flatten(start_dim=2) for x in before_nms_cls_feats], dim=2
            ).permute(0, 2, 1)  
            # [b,H1*W1+H2*W2+H3*W3,channels]
            reg_feat_flatten = torch.cat(
                [x.flatten(start_dim=2) for x in before_nms_reg_feats], dim=2
            ).permute(0, 2, 1)
            

            ### 第一个维度不不对
            features_cls, features_reg, cls_scores, fg_scores = self.find_feature_score(cls_feat_flatten, reg_feat_flatten,
                                                                                    pred_idx, pred_result)
        
        if (not self.training) and self.memory:
            cls_scores = cls_scores.to(cls_feat_flatten.dtype)
            fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        
        if self.memory and self.memory_cls is None and self.memory_reg is None:
            self.memory_cls = features_cls.clone().detach()
            self.memory_reg = features_reg.clone().detach()


        if self.memory:
            if self.use_score:
                trans_cls,self.memory_cls,self.memory_reg = self.tem_fam(
                    features_cls,
                    features_reg, 
                    self.memory_cls,
                    self.memory_reg, 
                    cls_scores, 
                    fg_scores, 
                    sim_thresh=self.sim_thresh,
                    # ave=self.ave, 
                    use_mask=self.use_mask) 
            else:
                trans_cls,self.memory_cls,self.memory_reg = self.tem_fam(
                    features_cls, 
                    features_reg, 
                    self.memory_cls,
                    self.memory_reg, 
                    None, 
                    None,
                    sim_thresh=self.sim_thresh, 
                    ave=self.ave)
            self.memory_reg = self.memory_reg.detach()
            self.memory_cls = self.memory_cls.detach()
            # trans_cls [2,memory_num, 1024]
            # fc_output = [60,11]
            fc_output = self.linear_pred(trans_cls) # # Mlp(in_features=4*width,hidden_features=self.num_classes+1)
            # fc_output = [2,memory_num, 1024]
            # fc_output = torch.reshape(fc_output, [outputs_decode.shape[0], -1, self.num_classes]) # [:, :, :-1] #  + 1
        else:
            fc_output = None
        # old: outputs:[[B,85,H1,W1],[B,85,H2,W2],[B,85,H3,W3]]
        # now: outputs: [[B,1*H*W,(4+1+CLS_NUM)]]
        if self.training:
            # [B,85,H,W]s -> [batch, n_anchors_all, 85]
            outputs = torch.cat([x for x in outputs], dim=1)# .permute(0, 2, 1)

            (
                total_loss,
                iou_loss,
                conf_loss,
                cls_loss,
                l1_loss,
                ref_loss,
                pr, # 不是召回率，只是有匹配真实框的锚框与真实框的比值
                num_others,
            )  = self._cal_losses(labels,outputs,origin_preds,fc_output,pred_idx,pred_result)
            
            output_losses = {
                "total_loss": total_loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": pr,
                'num_others':num_others,
            }
            if self.use_l1:
                output_losses['l1_loss'] = l1_loss
            if self.memory:
                output_losses['ref_loss'] = ref_loss
            return output_losses
        else:

            result, result_ori = postprocess(pred_result, self.num_classes, fc_output,nms_thre=self.nms_thresh,conf_thre=self.conf_thre)
            
            if self.memory:
                return result
            else:
                return result_ori

    def clear_memory(self):
        self.memory_cls = None
        self.memory_reg = None

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
    
    def find_feature_score(self, features, reg_features, idxs, predictions=None):
        '''
        features [b,H1*W1+H2*W2+H3*W3,channels] 
        idxs:list(idxs[topK] for img1,idxs[topK] for img2 ...)
        reg_features [b,H1*W1+H2*W2+H3*W3,channels] 
        imgs
        predictions: list(res[TopK,17] for img1,res[TopK,17] for img2 ...)

        return:
            features_cls [b,self.simN,channel_num]
            features_reg [b,self.simN,channel_num]
            cls_scores [b,self.simN]
            fg_scores [b,self.simN]
        '''
        features_cls = []
        features_reg = []
        cls_scores = []
        fg_scores = []
        for i, feature in enumerate(features):
            features_cls.append(feature[idxs[i][:self.memory_num]])
            features_reg.append(reg_features[i, idxs[i][:self.memory_num]])
            cls_scores.append(predictions[i][:self.memory_num, 5])
            fg_scores.append(predictions[i][:self.memory_num, 4])
        features_cls = torch.stack(features_cls)
        features_reg = torch.stack(features_reg)
        cls_scores = torch.stack(cls_scores)
        fg_scores = torch.stack(fg_scores)
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
            # [H1*W1+H2*W2+H3*W3,4+1 +1+ +CLS_NUM]
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + self.num_classes]), 1)

            conf_score = image_pred[:, 4] # [H1*W1+H2*W2+H3*W3]
            # 可以改成 mean + var ?
            top_pre = torch.topk(conf_score, k=min(self.pre_num,len(conf_score)))
            sort_idx = top_pre.indices[:self.pre_num]
            detections_temp = detections[sort_idx, :]

            # 预选出 750 个候选框然后进行 NMS
            nms_out_index = torchvision.ops.batched_nms(
                detections_temp[:, :4], # box
                detections_temp[:, 4] * detections_temp[:, 5], # score
                detections_temp[:, 6], # pred
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
        
    def _cal_losses(self,labels,outputs,origin_preds, refined_cls,idx,pred_res):
        '''
        outputs 是网络输出 [B,all_anchors_num,85]
        refined_cls [B,memeory_num,num_cls]
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

        if self.memory:
            ref_targets = []
            ref_masks = []

        num_fg = 0.0
        num_gts = 0.0
        
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                # 第一个维度为空，结果为 tensor([],shape=(0...))
                cls_target = outputs.new_zeros((0, self.num_classes))
                if self.memory:
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

                
                # from utils.functions import draw_weight_plus
                # logger.info(fg_mask.shape)
                # logger.info(self.hw)
                # draw_weight_plus(fg_mask,self.hw)

                torch.cuda.empty_cache()
                num_fg += num_fg_per_image # 所有匹配到真实框的锚框
                # num_fg 应该小于等于 num_gt，可能有的真实框没找到 锚框？
                
                # cls_target [num_fg_per_image,num_classes]
                cls_target = (F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) 
                                     * pred_ious_this_matching.unsqueeze(-1))

                fg_idx = torch.where(fg_mask)[0]

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

                if self.memory:
                    # ref assignment
                    ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                    fg = 0

                    gt_xyxy = cxcywh2xyxy(reg_target.clone().detach())
                    pred_box = pred_res[batch_idx][:, :4]
                    cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                    max_iou = torch.max(iou, dim=-1)
                    for ele_idx, ele in enumerate(idx[batch_idx]):
                        loc = torch.where(fg_idx == ele)[0]

                        if len(loc): # 如果是正样本
                            ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                            fg += 1
                            continue
                        if max_iou.values[ele_idx] >= 0.6:  # 如果 IOU 大于0.6
                            max_idx = int(max_iou.indices[ele_idx])
                            ref_target[ele_idx, :self.num_classes] = cls_target[max_idx, :] * max_iou.values[ele_idx]
                            fg += 1
                        else:
                            ref_target[ele_idx,:] = 0.0
                            # ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]
                    # ----------------

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(dtype))
            fg_masks.append(fg_mask)
            if self.memory:
                ref_targets.append(ref_target[:, :self.num_classes])
                ref_masks.append(ref_target[:, -1] == 0)

            if self.use_l1:
                l1_targets.append(l1_target)   
        
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        
        fg_masks = torch.cat(fg_masks, 0)
        if self.memory:
            ref_targets = torch.cat(ref_targets, 0)
            ref_masks = torch.cat(ref_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        

        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg

        num_others = (cls_targets[:,2] > 0).sum()

        if self.memory:
            loss_ref = self.ref_loss(refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]).sum() / num_fg
        else:
            loss_ref = 0.0

        if self.use_l1:
            loss_l1 = self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + loss_ref

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            2*loss_ref,
            num_fg / max(num_gts, 1), # 召回率？
            num_others,
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
    
    def _gen_points(self):
        expanded_strides_per_image = self.expanded_strides[0]
        x_shifts_per_image = self.x_shifts[0] * expanded_strides_per_image
        # [total_anchors_num]
        y_shifts_per_image = self.y_shifts[0] * expanded_strides_per_image
        
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
        )   # [total_anchors_num] -> [num_gt, total_anchors_num]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
        )

        points = torch.stack([x_centers_per_image,y_centers_per_image],dim=0).transpose(0,1)
        return points


    def _get_in_boxes_info(self,gt_bboxes_per_image):
        '''
        其中心在真实框 或 在真实框中心区域的锚框有哪些 is_in_boxes_anchor
        (is_in_boxes_anchor 的 长度是 total_anchors_num)
        中心在真实框 且 在真实框中心区域的锚框有哪些 is_in_boxes_and_center
        (is_in_boxes_and_center 的长度是 is_in_boxes_anchor 为真的数量)
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


class YOLOXNet(nn.Module):
    
    def __init__(self,num_classes, in_channels=[256, 512, 1024],backbone=None, 
                    head=None,depth=1.0,width=1.0,depthwise=False,memory=False,conf_thre=0.05,memory_num = 100,heads=4):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN(in_channels=in_channels,depth=depth,width=width,depthwise=depthwise)
        if head is None:
            head = YOLOXHead(num_classes,in_channels=in_channels,width=width,
                    depthwise=depthwise,memory=memory,conf_thre=conf_thre,
                    heads = heads,memory_num=memory_num)

        self.backbone = backbone
        self.head = head
        
    def forward(self, x, labels = None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        # head output [[B,85,H1,W1],[B,85,H2,W2],[B,85,H3,W3]]
        return self.head(fpn_outs,labels=labels)
        

@logger.catch
def test():


    input_x = torch.randn([1,3,160,160],dtype=torch.float32).to('cuda')
    labels = torch.randint(0,4,[1, 50, 5]).to('cuda')

    yolox_net = YOLOXNet(4,memory=False).to('cuda') # num_classes = 15
    yolox_net.head.use_l1 = True


    with torch.no_grad():
        outputs = yolox_net(input_x,labels)
        print(outputs)

        
    
if __name__ == "__main__":
    test()
    
    

        
        
        
