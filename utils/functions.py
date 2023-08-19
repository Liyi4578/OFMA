# -*- encoding:utf-8 -*-
# --Based on YOLOX made by Megavii Inc.--
import os
import shutil
import copy

from loguru import logger
from copy import deepcopy
from typing import Sequence
import contextlib
import time
import torchvision

import torch
import torch.nn as nn
from torchvision.ops.boxes import box_area

import cv2
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt




def draw_weight_plus(weights,hws,points,savename = None,bboxes = None):
    # num_points = []
    # for it in hws:
        # num_points.append(it[0]*it[1])
    # weight_list = weights.split(num_points, 0)

    count = 0
    bboxes = cxcywh2xyxy(bboxes)
    for i,hw in enumerate(hws):
        h, w = hw
        weight = weights[count:count + h*w]
        weight1 = weight.reshape(h,w).detach().cpu().numpy().astype(np.uint8)
        sl_points = points[count:count + h*w]
        count += h*w
        
        print(f'level - {i+1}/{len(hws)}')
        w_scale = int((sl_points[w,1] - sl_points[0,1]).item())
        h_scale =  int((sl_points[1,0] - sl_points[0,0]).item())
        weight = cv2.resize(weight1, (h*h_scale, w*w_scale),interpolation=cv2.INTER_NEAREST) # ,interpolation=cv2.INTER_NEAREST
        heatmap0 = np.uint8(255 * weight)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        weight = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = weight  # 这里的0.4是热力图强度因子



        for x,y in sl_points:
            cv2.circle(superimposed_img,(int(x),int(y)),1,(255,0,0),2)
            
        print(weight.shape)
        
        for pred in bboxes:
            box = [int(t) for t in pred[0:4]]
            # print(box)
            # print(cat_id2str_dict[box[4]])
            cv2.rectangle(superimposed_img, (box[0], box[1]), (box[2], box[3]), (204, 204, 51), 1)


        superimposed_img = cv2.resize(superimposed_img, (h*h_scale*2, w*w_scale*2),interpolation=cv2.INTER_NEAREST) 
        cv2.imshow(f'level - {i+1}/{len(hws)}',superimposed_img)
        cv2.waitKey(0)



# 神奇的广播机制: max([box_num,1,2],[box_num,2]) -> [box_num,box_num,2]
# torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2]) 很妙哇
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # max([box_num,1,2],[box_num,2]) -> [box_num,box_num,2]
        # 就是每个boxa的左上与每个boxb的左上,取最大值
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # 就是每个boxa的右下与每个boxb的右下,取最小值
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # en [boxa_num,boxb_num]
    # 互相有交集的box有哪些
    en = (tl < br).type(tl.type()).prod(dim=2)
    
    # 交集的面积
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    # [boxa_num,1] + [boxb_num] - [boxa_num,boxb_num]
    # -> [[boxa_num,boxb_num]] Iou 每个boxa与每个boxb之间的IoU
    # 没有交集的 IoU=0
    temp = (area_a[:, None] + area_b - area_i)
    pair_wise_ious = area_i / temp
    return pair_wise_ious
    
    
# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area,iou

def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5 # x1
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5 # y1
    # 注意现在的 0 1 索引到的值已经由上面两行改变了，他们变成了 左上角点
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] # x2
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] # y2
    return bboxes

def xyxy2xywh(bboxes):
    '''
    转换为 左上角点坐标 与 宽高。
    '''
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes
    
    
def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] # width
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] # height
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5 # c_x
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5 # c_y
    return bboxes


def postprocess_origin(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    '''
    进行 NMS，返回一个大小为我batch_size的列表，每个元素是
    该图片的符合 conf_thre 与 nms_thre 的锚框与其对应的信息
    即 (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    '''
    # box_corner = prediction.new(prediction.shape)
    # prediction [B,total_anchors_num,4+1+num_classes]
    # 转换为左上，右下两个点的坐标，类似于上面的 cxcywh->xyxy
    prediction[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    prediction[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    prediction[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2]
    prediction[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3]
    # prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # [total_anchors_num,4+1+1+1]
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def postprocess(prediction, num_classes, fc_outputs, conf_thre=0.001, nms_thre=0.5):
    '''
    prediction: list([self.memory_num,4+1+1(cls_score)+1(cls_pred)+num_cls])
    fc_output [B,memeory_num,num_cls]
    '''
    output = [None for _ in range(len(prediction))]
    output_ori = [None for _ in range(len(prediction))]
    prediction_ori = [pred.clone() for pred in prediction]# copy.deepcopy(prediction)
    if fc_outputs is not None:
        cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False)

    for i, detections in enumerate(prediction):
        # detections:[memory_num,4+1+1(cls_score)+1(cls_pred)+num_cls]
        if not detections.size(0):
            continue

        
        if fc_outputs is not None:
            detections[:, 5] = cls_conf[i].sigmoid()
            detections[:, 6] = cls_pred[i]
            # tmp_cls_score:[memeory_num,num_cls]
            tmp_cls_score = fc_outputs[i].sigmoid()
            cls_mask = tmp_cls_score >= conf_thre
            cls_loc = torch.where(cls_mask)
            # set t = len(cls_loc[0])
            # tmp_cls_score[cls_loc[0]]:[t,num_cls] 
            # cls_loc[1].unsqueeze(1):[t,1]
            # scores: [t,1] -> 对应于 cls_mask 的那些 score
            scores = torch.gather(tmp_cls_score[cls_loc[0]],dim=-1,index=cls_loc[1].unsqueeze(1))#[:,cls_loc[1]]#tmp_cls_score[torch.stack(cls_loc).T]#torch.gather(tmp_cls_score, dim=1, index=torch.stack(cls_loc).T)

            detections[:, -num_classes:] = tmp_cls_score
            detections_raw = detections[:, :7]
            new_detetions = detections_raw[cls_loc[0]]
            new_detetions[:, -1] = cls_loc[1] # vpred
            new_detetions[:,5] = scores.squeeze() # cls_score
            # detections_high:[t,7]
            detections_high = new_detetions  # new_detetions
            
            #print(len(detections_high.shape))

            # conf_mask: [t]
            conf_mask = (detections_high[:, 4] * detections_high[:, 5] >= conf_thre).squeeze()
            # detections_high: [sum(conf_mask),7]
            detections_high = detections_high[conf_mask]

            if not detections_high.shape[0]:
                continue
            if len(detections_high.shape)==3: # ???
                detections_high = detections_high[0]

            nms_out_index = torchvision.ops.batched_nms(
                detections_high[:, :4],
                detections_high[:, 4] * detections_high[:, 5],
                detections_high[:, 6],
                nms_thre,
            )
            # detections_high: [len(nms_out_index),7]
            detections_high = detections_high[nms_out_index]
            output[i] = detections_high

        # detections_ori: [memory_num,4+1+1(cls_score)+1(cls_pred)+num_cls]
        detections_ori = prediction_ori[i] # 
        detections_ori = detections_ori[:, :7]
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

    return output, output_ori
    
def show_box_on_img(img,boxes,cat_id2str_dict = None,mode='xyxy',delay = 0):
    '''
    img: can be show with opencv
    boxes: iterable of [x1,y1,x2,y2,category]
    '''
    if mode == 'cxcywh':
        boxes = cxcywh2xyxy(boxes)
        # print(boxes)
    if len(img.shape) == 3 and img.shape[2] != 3:
        img=img.transpose(1,2,0)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    
    for pred in boxes:
        
        box = [int(t) for t in pred[0:5]]
        # print(box)
        # print(cat_id2str_dict[box[4]])
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (204, 204, 51), 1)

        if len(pred) == 5:
            category = cat_id2str_dict[int(pred[4])] if cat_id2str_dict is not None else int(pred[4])
        elif len(pred) > 5:
            conf = pred[4] * pred[5]
            category = (cat_id2str_dict[int(pred[-1])] if cat_id2str_dict is not None else int(pred[-1])) + '|' + str(conf)
        else:
            category = ''
        
            
        font_color = (0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (fw, uph), dh = cv2.getTextSize(category, font, font_scale, thickness)
        cv2.rectangle(img, 
                    (box[0], box[1]-uph-dh), 
                    (box[0]+fw, box[1]), 
                    (204, 204, 51),
                    -1, 8)

        cv2.putText(img, 
                   category, 
                   (box[0], box[1]-dh), 
                   font, font_scale, 
                   (0, 0, 0), 
                   thickness)
                   
    cv2.imshow('test',img)
    cv2.waitKey(delay)
    


# ------------ 显存相关 ------------
 
def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        cuda_device = int(visible_devices[cuda_device])
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    为训练预分配gpu内存，避免内存碎片。
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    # block_mem 的 单位是 MB
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)
    
   

# -------------- checkpoint ------------------
# 这两个都融合进 trainer 里去了
def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    # 检查是否与 model 匹配
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt
    
    # 可能会有缺失的？！
    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best:bool, save_dir, model_name=""):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    logger.info('State will be saved as {}',filename)
    
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)

# -------------- evalute ------------------

def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
    
@contextlib.contextmanager
def adjust_status(module: nn.Module, training: bool = False) -> nn.Module:
    """Adjust module to training/eval mode temporarily.

    Args:
        module (nn.Module): module to adjust status.
        training (bool): training mode to set. True for train mode, False fro eval mode.

    Examples:
        >>> with adjust_status(model, training=False):
        ...     model(data)
    """
    status = {}

    def backup_status(module):
        for m in module.modules():
            # save prev status to dict
            status[m] = m.training
            m.training = training

    def recover_status(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = status.pop(m)

    backup_status(module)
    yield module
    recover_status(module)

# -------------- 其他------------------
def meshgrid(*tensors): # 版本向后兼容
    _TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)   


def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)

    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info
    

    