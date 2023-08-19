# -*- encoding:utf-8 -*-

from loguru import logger

import torch
from utils.functions import meshgrid


class Decoder():
    '''
    @param outputs_from_net:[[B,4+1+num_classes,H1,W2],[B,4+1+num_classes,H2,W2],[B,4+1+num_classes,H3,W3]]
    
    @return [B, total_anchors_num, 4+1+num_classes]
    '''
    def __init__(self,strides=[8, 16, 32]):
        self.strides = strides
    
    def __call__(self,outputs_from_net):
        # outputs_from_net: 
        # [[B,4+1+num_classes,H1,W2],[B,4+1+num_classes,H2,W2],[B,4+1+num_classes,H3,W3]]
        
        for output in outputs_from_net:
            output[:,4,:,:].sigmoid_()
            output[:,5:,:,:].sigmoid_()
        
        hws = [x.shape[-2:] for x in outputs_from_net]
        # [B,85,H,W] -> [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs_from_net], dim=2
        ).permute(0, 2, 1)
        
        return self._decode_outputs(outputs,hws)
        
        
    def _decode_outputs(self,outputs,hws):
        dtype=outputs.type()
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hws, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        
        # 预测的左上角（中心？）和宽高
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    
