#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

import math
from loguru import logger
# 参考 yolov5/3/7，act 默认为 True 即为 SiLU，若需要其他直接传入 nn.module


class BaseConv(nn.Module):
    """
    A Conv2d -> Batchnorm -> act block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act=True):
    
        super().__init__()
        # same padding
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=True):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act=True,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, kernel_size=1, stride=1, act=nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, kernel_size=3, stride=1, act=nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation=True
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels,kernel_size=1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act=True,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, kernel_size=1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act=True):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size=kernel_size, stride=stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class TBLFAM(nn.Module):
    '''
    Temporal Box-level Feature Aggregation Module 
    '''
    def __init__(self,num_heads = 8, num_channels = 256,out_channels = 1024, qkv_bias=False, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        # self.qkv = nn.Linear(num_channels, num_channels * 3, bias=qkv_bias)
        self.qkv_cls = nn.Linear(num_channels, num_channels * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(num_channels, num_channels * 3, bias=qkv_bias)

        self.qkv_m_cls = nn.Linear(num_channels, num_channels * 3, bias=qkv_bias)
        self.qkv_m_reg = nn.Linear(num_channels, num_channels * 3, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)

        self.scale = 1. #math.sqrt(num_channels // num_heads) #torch.sqrt(torch.Tensor(num_heads // num_heads)) 

        self.linear1 = nn.Linear(2 * num_channels, 2 * num_channels)
        self.linear2 = nn.Linear(4 * num_channels, out_channels)
        
        self.update_cls_linear = nn.Sequential(*[
            nn.Linear(2 * num_channels, 4 * num_channels),
            nn.Linear(4 * num_channels, num_channels)
        ])
        self.update_reg_linear = nn.Sequential(*[
            nn.Linear(2 * num_channels, 4 * num_channels),
            nn.Linear(4 * num_channels, num_channels)
        ])


    def forward(self, x_cls,x_reg, memory_cls,memory_reg, cls_score=None, fg_score=None,
                use_mask = False,sim_thresh=0.75):
        
        B, N, C = x_cls.shape
        self.device = x_cls.device
        # qkv_cls:[3,1,self.num_heads,B*YOLOX_head.simN,um_channel(256) / self.num_head(=256/4=64)]
        # 3, B, num_head, N, c/num_head
        # qkv_com = 
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        qkv_pre_cls = self.qkv_m_cls(memory_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        qkv_pre_reg = self.qkv_m_reg(memory_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        del memory_cls
        del memory_reg

        # q_cls: [B,num_head,N, c/num_head]
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2] 
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]
        q_pre_cls, k_pre_cls, v_pre_cls = qkv_pre_cls[0], qkv_pre_cls[1], qkv_pre_cls[2] 
        q_pre_reg, k_pre_reg, v_pre_reg = qkv_pre_reg[0], qkv_pre_reg[1], qkv_pre_reg[2]


        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

        q_pre_cls = q_pre_cls / torch.norm(q_pre_cls, dim=-1, keepdim=True)
        k_pre_cls = k_pre_reg / torch.norm(k_pre_cls, dim=-1, keepdim=True)
        q_pre_reg = q_pre_reg / torch.norm(q_pre_reg, dim=-1, keepdim=True)
        k_pre_reg = k_pre_reg / torch.norm(k_pre_reg, dim=-1, keepdim=True)
        v_pre_cls_normed = v_pre_cls / torch.norm(v_pre_cls, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else: 
            # cls_score: [B,N] ->  [B,num_head,N,N]
            cls_score = torch.reshape(cls_score, [B, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [B, 1, 1, -1]).repeat(1, self.num_heads, N, 1)


        if use_mask:
            # only reference object with higher confidence.
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # -----------------------------------
        # 微调分类特征
        # -----------------------------------

        # attn_cls: [3, B, num_head, N, N]
        attn_cls = (q_cls @ k_pre_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_pre_reg.transpose(-2, -1)) * self.scale * fg_score * fg_score_mask
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        # [B, num_head, N, N] @ [B, num_head, N,  C/num_head] = [B, num_head, N, C/num_head]
        # -> [B,N,num_head, C/num_head] -> [B,N,C]
        x = (attn @ v_pre_cls).transpose(1, 2).reshape(B, N, C)

        # [B,N,num_heads,C/num_heads] -> [B,N,C]
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        # [B,N,num_heads,C*2]
        refined_cls = torch.cat([x, x_ori], dim=-1) 
        
        msa = self.linear1(refined_cls)
        ori_feat = msa.clone()
        temp_feat =  msa

        # [B,num_head,N, c/num_head]
        attn_cls_raw = v_cls_normed @ v_pre_cls_normed.transpose(-2, -1)
        ones_matrix = torch.ones(attn.shape[2:]).to(self.device)
        zero_matrix = torch.zeros(attn.shape[2:]).to(self.device)
        # attn_cls_raw : [1,self.num_heads,B*YOLOX_head.simN,B*YOLOX_head.simN] -> [1,1,B*YOLOX_head.simN,B*YOLOX_head.simN]
        attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads

        sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
        # sim_attn : [1,1,B*YOLOX_head.simN,B*YOLOX_head.simN]
        sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

        sim_round2 = torch.softmax(sim_attn, dim=-1)
        sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)).clamp(min=1e-6)
 
        soft_sim_feature = (sim_round2@temp_feat)#.transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature,ori_feat],dim=-1)

        out_feat = self.linear2(cls_feature)

        # -----------------------------------
        # 更新记忆
        # -----------------------------------
        # mem_attn_cls = (k_pre_cls @ k_cls.transpose(-2, -1)) * self.scale
        mem_attn_cls = (q_pre_cls @ k_cls.transpose(-2, -1)) * self.scale
        mem_attn_cls = attn_cls.softmax(dim=-1)
        mem_attn_cls = self.attn_drop(attn_cls)

        mem_attn_reg = (q_pre_reg @ k_reg.transpose(-2, -1)) * self.scale
        mem_attn_reg = attn_reg.softmax(dim=-1)
        mem_attn_reg = self.attn_drop(attn_reg)

        mem_x_cls = (mem_attn_cls @ v_cls).transpose(1, 2).reshape(B, N, C) 
        mem_x_ori_cls = v_pre_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        # [B,N,num_heads,C*2]
        refined_update_cls = torch.cat([mem_x_cls, mem_x_ori_cls], dim=-1) 

        mem_x_reg = (mem_attn_reg @ v_reg).transpose(1, 2).reshape(B, N, C) 
        mem_x_ori_reg = v_pre_reg.permute(0, 2, 1, 3).reshape(B, N, C)
        refined_update_reg = torch.cat([mem_x_reg, mem_x_ori_reg], dim=-1) 

        memory_cls_ = self.update_cls_linear(refined_update_cls)
        memory_reg_ = self.update_reg_linear(refined_update_reg)

        

        return out_feat,memory_cls_,memory_reg_
    



def test():

    x_cls = torch.randn(2,100,256).cuda()
    x_reg = torch.randn(2,100,256).cuda()
    memory_cls = torch.randn(2,100,256).cuda()
    memory_reg = torch.randn(2,100,256).cuda()

    net = TBLFAM().cuda()

    out_feat,memory_cls,memory_reg = net(x_cls,x_reg,memory_cls,memory_reg)

    print(out_feat.shape)
    print(memory_cls.shape)
    print(memory_reg.shape)


if __name__ == '__main__':
    test()

