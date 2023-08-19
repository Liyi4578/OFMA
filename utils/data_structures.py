#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import functools
import os
import time
from collections import defaultdict, deque

import numpy as np

import torch


class AverageMeter:
    """
    通过 update跟踪一个序列的值，并统计一个窗口或全部的序列平均值/中位数，来平滑整个序列值。
    Track a series of values and provide access to smoothed values over a
    window or the global series average.

    @param window_size：窗口大小，未指定最窗口无限大，默认50。
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        if isinstance(value,torch.Tensor):
            value = value.cpu()
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    """
    一个 dict ，用AverageMeter 作为每个元素的初始构造器。
    Computes and stores the average and current value
    
    """

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        '''
        返回 用键值 filter_key 过滤后的新字典
        '''
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()
