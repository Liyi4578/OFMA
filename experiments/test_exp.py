# -*- coding:utf-8 -*-
# --Based on YOLOX made by Megavii Inc.--
import os
from pathlib import Path

import torch.nn as nn

from experiments.ours_exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.memory = True
        self.heads = 4
        self.memory_num=100

        self.input_size = (320, 320)  # (height, width)
        self.test_size = (320, 320)# (480, 640)
        
        self.data_num_workers = 4
        
        self.data_dir = Path('D:/Liyi/Datasets/DETRAC').resolve(strict=True)
        
        self.batch_size = 2


