# -*- coding:utf-8 -*-
# --Based on YOLOX made by Megavii Inc.--
from experiments.yolo_exp import Exp as MyExp

import os

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        