# -*- coding:utf-8 -*-
# --Based on YOLOX made by Megavii Inc.--
import os
from pathlib import Path

import torch.nn as nn

from experiments.yolo_exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.memory = False
        self.heads = 4
        self.memory_num=100
        
        self.ckpt = 'yolox_s.pth'
        
        self.cache = False
        self.seq_len = 42

        # factor of model depth
        self.depth = 0.33
        # factor of model width
        self.width = 0.50
        self.depethwise = False
        
        self.input_size = (480, 640)  # (height, width)
        self.test_size = (480, 640)# (480, 640)
        
        self.data_num_workers = 4
        
        self.batch_size = 128
        self.max_epoch = 30
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.005 / self.batch_size
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = Path('').resolve(strict=True)
        # name of annotation file for training
        self.train_ann = "DETRAC-Train-Annotations-XML"# "instances_train2017.json"
        self.train_img_dir = 'Insight-MVT_Annotation_Train' # 'train2017'
        # name of annotation file for evaluation
        self.val_ann = "DETRAC-Test-Annotations-XML"
        self.val_img_dir = 'Insight-MVT_Annotation_Test'
        
        self.multiscale_range = 1
        self.eval_interval = 1
        self.print_interval = 100
        self.save_history_ckpt = True


    def get_model(self):
    
        from nets.YOLOX import YOLOXNet

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            self.model = YOLOXNet(self.num_classes,in_channels=in_channels,
                                width=self.width,depth=self.depth,depthwise=self.depethwise,memory=self.memory,
                                heads=self.heads,memory_num=self.memory_num)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model


