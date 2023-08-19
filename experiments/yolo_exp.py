#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# --Based on YOLOX made by Megavii Inc.--
import os
import random
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from experiments.base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- VOD config ---------------- #
        self.seq_len = 100
        self.memory = True
        
        
        self.cache = False
        
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 4
        # factor of model depth
        self.depth = 1.0
        # factor of model width
        self.width = 1.0
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        # --------------------------------------------------- #
        self.input_size = (640, 640)  # (height, width)
        # --------------------------------------------------- #
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5 
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.max_labels = 120
        
        # self.batch_size = 16
        
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = None # Path('D:/Liyi/Datasets/DETRAC').resolve(strict=True)
        # name of annotation file for training
        self.train_ann = "DETRAC-Train-Annotations-XML"# "instances_train2017.json"
        self.train_img_dir = 'Insight-MVT_Annotation_Train' # 'train2017'
        # name of annotation file for evaluation
        self.val_ann = "DETRAC-Test-Annotations-XML"
        self.val_img_dir = 'Insight-MVT_Annotation_Test'

        # # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        # self.data_dir = Path("./datasets/coco/coco2017").resolve(strict=True)
        # # name of annotation file for training
        # self.train_ann = "instances_train2017.json"# "instances_train2017.json"
        # self.train_img_dir = 'train2017' # 'train2017'
        # # name of annotation file for evaluation
        # self.val_ann = "instances_val2017.json"
        # self.val_img_dir = 'val2017'
        # # name of annotation file for testing
        # self.test_ann = "instances_test2017.json"
        # output dir. The default comes from BaseExp 
        self.output_dir = Path("output_files")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        # self.mosaic_prob = 1.0
        # prob of applying mixup aug
        # self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        # self.enable_mixup = True
        # self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 100
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 2
        # apply EMA during training
        self.ema = True
        

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 50
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        # --------------------------------------------------- #
        self.eval_interval = 10
        # --------------------------------------------------- #
        
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # --------------------------------------------------- #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # --------------------------------------------------- #
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.05
        # nms threshold
        self.nmsthre = 0.5

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
                        width=self.width,depth=self.depth,memory=self.memory,conf_thre=self.test_conf)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_data_loader(self, batch_size =16,mode = "train",mosaic = False):
        from data.data_utils import get_dataloader
        return get_dataloader(self,batch_size,mode=mode)


    def get_optimizer(self,batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from utils.lr_scheduler import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, testdev, legacy):
        
        return self.get_data_loader(batch_size=batch_size,mode = "val")
    
    def get_decoder(self):
        from utils.decoder import Decoder
        return Decoder()

    def get_evaluator(self, batch_size, testdev=False, legacy=False):
        from utils.coco_evaluater import COCOEvaluator
        
        val_loader = self.get_eval_loader(batch_size, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def get_trainer(self, args):
        from trainer import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    # def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        # return evaluator.evaluate(model, is_distributed, half, return_outputs=return_outputs)
        

