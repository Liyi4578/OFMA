#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --Based on YOLOX made by Megavii Inc.--
import datetime
import os
import shutil
import sys
import time
from loguru import logger
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.data_prefetcher import DataPrefetcher
from experiments.yolo_exp import Exp

from utils.functions import get_model_info,occupy_mem,adjust_status,gpu_mem_usage,load_ckpt
from utils.data_structures import MeterBuffer
from utils.ema_model import ModelEMA




class Trainer:
    def __init__(self, exp: Exp, args):

        self.exp = exp
        self.args = args


        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
        self.no_aug = True
        
        # multi gpus
        # self.is_distributed = get_world_size() > 1
        # self.rank = get_rank()
        # self.local_rank = get_local_rank()
        
        self.device = "cuda" # "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.train_size = self.input_size # 可能会随机改变
        self.best_ap = 0

        # metric record
        # 平滑字典
        self.meter = MeterBuffer(window_size=exp.print_interval)
        
        self.dir_name = os.path.join(exp.output_dir, args.experiment_name)
        # if self.rank == 0:
            # os.makedirs(self.file_name, exist_ok=True)
        os.makedirs(self.dir_name, exist_ok=True)
        
        # 保存日志
        train_log_file = os.path.join(self.dir_name, "log_train_{time:YY_MM_DD_HH_mm_ss}.txt")
        # logger.add(sys.stdout,format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",level="INFO",enqueue=True)
        # 
        logger.add(train_log_file,format="{time:YYYY-MM-DD HH:mm:ss} {level} {file}:{line}: \n{message}",level="INFO",enqueue=True)

        
    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()
    
    def before_train(self):
        logger.info("args: {}",self.args)
        logger.info("exp:\n{}",self.exp)

        # model related init
        # torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        model.to(self.device)
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)
        
        self.decode = self.exp.get_decoder()
        
        model.head.use_l1 = True # 开头就使用 L1 损失

        # data related init
        self.no_mosaic = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        # dataloader 其实已经是 prefetcher 了
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            mode = 'train',
            mosaic= not self.no_mosaic,
        )
        # logger.info("init prefetcher, this might take one minute or less...")
        # self.prefetcher = DataPrefetcher(self.train_loader)
        
        # iters_per_epoch means iters per epoch
        # max_ter in YOLOX
        self.iters_per_epoch = len(self.train_loader)*self.exp.seq_len # * 100

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.iters_per_epoch
        )
        if self.args.occupy:
            occupy_mem(0) # occupy_mem(self.local_rank)

        # if self.is_distributed:
            # model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.iters_per_epoch * self.start_epoch

        self.model = model
        
        logger.info("init get_evaluator...")
        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size // 2
        )

        # if self.rank == 0:
        self.tblogger = SummaryWriter(os.path.join(self.dir_name, "tensorboard"))

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )


    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_one_epoch()
            self.after_epoch()
    
    
    def before_epoch(self):
        logger.info("---> start train epoch [{}] <---".format(self.epoch + 1))
        self.epoch_start_time = time.time()
        # 是否加入 mosaic 与 l1


    def after_epoch(self):
        epoch_need_time = datetime.timedelta(seconds=int(time.time()-self.epoch_start_time))
        logger.info("---> finish train epoch{},used time:{} <---".format(self.epoch + 1,epoch_need_time))
        self.save_ckpt(ckpt_name="latest") # 用于恢复训练
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")
        # 在适当的间隔处评估并保存
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            self.evaluate_and_save_model()
            
    @property
    def progress_in_iter(self):
        return self.epoch * self.iters_per_epoch + self.iter
        
    def train_one_epoch(self):
        # torch.autograd.set_detect_anomaly = True
        
        torch.autograd.set_detect_anomaly = True
        self.model = self.model.train()
        # not need
        torch.cuda.empty_cache()  
        self.iter = 1
        for seqs in self.train_loader: # self.prefetcher:
            
            # test for code
            # if self.iter > 50:
                # break
            self.model.head.clear_memory()

            for idx in range(len(seqs[0])):
                iter_start_time = time.time()
                
                imgs_list = []
                labels_list = []
                for seq in seqs:
                    img,label = seq[idx]
                    imgs_list.append(img)
                    labels_list.append(label)
                imgs = torch.stack(imgs_list,dim=0)
                targets = torch.stack(labels_list,dim=0)

                imgs = imgs.type(self.data_type).to(self.device)
                targets = targets.type(self.data_type).to(self.device)
                targets.requires_grad = False
                # imgs, targets = self.resize(imgs, targets)
                
                data_end_time = time.time()
                # with torch.autograd.detect_anomaly():
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    losses = self.model(imgs,targets)
                loss = losses["total_loss"]
                
                self.optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():

                self.grad_scaler.scale(loss).backward() # retain_graph=True
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                if self.use_model_ema:
                    self.ema_model.update(self.model)
                
                # 更新学习率
                lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                iter_end_time = time.time()
                self.meter.update(
                    iter_time=iter_end_time - iter_start_time,
                    data_time=data_end_time - iter_start_time,
                    lr=lr,
                    **losses,
                )
                self.log_one_iter()
                iter_start_time = time.time()
                self.iter += 1

    def log_one_iter(self):
        """
        # log_one_iter == after_iter in YOLOX
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.iters_per_epoch * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.iters_per_epoch
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )
            self.tblogger.add_scalar("train/total_loss", loss_meter['total_loss'].avg, self.iters_per_epoch * self.epoch + self.iter + 1)
            self.tblogger.add_scalar("train/iou_loss", loss_meter['iou_loss'].avg, self.iters_per_epoch * self.epoch + self.iter + 1)
            self.tblogger.add_scalar("train/l1_loss", loss_meter['l1_loss'].avg, self.iters_per_epoch * self.epoch + self.iter + 1)
            self.tblogger.add_scalar("train/cls_loss", loss_meter['cls_loss'].avg, self.iters_per_epoch * self.epoch + self.iter + 1)
            self.tblogger.add_scalar("train/conf_loss", loss_meter['conf_loss'].avg, self.iters_per_epoch * self.epoch + self.iter + 1)
            
            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.train_size = self.get_random_size()
            
    def get_random_size(self):
        tensor = torch.LongTensor(2).cuda()
        # if rank == 0:
        size_factor = self.input_size[1] * 1.0 / self.input_size[0]
        if not hasattr(self, 'random_size'):
            min_size = int(self.input_size[0] / 32) - self.exp.multiscale_range
            max_size = int(self.input_size[0] / 32) + self.exp.multiscale_range
            self.random_size = (min_size, max_size)
        size = random.randint(*self.random_size)
        size = (int(32 * size), 32 * int(size * size_factor))
        tensor[0] = size[0]
        tensor[1] = size[1]
        
        train_size = (tensor[0].item(), tensor[1].item())
        return train_size
        
    def resize(self,inputs,targets):
        scale_y = self.train_size[0] / self.input_size[0]
        scale_x = self.train_size[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=self.train_size, mode="bilinear", align_corners=False
            )
            targets[..., 0:4:2] = targets[..., 0:4:2] * scale_x
            targets[..., 1:4:2] = targets[..., 1:4:2] * scale_y
        return inputs, targets


    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training ...")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.dir_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            self.start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            
            logger.info("loaded checkpoint '{}' (epoch {})",self.args.resume, self.start_epoch)
             
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning [{}]".format(self.args.ckpt))
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = self.load_ckpt(model, ckpt)
            if self.exp.ckpt is not None:
                logger.info("loading checkpoint for fine tuning [{}] (exp)".format(self.exp.ckpt))
                ckpt_file = self.exp.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = self.load_ckpt(model, ckpt)
            self.start_epoch = 0
            
        return model

    def evaluate_and_save_model(self):

        logger.info("Start evaluate ...")
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
        
        # 将 model 暂时 变为 非 training状态
        with adjust_status(evalmodel, training=False):
            # (ap50_95, ap50, summary), predictions = self.exp.eval(
                # evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            # )
            # 使用 exp.eval ->直接用 self.evaluator
            # return_outputs 这里似乎可以设为 False
            #(ap50_95, ap50, summary), predictions = self.evaluator.evaluate(evalmodel,decoder=self.decode,half = self.args.fp16,return_outputs = True)
            
            # half = self.args.fp16
            self.evaluator.evaluate(evalmodel,decoder=self.decode,half = False,return_outputs = False,epoch=self.epoch+1)

        # update_best_ckpt = ap50_95 > self.best_ap
        # self.best_ap = max(self.best_ap, ap50_95)
        self.best_ap = 0.0
        # if self.rank == 0:
        # self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
        # self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
        # logger.info("\n" + summary)
        # logger.info("-------ap50:{},ap50_95:{}-------",ap50,ap50_95)
        # Helper function to synchronize (barrier) among all processes when using distributed training
        # synchronize()
        # 保存最新的并检查是否最好
 
        self.save_ckpt("last_epoch", False)
        

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):

        save_model = self.ema_model.ema if self.use_model_ema else self.model
        # logger.info("Save state to {}".format(self.dir_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_ap": self.best_ap,
            "curr_ap": ap,
        }
        
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        filename = os.path.join(self.dir_name, ckpt_name + "_ckpt.pth")
        logger.info('State will be saved as {}',filename)
        
        torch.save(ckpt_state, filename)
        if update_best_ckpt:
            best_filename = os.path.join(self.dir_name, "best_ckpt.pth")
            shutil.copyfile(filename, best_filename)
    
    def load_ckpt(self,model,ckpt):
        
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
