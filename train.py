# -*- coding:utf-8 -*-
import argparse
import sys
import os
from loguru import logger
import importlib

import torch

from trainer import Trainer


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    
    # 输出文件夹的名称
    parser.add_argument("-expn", "--experiment-name", type=str, default=None,help="name of output directory")
    parser.add_argument("-b", "--batch-size", type=int, default=0, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=0, type=int, help="device for training"
    )
    # 实验名称 对应 exp
    parser.add_argument("-n", "--exp-name", type=str, default='None', help="model name")

    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )

    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )

    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def get_exp(exp_name):
        
    if exp_name.lower() == 'yolox':
        from experiments.yolo_exp import Exp
    elif exp_name.lower() == 'yolox-m':
        from experiments.yolox_m import Exp
    elif exp_name.lower() in ['normal_yoloxs','yolox_s_detrac'] :
        from experiments.normal_yoloxs import Exp
    elif exp_name.lower() in ['ours','ours_detrac'] :
        from experiments.ours_exp import Exp
    elif exp_name.lower() in ['test','test_exp'] :
        from experiments.test_exp import Exp
    else:
        try:
            sys.path.append('experiments')
            current_exp = importlib.import_module(os.path.basename(exp_name).split(".")[0])
            Exp = current_exp.Exp
        except Exception:
            raise ImportError("{} doesn't contains class named 'Exp'".format(exp_name))
    return Exp()
    
    
def get_trainer(exp_name):
    logger.remove()
    logger.add(sys.stdout, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="INFO",enqueue=True)
    
    args = make_parser().parse_args([])
    
    args.exp_name = exp_name
    print("args:\n{}".format(args))
    exp = get_exp(exp_name)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
        
    trainer = Trainer(exp,args)
    
    return trainer    
        

# python train.py --opts input_size (320,320) batch_size 2 data_dir
def main():
    logger.remove()
    logger.add(sys.stdout, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="INFO",enqueue=True)
    logger.add(sys.stderr, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="WARNING",enqueue=True)
    args = make_parser().parse_args()
    logger.info("args:\n{}".format(args))
    
    if args.exp_name in [None,'None']:
        args.exp_name = 'ours_exp'# "normal_yoloxs"
    exp = get_exp(args.exp_name)

    exp.merge(args.opts)
    if args.batch_size == 0:
        args.batch_size = exp.batch_size
        
    args.fp16 = True
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    # 会导致推理结果不一样？
    # torch.backends.cudnn.benchmark = True 
    trainer = Trainer(exp,args)
    trainer.train()
    


    
def for_kaggle():
    logger.remove()
    logger.add(sys.stdout, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="INFO",enqueue=True)
    logger.add(sys.stderr, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="WARNING",enqueue=True)
                    
    class Args:
        experiment_name=None
        batch_size=64
        devices=0
        exp_name='None'
        resume=False
        ckpt=None
        start_epoch=None
        fp16=False
        occupy=False
        opts=[]
        
    args = Args()
    args.exp_name = "normal_yoloxs"
    exp = get_exp(args.exp_name)
    exp.merge(args.opts)
    args.batch_size = exp.batch_size
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    print("input:\n{}".format(args))
    
    return exp,args
    
    
if __name__ == "__main__":
    main()