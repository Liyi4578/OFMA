# -*- coding:utf-8 -*-
import copy
import sys
import os
from experiments.normal_yoloxs import Exp
import torch
import numpy as np
from loguru import logger
from utils.functions import get_model_info,gpu_mem_usage,show_box_on_img
from data.detrac_dataset import DetracDataset
from data.data_transform import ValTransform


def main():
    logger.remove()
    logger.add(sys.stdout, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="INFO",enqueue=True)
    logger.add(sys.stderr, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="WARNING",enqueue=True)
    logger.add(eval_log_file,format="{time:YYYY-MM-DD HH:mm:ss} {level} {file}:{line}: \n{message}",level="INFO",enqueue=True)
    
    
    exp = Exp()
    exp.test_conf = 0.05 # Can't be too small,otherwise it will be jam(ka si)
    evaluator = exp.get_evaluator(
        batch_size=128
    )
    
    model = exp.get_model()
    model = model.to('cuda')
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    # logger.info("Model Structure:\n{}".format(str(model)))
    
    model.eval()

    ckpt = torch.load('latest_ckpt.pth', map_location='cuda:0')
    model.load_state_dict(ckpt["model"])
    
    evaluator.evaluate(model,half = False,return_outputs = False)
    logger.info("-------ap50:{},ap50_95:{}-------",ap50,ap50_95)
    logger.info("\n" + summary)
    

if __name__ == '__main__':
    main()