import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent).replace('/','\\'))
sys.path.append(str(Path(__file__).parent).replace('/','\\'))

from experiments.yolo_exp import Exp
from data.detrac_dataset import DetracDataset
from data.data_transform import TrainTransform,ValTransform
from data.data_prefetcher import DataPrefetcher

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)


# tranform 才有 need_size / input_size 
# dataset 不应该有？ 



def seq_collate_fn(batch):
    return batch

def get_dataloader(exp:Exp,batch_size,mode='train'):


    if mode == 'train':
        annp_dir = exp.train_ann
        img_dir = exp.train_img_dir
        transform = TrainTransform(max_labels=exp.max_labels,
                flip_prob=exp.flip_prob,hsv_prob=exp.hsv_prob,need_size=exp.input_size)
    elif mode == 'val' or mode == 'evaluate':
        annp_dir = exp.val_ann
        img_dir = exp.val_img_dir
        transform = ValTransform(need_size=exp.test_size)
    else:
        raise ValueError("mode should be 'train'/'val'/'evaluate' not be {}".format(mode))
    
    detracDataset = DetracDataset(exp.data_dir,annp_dir,img_dir,preproc=transform,seq_len=exp.seq_len,cache = exp.cache,mode=mode)
    sampler = None if mode == 'train' else torch.utils.data.SequentialSampler(detracDataset)
    shuffle = True if mode == 'train' else False
    
    data_loader = DataLoader(
                        detracDataset,
                        batch_size=batch_size,
                        drop_last=False,
                        shuffle=shuffle,
                        sampler = sampler,
                        num_workers=exp.data_num_workers ,
                        pin_memory=True,
                        collate_fn = seq_collate_fn,
                        worker_init_fn = worker_init_reset_seed,
                        # collate_fn = detection_collate
                        )
    # prefetcher 后面再用...
    # prefetcher = DataPrefetcher(data_loader)
    return data_loader
 
def test():
    from loguru import logger

    detracDataset = DetracDataset('D:/Liyi/Datasets/DETRAC',
                                anno_dir = 'DETRAC-Test-Annotations-XML',
                                seqs_dir = 'Insight-MVT_Annotation_Test',
                                preproc=TrainTransform(max_labels=50,
                                    flip_prob=0.5,hsv_prob=0.1,need_size=[214,360])
                                
                                )

    logger.info(len(detracDataset))
    train_loader = DataLoader(
                        detracDataset,
                        batch_size=2,
                        drop_last=False,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=True,
                        collate_fn = seq_collate_fn,
                        worker_init_fn = worker_init_reset_seed,
                        # collate_fn = detection_collate
                        )
    logger.info(len(train_loader))
    # return train_loader
    with torch.no_grad():
        seqs= next(iter(train_loader))
        for idx in range(len(seqs[0])):
            imgs_list = []
            labels_list = []
            for seq in seqs:
                img,label = seq[idx]
                imgs_list.append(img)
                labels_list.append(label)
            imgs = torch.stack(imgs_list,dim=0)
            targets = torch.stack(labels_list,dim=0)
        
        # print(inputs.shape)
        # print(targets)
        # nlabel = (targets.sum(dim=2) > 0).sum(dim=1)
        # print(nlabel)
        print(imgs.shape)
        print(targets.shape)


if __name__ == '__main__':
    test()