#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from pathlib import Path
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
# 用 cv2 是因为需要插值 resize....


class CocoDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir = '',
        json_file="instances_train2017.json",
        imgs_dirname="train2017",
        preproc=None,
        coco = None
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            imgs_dirname (str): COCO data imgs_dir (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.data_dir = data_dir
        if coco is not None:
            self.coco = coco
        elif len(json_file) > 0:
            self.json_file = json_file
            self.coco = COCO(self.json_file)
        else:
            raise RuntimeError('where is the data?')
            return
        
        self._remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        # category
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = tuple([c["name"] for c in self.cats]) # 
        # self.imgs = None # 没这么大的内存去将数据集放RAM中....
        self.imgs_dirname = imgs_dirname
        self.preproc = preproc
        # list of (list of [x1,y1,x2,y2,category_id], img_info, resized_info, file_name)
        self.annotations = self._load_coco_annotations()

    def _remove_useless_info(self,coco):
        """
        Remove useless info in coco dataset. COCO object is modified inplace.
        This function is mainly used for saving memory (save about 30% mem).
        """
        if isinstance(coco, COCO):
            dataset = coco.dataset
            dataset.pop("info", None)
            dataset.pop("licenses", None)
            for img in dataset["images"]:
                img.pop("license", None)
                img.pop("coco_url", None)
                img.pop("date_captured", None)
                img.pop("flickr_url", None)
            if "annotations" in coco.dataset:
                for anno in coco.dataset["annotations"]:
                    anno.pop("segmentation", None)
        



    def __len__(self):
        return len(self.ids)

    
    def _load_coco_annotations(self):
        # 每张图片都有一个 tuple 的 annotation
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]


    def load_anno_from_ids(self, id_):
        # 相比原来去掉了 resized_info ，标签缩放与图片缩放都在 _resize_img_and_labels
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []

        # 读取标签，得到 box 的左上角点与右下角点(原json文件是[xmin,ymin,w,h]格式)
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        labels = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            labels[ix, 0:4] = obj["clean_bbox"]
            labels[ix, 4] = cls
        img_hw = (height, width)
        file_name =  im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
            
        return (labels,img_hw,file_name)

    def load_anno(self, index):
        
        return self.annotations[index][0]


    def load_image(self, file_name):
        '''
        读取图片
        默认使用 opencv 的 imread 读取，颜色 BGR，shape[H,W,3]
        '''
        img_file_path = os.path.join(self.data_dir, self.imgs_dirname, file_name)
        img = cv2.imread(img_file_path)
        assert img is not None, f"file named {img_file_path} not found"
        
        # # Tensor[image_channels, image_height, image_width] uint8
        # img = read_image(img_file_path) 
        return img


    def __getitem__(self, index):
        """
        通过给定 index 获取 图片与标签即图片COCO的id
        
        
        """
        if index > len(self) - 1:
            raise IndexError("index {} is out of range".format(index))
        img_id = np.array([self.ids[index]]) # 获取图片的 id ndarray([id])

        labels,img_hw, file_name = self.annotations[index]
        
        file_names = file_name.split('/')
        file_name = file_names[-2] + '/' + file_names[-1]
        logger.info('[{}][{}][{}]',labels,img_hw, file_name)
        img = self.load_image(file_name)
        
        # 图片缩放放至 preproc 中
        # img,labels = _resize_img_and_labels(img,labels)
        
        if self.preproc is not None:
            img, labels = self.preproc(img, labels)
        return img, labels, np.array(img_hw),img_id
        


def test_coco():
    data_dir ='D:/Liyi/Datasets/DETRAC'
    jsonfile = "coco_anno.json"
    coco = COCO(os.path.join(data_dir, "annotations", jsonfile))
    
    ids = coco.getImgIds()
    print('ids',len(ids),' ',ids[0])
    class_ids = sorted(coco.getCatIds())
    print('class_ids',len(class_ids),' ',class_ids[0])
    cats = coco.loadCats(coco.getCatIds())
    print('cats',len(cats),cats[0])
    classes = tuple([c["name"] for c in cats])    
    print(classes[cats[0]['id']-1])
    
    test_id = ids[0]
    im_ann = coco.loadImgs(test_id) #dict
    print('im_ann',len(im_ann),' ',im_ann[0])
    im_ann = im_ann[0]
    width = im_ann["width"]
    height = im_ann["height"]
    # 一张图片对应多个标签
    anno_ids = coco.getAnnIds(imgIds=[int(test_id)], iscrowd=False)
    print('anno_ids',anno_ids)
    annotations = coco.loadAnns(anno_ids)
    print('annotations',len(annotations),annotations[2])
    print('dataset',len(coco.dataset['annotations']),coco.dataset['annotations'][0]['category_id'])
    
    print('-----------------------------------------------------------------------')
 
def test():
    from torchvision.utils import make_grid
    from data_transform import TrainTransform

    print(__file__)
    data_dir = 'D:/Liyi/Datasets/DETRAC'
    jsonfile = 'test.json'# "D:/Liyi/Datasets/DETRAC/DETRAC-Train-Annotations-XML/coco_anno.json"
    cocoDataset = CocoDataset(data_dir,jsonfile,"Insight-MVT_Annotation_Train")
    
    # print(cocoDataset.cats)
    preproc = TrainTransform(need_size=(640,640),
                        max_labels=120,
                        flip_prob=0.5,
                        hsv_prob=0.5)
    
    import sys
    import copy
    sys.path.append(str(Path(__file__).parent.parent).replace('/','\\'))
    from utils.functions import show_box_on_img,cxcywh2xyxy




    
    for idx in np.random.randint(low=0,high=len(cocoDataset),size=(10,)):
        orig_img,orig_targets,img_hw,_ = cocoDataset[idx]
        
        img, targets = preproc(copy.deepcopy(orig_img), copy.deepcopy(orig_targets))
        logger.info(orig_img.shape)
        logger.info(img.shape)
        logger.info(orig_targets)
        img_file_name = cocoDataset.annotations[idx][2]
        logger.info(img_file_name)
        # print(cocoDataset.annotations[idx][0])
        # show_box_on_img(img.astype(np.uint8),targets,cocoDataset.classes,mode='xyxy')  
        # 数据增强后变为了 cxcywh
        # print(targets)
        logger.info('cxcywh\n {}',targets[targets.sum(1)>0])

        img = img.numpy().astype(np.uint8).transpose(1,2,0)

        # 这之前应该是没问题的
        show_box_on_img(img,targets,cocoDataset.classes,mode='cxcywh')
        # target:[x1,y1,x2,y2,category]
  
        # 缩放逆运算
        logger.info('xyxy\n {}',targets[targets.sum(1)>0])
        # opencv img : [h,w,3]

        scale = min(
                img.shape[1] / float(img_hw[0]), img.shape[2] / float(img_hw[1])
            )
        start_x = int((img.shape[2] - int(img_hw[1]*scale)) / 2)
        start_y = int((img.shape[1] - int(img_hw[0]*scale)) / 2)
        logger.info('{},{},{}',start_x,start_y,scale)
        logger.info('scale orig xyxy\n {}',orig_targets*scale)
        # 这一步仅缩放时图片靠左靠上才行，居中不行
        targets[targets.sum(1)>0,0:4:2] -=  start_x
        targets[targets.sum(1)>0,1:4:2] -=  start_y
        targets[:,:4] /= scale
        logger.info(targets[targets.sum(1)>0])
        # show_box_on_img(orig_img,targets,cocoDataset.classes,mode='xyxy')
        logger.info('-'*42)
 
 
if __name__ == "__main__":
    test()
    
    
