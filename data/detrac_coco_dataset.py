# -*- coding:utf-8 -*-

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent).replace('/','\\'))
sys.path.append(str(Path(__file__).parent).replace('/','\\'))

import json
from loguru import logger
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
# 用 cv2 是因为需要插值 resize....
from pycocotools.coco import COCO

from utils.functions import cxcywh2xyxy
from data.data_transform import TrainTransform,ValTransform

'''
DETRAC
├── DETRAC-Test-Annotations-XML
│   ├── MVI_39031.xml
│   └── MVI_39051.xml
└── Insight-MVT_Annotation_Test
    ├── MVI_39031
    │   ├── img00001.jpg
    │   ├── img00002.jpg
    │   └── ...
    ├── MVI_39051
    │   ├── img00001.jpg
    │   ├── img00002.jpg
    │   └── ...
    └── MVI_39051
 25 frames per seconds (fps), with resolution of 960x540 pixel
'''

def load_image(file_name):
    '''
    读取图片
    默认使用 opencv 的 imread 读取，颜色 BGR,shape[H,W,3]
    '''
    if isinstance(file_name,Path):
        file_name = str(file_name)
    img = cv2.imread(file_name)
    assert img is not None, f"file named {file_name} not found"
    
    # # Tensor[image_channels, image_height, image_width] uint8
    # img = read_image(img_file_path) 
    return img


class Seq():
    def __init__(self,seq_imgs,labels_list,preproc = None,ids_list = [],all_imgs = None,ignored_regions=None):
        self.seq_imgs = seq_imgs
        self.labels_list = labels_list
        self.idx = 0
        self.preproc = preproc
        self.ids_list = ids_list
        self.length = len(self.seq_imgs)
        self.all_imgs = all_imgs
        self.ignored_regions = ignored_regions
    def __len__(self):
        return self.length
    
    
    def _load_data(self,index):
        file_name = self.seq_imgs[index]
        if self.all_imgs is None:
            img = load_image(file_name)
        else:
            img = self.all_imgs[file_name]
        label = self.labels_list[index]
        
        # 遮住忽略区域？
        if self.ignored_regions is not None:
            ave = img.mean()
            for ignored in self.ignored_regions:
              img[ignored[1]:ignored[3],ignored[0]:ignored[2],:] = ave

        
        if self.preproc is not None:
            img, label = self.preproc(img, label)
            
        if len(self.ids_list) > 0:
            anno_id = self.ids_list[index]
            return img, label, anno_id,file_name
        else:
            return img, label
    
    
    def __getitem__(self, index):
        return self._load_data(index);
        
    def __iter__(self):
        # 由于自己就是一个迭代器，所以返回自己就行
        return self

    def __next__(self):

        if self.idx >= len(self.seq_imgs):
            raise StopIteration
        else:
            self.idx += 1
            return self._load_data(self.idx);


class DetracDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir, # 'xxx/xxx/DETRAC'
        anno_dir = 'DETRAC-Train-Annotations-XML',
        seqs_dir = 'Insight-MVT_Annotation_Train',
        seq_len = 18,
        preproc = None,
        cache = False,
        mode = 'train',
        coco_anno_dir = ""
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
        self.data_dir = Path(data_dir)
        self.anno_dir = self.data_dir / anno_dir
        self.seqs_dir = self.data_dir / seqs_dir
        self.mode = mode
        # self.test_anno_dir = data_dir / 'DETRAC-Test-Annotations-XML'
        # self.train_anno_dir = data_dir / 'DETRAC-Train-Annotations-XML'
        # self.test_data_dir = data_dir / 'Insight-MVT_Annotation_Test'
        # self.train_data_dir = data_dir / 'Insight-MVT_Annotation_Train'
        self.cls_idx = {'car':0, 'bus':1, 'others':2, 'van':3}
        self.idx2cls = ['car', 'bus', 'others', 'van']
        self.preproc = preproc
        
        
        self.coco_anno_dir = Path(coco_anno_dir)
            
            
        self.seq_len = seq_len
        self.all_annonations,imgs_count = self.read_all_anno()
        self.len = imgs_count // self.seq_len
        
        self.cache = cache
        self.all_imgs =None
        
        logger.info('load [{}] dataset from {}.',self.mode,self.anno_dir)
        if self.mode in ['val']:
            self._prepare_coco()
            
        if cache:
            self.all_imgs = {}
            self._cache_all()
            
        self._get_seqs_map()


    def _prepare_coco(self):
        json_filename = self.coco_anno_dir / "coco_anno.json"
        self.convert_to_json(json_filename)
        self.coco = COCO(json_filename)
        self.ids = self.coco.getImgIds()
        # category
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = tuple([c["name"] for c in self.cats]) # 
        self.annotations = self._load_coco_annotations()

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

    def read_xml(self,xmlfile):
        '''
        
        return:
            frames: list(frame), 
                    frame: {
                            'density': '2', 
                            'num': '1120', 
                            'targets': [
                                    np.array([x1,y1,x2,y2,cls_idx]) # {'id': 29, 'box': [x1,y1,x2,y2,cls_idx]},
                                    ... 
                                    ]
                            }
            name: 'MVI_39031'
            ignored_region: list([float(left),top,width,height],)
            sequence_attribute
        '''
        detrac_width = 960
        detrac_height = 540


        tree = ET.parse(xmlfile)
        root = tree.getroot()
        name = root.attrib['name']
        idx = 0
        if root[idx].tag == 'sequence_attribute':
            camera_state = root[idx].attrib['camera_state']
            sence_weather = root[idx].attrib['sence_weather']
            idx += 1
        else:
            camera_state = None
            sence_weather = None
        sequence_attribute = {'camera_state':camera_state,'sence_weather':sence_weather}
        ignored_region = []
        if root[idx].tag == 'ignored_region':
            for child in root[idx]:
                box = [
                        float(child.attrib['left']),
                        float(child.attrib['top']),
                        float(child.attrib['width']),
                        float(child.attrib['height'])
                    ]
                x1 = max(0, box[0])
                y1 = max(0, box[1])
                x2 = min(detrac_width, x1 + max(0, box[2]))
                y2 = min(detrac_height, y1 + max(0, box[3]))	
                box = [int(x1),int(y1),int(x2),int(y2)]		
                ignored_region.append(box)
            idx += 1

        frames = []
        
        for i in range(idx,len(root)):
            frame = {}
            frame['density'] = root[i].attrib['density']
            frame['num'] = root[i].attrib['num']
            frame['targets'] = []
            if root[i][0].tag == 'target_list' and len(root[i]) == 1:

                for target_item in root[i][0]:
                    if target_item.tag == 'target':
                        target = {}
                        target['id'] = int(target_item.attrib['id'])
                        if target_item[0].tag == 'box':
                            target['box'] = [
                                    float(target_item[0].attrib['left']),
                                    float(target_item[0].attrib['top']),
                                    float(target_item[0].attrib['width']),
                                    float(target_item[0].attrib['height'])]		
                            x1 = max(0, target["box"][0])
                            y1 = max(0, target["box"][1])
                            x2 = min(detrac_width, x1 + max(0, target["box"][2]))
                            y2 = min(detrac_height, y1 + max(0, target["box"][3]))	
                            target['box'] = [x1,y1,x2,y2]		
                            # target['box'] = [int(x1),int(y1),int(x2),int(y2)]
                        else:
                            print('this target has not box!')
                            raise AttributeError('this target has not box!')
                        if target_item[1].tag == 'attribute':
                            if target_item[1].attrib['vehicle_type'] in self.cls_idx:
                                target['cls'] = self.cls_idx[target_item[1].attrib['vehicle_type']]
                            else:
                                print('this target has not this cls name!')
                                raise AttributeError('this target has not this cls name!')
                        else:
                            print('this target has not attribute!')
                            raise AttributeError('this target has not attribute!')
                        
                        # 省去了 ID
                        target = np.array(target['box'] + [float(target['cls'])])
                        frame['targets'].append(target)
                    else:
                        print(target_item.tag)
                        raise AttributeError('this tag is not "target"!')
            else:
                print(root[i][0].tag)
                print(len(root[i][0]))
                raise AttributeError('this frame has error!')
            frames.append(frame)
        
        res = {
            'name':name,
            'sequence_attribute':sequence_attribute,
            'ignored_region':ignored_region,
            'frames':frames,
        }

        return res

    def eval(self):
        if self.mode in ['val']:
            return
        self.mode = 'val'
        self._prepare_coco()
    
    
    def _get_seqs_map(self):
    
        seqs_map = []
        
        img_count = 0
        for video in self.all_annonations:
            labels_list = []
            img_filenames = []
            ids_list = []
        
            video_imgs_dir = self.seqs_dir / video['anno']['name']
            frames = video['anno']['frames']
            ignored_regions = video['anno']['ignored_region']
            for frame in frames:
                file_name = video_imgs_dir / ('img' + str(int(frame['num'])).zfill(5) + '.jpg')
                frame_targets = []
                # frame_idx_list.append(frame['num'])
                for target in frame['targets']:
                    frame_targets.append(target)
                labels = np.array(frame_targets)
                
                if len(img_filenames) >= self.seq_len:
                    seqs_map.append(Seq(img_filenames,labels_list,self.preproc,ids_list,all_imgs = self.all_imgs,ignored_regions=ignored_regions))
                    labels_list = []
                    img_filenames = []
                    ids_list = []
                
                labels_list.append(labels)
                img_filenames.append(str(file_name))
                if self.mode in ['val']:
                    ids_list.append(frame['img_id'])

                img_count += 1
            
            labels_list = []
            img_filenames = []
            ids_list = []
                    
            for frame in frames[-self.seq_len:]:
                file_name = video_imgs_dir / ('img' + str(int(frame['num'])).zfill(5) + '.jpg')
                frame_targets = []
                # frame_idx_list.append(frame['num'])
                for target in frame['targets']:
                    frame_targets.append(target)
                labels = np.array(frame_targets)

                labels_list.append(labels)
                img_filenames.append(str(file_name))
                if self.mode in ['val']:
                    ids_list.append(frame['img_id'])

            if len(img_filenames) >= self.seq_len:
                seqs_map.append(Seq(img_filenames,labels_list,self.preproc,ids_list,all_imgs = self.all_imgs,ignored_regions=ignored_regions))

        self.seqs_map = seqs_map
        logger.info('load all data into seqs_map[{}/{}] ok!',len(seqs_map),img_count//self.seq_len)
        

    def read_all_anno(self):
        anno = []
        logger.info('load annotions...')
        imgs_count = 0
        boxes_count = 0
        videos_count = 0
        for anno_xml_file in self.anno_dir.iterdir():
            if str(anno_xml_file)[-4:].lower() != '.xml':
                continue
            temp = {'filename':anno_xml_file.resolve()}
            temp['anno'] = self.read_xml(anno_xml_file.resolve())
            imgs_count += len(temp['anno']['frames'])
            for frame in temp['anno']['frames']:
                boxes_count += len(frame["targets"])
            videos_count += 1
            anno.append(temp)
        logger.info('load annotions [{} boxes][{} imgs][{} videos] seccessfully!',boxes_count,imgs_count,videos_count)

        return anno,imgs_count

    def __len__(self):
        return len(self.seqs_map)
    
    def _cache_all(self):
        logger.info('start load all imgs!')
        img_count = 0
        video_count = 0
        for video in self.all_annonations:
            video_imgs_dir = self.seqs_dir / video['anno']['name']
            frames = video['anno']['frames']
            for frame in frames:
                file_name = video_imgs_dir / ('img' + str(int(frame['num'])).zfill(5) + '.jpg')
                self.all_imgs[file_name] = load_image(file_name)
                img_count += 1
            logger.info('video [{} {}][{} imgs]',video['anno']['name'],video_count,img_count)
            video_count += 1
        
        logger.info('load all data[{} imgs][{} videos] ok!',img_count,video_count)

    def clip_seqs(self,frames,video_imgs_dir,ignored_regions = []):
        '''
        
        ignored_regions: [[x1,y1,x2,y2],...]

        return:
        labels: [[x1,y1,x2,y2,category_id],...]
        '''
        video_size = len(frames)

        start = np.random.randint(video_size - self.seq_len)
        # print('random:',start)

        # frame_idx_list = []
        labels_list = []
        # img_list = []
        img_filenames = []
        ids_list = []
        
        for frame in frames[start:start+self.seq_len]:
            frame_targets = []
            # frame_idx_list.append(frame['num'])
            for target in frame['targets']:
                frame_targets.append(target)
            labels = np.array(frame_targets)
            

            file_name = video_imgs_dir / ('img' + str(int(frame['num'])).zfill(5) + '.jpg')
            img_filenames.append(file_name)
            
            # img = load_image(file_name)

            # 遮住忽略区域？
            # ave = img.mean()
            # for ignored in ignored_regions:
            #     img[ignored[1]:ignored[3],ignored[0]:ignored[2],:] = ave

            # if self.preproc is not None:
            #     img, labels = self.preproc(img, labels)

            if self.mode in ['val']:
                ids_list.append(frame['img_id'])
            labels_list.append(labels)
            # img_list.append(img)
        # print(labels_list)
        
        seq = Seq(img_filenames,labels_list,self.preproc,ids_list,all_imgs = self.all_imgs)
        return seq
        # if self.mode in ['val']:
        #     return img_filenames,labels_list,ids_list
        # else:
        #     return img_filenames,labels_list
        


    def pull_seqitem(self,video_idx,index):

        video = self.all_annonations[video_idx]
        video_imgs_dir = self.seqs_dir / video['anno']['name']
        frames = video['anno']['frames']
        ignored_regions = video['anno']['ignored_region']

        return self.clip_seqs(frames,video_imgs_dir,ignored_regions)


    def __getitem__(self, index):
        """
        通过给定 index 获取 图片与标签即图片COCO的id
        
        
        """
        return self.seqs_map[index]
        # return self.pull_seqitem(index)
        
    def convert_to_json(self,file_name):
        from data.detrac_classes import DETRAC_CLASSES
        json_data = {}
        json_data['categories'] = [
            {
                "id": 0,
                "name": DETRAC_CLASSES[0],
                "supercategory": ""
            },
            {
                "id": 1,
                "name": DETRAC_CLASSES[1],
                "supercategory": ""
            },
            {
                "id": 2,
                "name": DETRAC_CLASSES[2],
                "supercategory": ""
            },
            {
                "id": 3,
                "name": DETRAC_CLASSES[3],
                "supercategory": ""
            }
        ]
        json_data["images"] = []
        json_data["annotations"] = []
        img_id = 1
        anno_id = 1
        width = 960
        height = 540
        
        for video in self.all_annonations:
            video_name = video['anno']["name"]
            for frame in video['anno']['frames']:
                video_imgs_dir = self.seqs_dir / video_name
                img_filename = video_imgs_dir / ('img' + str(int(frame['num'])).zfill(5) + '.jpg')
                img_filename = str(img_filename)
                json_data['images'].append({
                                            'id':img_id,
                                            'file_name':img_filename,
                                            'width':width,
                                            'height':height,
                                            'valid':True,
                                            "rotate":0
                                            })
                frame['img_id'] = img_id
                order_idx = 1
                for target in frame['targets']:
                    box = {
                        "image_id": img_id,
                        "id": anno_id,
                        "bbox": [
                            target[0],
                            target[1],
                            target[2] - target[0],
                            target[3] - target[1],
                        ],
                        "iscrowd": 0,
                        "segmentation": [],
                        "category_id": int(target[4]),
                        "area": (target[2] - target[0]) * (target[3] - target[1]),
                        "order": order_idx
                    }
                    order_idx += 1
                    anno_id += 1
                    json_data["annotations"].append(box)
                img_id += 1
        logger.info('write to json file[{}]'.format(file_name))
        with open(file_name,'w') as f:
            json.dump(json_data,f)



def test_origin():
    detracDataset = DetracDataset('D:/Liyi/Datasets/DETRAC',
                                anno_dir = 'DETRAC-Test-Annotations-XML',
                                seqs_dir = 'Insight-MVT_Annotation_Test',
                                preproc=TrainTransform(max_labels=50,
                                    flip_prob=0.5,hsv_prob=0.1,need_size=[540,960])
                                )
    print(len(detracDataset))
    detracDataset.convert_to_json("test.json")
    from utils.functions import show_box_on_img

    for video in detracDataset.all_annonations:
        video_name = video['anno']["name"]
        video_imgs_dir = detracDataset.seqs_dir / video_name
        print(video_name)
        for frame in video['anno']['frames']:
            img_filename = video_imgs_dir / ('img' + str(int(frame['num'])).zfill(5) + '.jpg')
            print(frame)
            print(str(img_filename))
            boxes = []
            for target in frame['targets']:
                boxes.append(target)
            
            img = cv2.imread(str(img_filename))
            show_box_on_img(img,boxes,cat_id2str_dict=detracDataset.idx2cls,delay=1000)

def test():
    detracDataset = DetracDataset('D:/Liyi/Datasets/DETRAC',
                                anno_dir = 'DETRAC-Test-Annotations-XML',
                                seqs_dir = 'Insight-MVT_Annotation_Test',
                                preproc=TrainTransform(max_labels=50,
                                    flip_prob=0.5,hsv_prob=0.1,need_size=[540,960])
                                )
    print(len(detracDataset))

    detracDataset.convert_to_json("test.json")
    from utils.functions import show_box_on_img

    seq = detracDataset[2]
    for seq in detracDataset:
        for i in range(len(seq)):
            img, label = seq[i]
            # print(label)
            print(label.shape)
            mask = (label[:,-1] == 2)
            label = label[mask]
            if label.shape[0]>0:
                print(label)
                img = img.numpy()
                # img = img.transpose(2,0,1)
                show_box_on_img(img,label,cat_id2str_dict=detracDataset.idx2cls,delay=1000,mode='cxcywh')


if __name__ == '__main__':
    test()