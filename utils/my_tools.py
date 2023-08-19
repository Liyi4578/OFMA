# -*- coding:utf-8 -*-
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme()
import torch
import numpy as np
import cv2
import os
import sys
import torch.nn.functional as F

cur_img = None
cur_boxes = None

def draw_weight_plus(weights,hws = [[80,80],[40,40],[20,20],[10,10],[5,5]],img = None,savename = None,bboxes = None):
    # num_points = []
    # for it in hws:
        # num_points.append(it[0]*it[1])
    # weight_list = weights.split(num_points, 0)
    count = 0
    for i,hw in enumerate(hws):
        h, w = hw
        weight = weights[count:count + h*w]
        weight1 = weight.reshape(h,w).detach().cpu().numpy()
        count += h*w
        
        print(f'level - {i+1}/{len(hws)}')
        draw_numpy(weight1)
        

def draw_weight(weight,savename = None):
    weight1 = weight[:6400].reshape(80,80).detach().cpu()
    weight2 = weight[6400:6400+1600].reshape(40,40).detach().cpu()
    weight3 = weight[8000:8000+400].reshape(20,20).detach().cpu()
    weight4 = weight[8400:8400+100].reshape(10,10).detach().cpu()
    weight5 = weight[8500:8500+25].reshape(5,5).detach().cpu().numpy()
    weights = [weight1,weight2,weight3,weight4,weight5]
    for i in range(len(weights)):
        print(f'level - {i+1}/{len(weights)}')
        if savename is not None:
            draw_numpy(weights[i],savename = savename + f'_level-{i+1}.png')
        else:
            draw_numpy(weights[i])

def draw_numpy(x,labels=None,savename = None,bboxes = None):
    plt.imshow(x)
        
    if savename is not None:
        # print('save to',savename)
        plt.savefig(savename, dpi=200)
    else:
        pass
        plt.show()
    plt.close()
    
def show_statistics(data,name):
    # print(tuple(data.shape))
    print('{} {:.2f}->{:.2f},mean:{:.2f},std:{:.2f},sum:{:.2f},shape{}'.format(name,float(data.min()),float(data.max()),float(data.mean()),float(data.std()),float(data.sum()),tuple(data.shape)))

def show_plot(x,name= None,savename = None):
    x = x.detach().cpu().squeeze().numpy()
    plt.plot(x)
    if name is not None:
        plt.title(name)
    # plt.show()
    if savename is not None:
        plt.savefig(savename + '.png', dpi=200)
    else:
        plt.show()
    plt.close()


def show_box_on_img(img,boxes,cat_id2str_dict = None,mode='xyxy',name=None):
    '''
    img: can be show with opencv
    boxes: iterable of [x1,y1,x2,y2,category]
    '''
    
    if mode == 'cxcywh':
        raise ValueError('cxcywh2xyxy?')
        pass#boxes = cxcywh2xyxy(boxes)
        # print(boxes)
    img = img.permute(1,2,0)
    std =torch.tensor([58.395, 57.12, 57.375])
    mean=torch.tensor([123.675, 116.28, 103.53])
    img = img * std + mean
    # img = img[:,:,[2,1,0]]
    print('img shape:',img.shape)
    
    img = img.type(torch.uint8)
    rect_list =[]

    img = np.ascontiguousarray(img.numpy()[:,:,[2,1,0]])
    for box in boxes:
        
        box = [int(t) for t in box]

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        
        cv2.rectangle(img, p1, p2, (204, 204, 51), 2)

    # cv2.imshow('img',img)
    # ax.imshow(img)
    # plt.show()

    # 保存图片
    if name is not None:
        imgs_dir = "temp_imgs"
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        cv2.imwrite(imgs_dir + "/" + name[:-3]+'jpg',img)
   
    print('num gt',len(boxes))
    # cv2.imshow('test',img)
    # cv2.waitKey(0)


def draw_heatmap(heatmap,strides):
    h,w = heatmap.shape
    canvas = np.zeros([h*strides[0],w*strides[1],3],dtype=np.uint8)
    xs = [xs*strides[1] for xs in range(w)]
    ys = [ys*strides[0] for ys in range(h)]

    canvas[ys,:,:]  = 233
    canvas[:,xs,:]  = 233
    xs += [w*strides[1]]
    ys += [h*strides[0]]
    for y in range(h):
        for x in range(w):
            canvas[ys[y]:ys[y+1],xs[x]:xs[x+1],:] = heatmap[y,x]
    
    return canvas

def draw_heatmap_on_img(heatmap,hws,img_path,img_dir='temp_imgs',save_dir="heatmap",name=None,origin_size=None):
    '''
    heatmap [H,W]
    '''
    img = cv2.imread(img_dir + '/' + img_path[:-3] + 'jpg')
    if name is None:
        name = img_path[:-4]
    count = 0

    for i,hw in enumerate(hws):
        h, w = hw
        weight = heatmap[count:count + h*w]
        count += h*w
        weight = weight.reshape(h,w).detach().cpu().numpy()

        heatmap_grid =np.uint8(255 * weight)
        # heatmap_grid = draw_heatmap(np.uint8(255 * weight),(int(img.shape[0]/h), int(img.shape[1]/w)))
        heatmap_grid = cv2.resize(heatmap_grid, (img.shape[1], img.shape[0])) # ,interpolation=cv2.INTER_NEAREST
        print('heat map shape {} -> {}'.format(weight.shape,heatmap_grid.shape[:2]))
        
        # heatmap0 = np.uint8(255 * heatmap0)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap0 = cv2.applyColorMap(heatmap_grid, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap0 * 0.4 + img *0.6 # 这里的0.4是热力图强度因子
        superimposed_img = superimposed_img[:origin_size[0],:origin_size[1]]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name[:-4] + f"_{i}"+ '.jpg'), superimposed_img)
        print(os.path.join(save_dir, name[:-4] + f"_{i}" + '.jpg'))

    plt.close()	#关掉展示的图片

def draw_points_on_img(heatmap,points,hws,img_path,img_dir='temp_imgs',save_dir="heatmap",name=None):
    '''
    heatmap [H,W]
    '''
    img = cv2.imread(img_dir + '/' + img_path[:-3] + 'jpg')
    if name is None:
        name = img_path[:-4]
    count = 0

    # canvas = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

    for i,hw in enumerate(hws):
        h, w = hw
        weight = heatmap[count:count + h*w]
        weight = weight.reshape(h,w).detach().cpu().numpy()
        sl_points = points[count:count + h*w]
        count += h*w

        weight = cv2.resize(weight, (img.shape[1], img.shape[0])) # ,interpolation=cv2.INTER_NEAREST
        heatmap0 = np.uint8(255 * weight)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        weight = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = weight * 0.4 + img *0.6 # 这里的0.4是热力图强度因子

        for x,y in sl_points:
            cv2.circle(superimposed_img,(int(x),int(y)),2,(255,0,0),2)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name + f"point_{i}"+ '.jpg'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看
        print(os.path.join(save_dir, name + f"point_{i}" + '.jpg'))


def test_flop():

    size_divisor = 32
    h,w = 800,1280

    from mmcv import Config
    from mmdet.models import build_detector,build_head

    from mmdet.utils import replace_cfg_vals
    from mmcv.cnn import get_model_complexity_info
    ori_shape = (3, h, w)
    divisor = size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor
    input_shape = (3, h, w)

    config_file = "configs/test.py"  # "D:/Liyi/mmdetection-2.26.0/configs/gfl/gfl_r50_fpn_1x_coco.py" # "configs/test.py" 

    cfg = Config.fromfile(config_file)
    cfg = replace_cfg_vals(cfg)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    if divisor > 0 and \
            input_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')





def test():

    global cur_img,cur_boxes
    import mmcv
    import torch
    import torch.distributed as dist
    from mmcv import Config, DictAction
    from mmcv.runner import get_dist_info, init_dist
    from mmcv.utils import get_git_hash
    from mmdet.apis import init_random_seed, set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
    from mmdet.apis import init_detector, inference_detector

    config_file = "configs/test.py"
    cfg = Config.fromfile(config_file)
    cfg = replace_cfg_vals(cfg)
    checkpoint_file = 'r50_wodd.pth'# 'epoch_defcn.pth'# 'fcos_center.pth'# None # 'epoch_1.pth'
    device = 'cuda:0'
    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)

    datasets = [build_dataset(cfg.data.wan)]
    dataset = datasets[0]
    
    model.CLASSES = dataset.CLASSES
    # model = build_detector(
        # cfg.model,
        # train_cfg=cfg.get('train_cfg'),
        # test_cfg=cfg.get('test_cfg'))
    # model.init_weights()

    cur = 0
    start_num = 60
    end_num = start_num + 10
    # '000000022969.jpg',
    show_imgs = ['000000430961.jpg','000000360661.jpg'] # '000000430961.jpg' 000000022969

    for data in dataset:
        # cur += 1
        # if cur < start_num:
        #     continue
        
        # dict_keys(['img_metas', 'img', 'gt_bboxes', 'gt_labels'])
        img_metas = data['img_metas'].data
        print(img_metas['filename'])

        if img_metas['ori_filename'] not in show_imgs:
            continue
        show_imgs.remove(img_metas['ori_filename'])
        if len(show_imgs) == 0:
            cur = end_num + 1

        img = data['img'].data.cuda()
        gt_bboxes = data['gt_bboxes'].data.cuda()
        gt_labels = data['gt_labels'].data.cuda()
        cur_img = img.cpu().clone()
        
        cur_boxes = gt_bboxes.cpu().clone()
        # print(cur_boxes)
        show_box_on_img(cur_img,cur_boxes,name=img_metas['ori_filename'])
        input_img = img[None]
        model.eval()
        with torch.no_grad():
            out = model.forward_train(input_img,[img_metas,],[gt_bboxes,],[gt_labels,])
            print(out)
            # print(out.keys())
            # outs = model.simple_test(input_img,[img_metas,])
            # res = model.get_bboxes(outs, img_metas=[img_metas,])
            # print(outs)
            # losses = model.loss(*outs,gt_bboxes=gt_bboxes,gt_labels=gt_labels,
            #     img_metas=img_metas)

        if cur >= end_num:
            break
        
        
if __name__ == '__main__':
    # test()
    test_flop()





