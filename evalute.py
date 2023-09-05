import os
import sys
import shutil
from loguru import logger
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch

exe_name = 'DETRAC_DET_EVAL.exe'
exp_name = 'YOLOX'
temp_dir = 'mnum_25_12'
detection_dir = '/Datasets/DETRAC-Test-Det/use_here/ours/' + temp_dir + '/'
test_files_dir = '/DETRAC-Test-Det/use_here/seqs/'
step = '0.1'
output_dir = Path('D:/Liyi/Datasets/DETRAC-Test-Det/use_here/ours/res/' + temp_dir) # 没啥用
output_dir.mkdir(exist_ok=True)
subSets = ['full','easy','medium','hard','cloudy','night','rainy','sunny']
aps = {}


folder_path = detection_dir# Path() # Path(detection_dir) / temp_dir

def xyxy2xywh(bboxes):
    '''
    转换为 左上角点坐标 与 宽高。
    '''
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes
    
def nms(detections,nms_thre = 0.5):
	old = detections.shape[0]
	nms_out_index = torchvision.ops.nms(
		detections[:, :4],
		detections[:, 4],
		nms_thre,
	)
	# if old != len(nms_out_index):
	# 	print('{}->{}\n'.format(old,len(nms_out_index)))
	return detections[nms_out_index]

def deal_raw_res():
    # 遍历文件夹中的所有文件
    logger.info('detection_dir: {}',detection_dir)
    for filename in os.listdir(detection_dir):
        if filename.endswith('.txt'):
            # 打开文件并读取内容
            with open(os.path.join(detection_dir, filename), 'r') as f:
                lines = f.readlines()
            
            res_lines = []
            cur_num = 1
            cur_bboxes = []
            for i, line in enumerate(lines):
                parts = line.strip().split(',')
                del parts[-1]
                
                box = [float(parts[2]),float(parts[3]),float(parts[2]) + float(parts[4]),float(parts[3]) + float(parts[5]),float(parts[6])]
                
                if cur_num != int(parts[0]):
                    cur_bboxes = torch.tensor(cur_bboxes)
                    res = nms(cur_bboxes)
                    cur_count = 1
                    res = xyxy2xywh(res)
                    for t_res in res:
                        res_lines.append("{},{},{},{},{},{},{}\n".format(cur_num,cur_count,t_res[0],t_res[1],t_res[2],t_res[3],t_res[4]))
                        cur_count += 1
                    cur_bboxes = []

                cur_num = int(parts[0])
                
                cur_bboxes.append(box)


            with open(os.path.join(folder_path, filename), 'w') as f:
                # 对每一行进行处理
                
                for i, line in enumerate(res_lines):
                    f.write(line)

            print(f"文件 {filename} 已修改")

    print("所有文件已处理")


def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    
def get_ap(filename):
    
    with open(filename,'r') as file:
        lines = file.readlines()
        
    # 将每行数据转换为二元组 (x, y)
    data = []
    for line in lines:
        parts = line.strip().split(' ')
        x = float(parts[0])
        y = float(parts[1])
        data.append((x, y))

    # 将数据转换为 numpy 数组格式
    rec = np.array([d[0] for d in data])
    pre = np.array([d[1] for d in data])
    return voc_ap(pre,rec)

def show_pr_cure(filename):
    # 读取文本文件
    with open('0.1YOLOX_detection_PR.txt', 'r') as f:
        lines = f.readlines()

    # 将每行数据转换为二元组 (x, y)
    data = []
    for line in lines:
        parts = line.strip().split(' ')
        x = float(parts[0])
        y = float(parts[1])
        data.append((x, y))

    # 将数据转换为 numpy 数组格式
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    # 绘制 PR 曲线
    plt.step(x, y, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.show()
  
def main():

    logger.remove()
    logger.add(sys.stdout, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="INFO",enqueue=True)
    logger.add(sys.stderr, colorize=True,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <yellow>{file}:{line}</yellow>: \n<level>{message}</level>", 
                    level="WARNING",enqueue=True)
    eval_log_file = "log_eval_{time:YY_MM_DD_HH_mm_ss}.txt"
    # logger.add(sys.stdout,format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",level="INFO",enqueue=True)
    # 
    logger.add(eval_log_file,format="{time:YYYY-MM-DD HH:mm:ss} {level} {file}:{line}: \n{message}",level="INFO",enqueue=True)
        
    logger.info('dir_name: {}',temp_dir)

    deal_raw_res()
    
    for setName in subSets:
        logger.info('cur set: [{}]',setName)
        file_list_file = test_files_dir + 'testlist-' + setName + '.txt'
        cur_output_name = str(output_dir / (exp_name + '_' + setName + '_PR.txt'))

        cmdline = [exe_name,exp_name,detection_dir,file_list_file,step,str(output_dir)]
        p = subprocess.run(cmdline,stdout=sys.stdout,stderr=sys.stderr) # ,capture_output=True
        res_file = '0.1YOLOX_detection_PR.txt'
        shutil.move(res_file,cur_output_name)
        logger.info('[{}] PR res_file save to [{}]',setName,cur_output_name)
        ap = get_ap(cur_output_name)
        logger.info('{} AP is [{}]',setName,ap)
        aps[setName] = ap
    
    logger.info('All AP is [{}]',aps)
    
    

if __name__ == '__main__':
    main()