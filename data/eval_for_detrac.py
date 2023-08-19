from pycocotools.coco import COCO







def coco_to_detrac(coco_res_json):
    json_filename = 'kaggle/2/detracvod/coco_anno.json'
    coco = COCO(json_filename)
    cocoGt = coco
    cocoDt = cocoGt.loadRes("kaggle/2/detracvod/temp_coco_res.json")
    return cocoGt,cocoDt