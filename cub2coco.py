from json.tool import main
import mmcv
import os.path as osp
def cub2coco(root):
    bbox_path = osp.join(root, "bounding_boxes.txt")
    label_path = osp.join(root, "image_class_labels.txt")
    label_mapping_path = osp.join(root, "classes.txt")
    label_class = dict(map(lambda p:p.strip().split(" "), open(label_mapping_path, "r").readlines()))
    label_class = {int(key): value for key, value in label_class.items()}
    
    imgid_bbox = {int(res[0]) : [float(t) for t in res[1:]] for res in map(lambda p:p.strip().split(" "), open(bbox_path, "r").readlines())}
    imgid_label = dict(map(lambda p:p.strip().split(" "), open(label_path, "r").readlines()))
    imgid_label = {int(key): int(value) for key, value in imgid_label.items()}
    
    
    
    
path = "../dataSet/CUB_200_2011/"
cub2coco(path)