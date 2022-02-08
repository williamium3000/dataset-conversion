import mmcv
import os.path as osp
import os
import tqdm
import json
def cub2coco(root, save_dir):
    info_train = {
        "categories":[],
        "annotations":[],
        "images":[]
            }
    info_test = {
        "categories":[],
        "annotations":[],
        "images":[]
            }
    info = {
        0:info_test,
        1:info_train
    }
    bbox_path = osp.join(root, "bounding_boxes.txt")
    label_path = osp.join(root, "image_class_labels.txt")
    label_mapping_path = osp.join(root, "classes.txt")
    img_id_path = osp.join(root, "images.txt")
    img_dir = osp.join(root, "images")
    split_file_path = osp.join(root, "train_test_split.txt")
    
    imgid_filepath = dict(map(lambda p:p.strip().split(" "), open(img_id_path, "r").readlines()))
    imgid_filepath = {int(key): value for key, value in imgid_filepath.items()}
    
    label_class = dict(map(lambda p:p.strip().split(" "), open(label_mapping_path, "r").readlines()))
    label_class = {int(key): value for key, value in label_class.items()}
    
    imgid_bbox = {int(res[0]) : [float(t) for t in res[1:]] for res in map(lambda p:p.strip().split(" "), open(bbox_path, "r").readlines())}
    imgid_label = dict(map(lambda p:p.strip().split(" "), open(label_path, "r").readlines()))
    imgid_label = {int(key): int(value) for key, value in imgid_label.items()}
    
    imgid_split = dict(map(lambda p:p.strip().split(" "), open(split_file_path, "r").readlines()))
    imgid_split = {int(key): int(value) for key, value in imgid_split.items()}
    ####################################################################################
    #                                  categories
    ####################################################################################
    print("processing categories")
    for id, name in tqdm.tqdm(label_class.items()):
        info[0]["categories"].append(
            {'id': id - 1, 'name': name}
        )
        info[1]["categories"].append(
            {'id': id - 1, 'name': name}
        )
    ####################################################################################
    #                                  images
    ####################################################################################
    print("processing images")
    for id, filepath in tqdm.tqdm(imgid_filepath.items()):
        filepath = os.path.join(img_dir, filepath)
        h, w, c = mmcv.imread(filepath).shape
        info[imgid_split[id]]["images"].append(
                    {
                'file_name': os.path.basename(filepath),
                'height': h,
                'width': w,
                'id': id
            }
        )
    ####################################################################################
    #                                  annotations
    ####################################################################################
    print("processing annotations")
    for id, bbox in tqdm.tqdm(imgid_bbox.items()):
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        info[imgid_split[id]]["annotations"].append(
                {
                'area': bbox[-1] * bbox[-2],
                'iscrowd': 0,
                'image_id': id,
                'bbox': bbox,
                'category_id': imgid_label[id] - 1,
                'id': id
        }
        )
    print("finish processing: train {} val {}".format(len(info[1]["images"]), len(info[0]["images"])))
    print("saving to {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    train_path = osp.join(save_dir, "cub_train.json")
    val_path = osp.join(save_dir, "cub_val.json")
    json.dump(info[1], open(train_path, 'w'))
    json.dump(info[0], open(val_path, 'w'))
        
    
    
    
path = "../dataSet/CUB_200_2011/"
cub2coco(path, "cub")