# from xsma.generel_utils.tool_json import load_json, write_json
import numpy as np
from infer_p import get_tra_val_test_list
# from tqdm import tqdm
import json
import mmcv

def write_json(x_struct: dict, json_file: str):
    with open(json_file, 'w') as fd:
        json.dump(x_struct, fd, indent=4, ensure_ascii=False)

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data



def coco90_to_coco80(raw_json_file, new_json_file):
    raw_json = load_json(raw_json_file)
    print(raw_json.keys())
    print(raw_json['categories'])
    
    new_json = {}
    for k in ['info', 'licenses']:
        if k not in raw_json:
            continue
        new_json[k] = raw_json[k]

    new_json['images'] = []
    for img_info in raw_json['images']:
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        img_info_new = {
            'file_name': file_name,
            'height': height,
            'width': width,
            'id': img_info['id']
        }
        new_json['images'].append(img_info_new)


    coco_name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']    
    new_json['categories'] = []
    mapping_old_to_new = {}
    idx_new = 0
    for idx in range(len(raw_json['categories'])):
        supercategory = raw_json['categories'][idx]['supercategory']
        id_raw = raw_json['categories'][idx]['id']
        name = raw_json['categories'][idx]['name']
        if name in coco_name_list:
            idx_new += 1
            cate_info = {
                'supercategory': supercategory,
                'id': idx_new,
                'name': name
            }
            mapping_old_to_new[id_raw] = idx_new
            new_json['categories'].append(cate_info)


    new_json['annotations'] = []
    
    prog_bar = mmcv.ProgressBar(len(raw_json['annotations']))
    for idx, ann_info in enumerate(raw_json['annotations']):
        if idx == 0:
            print(ann_info)
        prog_bar.update()
        segmentation = ann_info['segmentation']
        area = ann_info['area']
        iscrowd = ann_info['iscrowd']
        image_id = ann_info['image_id']
        bbox = ann_info['bbox']
        category_id_old = ann_info['category_id']
        id_raw = ann_info['id']
        if category_id_old not in mapping_old_to_new:
            continue
        category_id_new = mapping_old_to_new[category_id_old]

        if isinstance(segmentation, dict) and not isinstance(segmentation['counts'], list):
            segmentation['counts'] = str(segmentation['counts'])
        ann_info_new = {
            'segmentation': segmentation,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id_new,
            'id': id_raw
        }
        new_json['annotations'].append(ann_info_new)

    write_json(new_json, new_json_file)


# psg json for train and val
def f2(psg_data, id_list, output_json):

    out_json = {}
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = []
    out_json['relations_categories'] = []
    

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]
    THING_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    STUFF_CLASSES = [
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
        'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
        'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]

    # categories
    class2id = {}
    for idx in range(len(CLASSES)):
        supercategory = ''
        name = CLASSES[idx]
        isthing = 1 if name in THING_CLASSES else 0
        categorie = dict(
            supercategory=supercategory,
            isthing=isthing,
            id=idx+1,
            name=name,
        )
        out_json['categories'].append(categorie)
        class2id[name] = idx
    
    # relations_categories
    for idx in range(len(psg_data['predicate_classes'])):
        name = psg_data['predicate_classes'][idx]
        categorie = dict(
            id=idx+1,
            name=name,
        )
        out_json['relations_categories'].append(categorie)

    # images
    prog_bar = mmcv.ProgressBar(len(psg_data['data']))
    for idx in range(len(psg_data['data'])):
        prog_bar.update()
        psg_data_info = psg_data['data'][idx]
        file_name = psg_data_info['file_name']
        height = psg_data_info['height']
        width = psg_data_info['width']
        img_id = int(psg_data_info['image_id'])
        if str(img_id) not in id_list:
            continue
        image = dict(
            file_name=file_name,
            height=height,
            width=width,
            id=img_id,
        )
        out_json['images'].append(image)

    # annotations
    prog_bar = mmcv.ProgressBar(len(psg_data['data']))
    for idx in range(len(psg_data['data'])):
        prog_bar.update()
        psg_data_info = psg_data['data'][idx]
        file_name = psg_data_info['pan_seg_file_name']
        image_id = int(psg_data_info['image_id'])
        relations = psg_data_info['relations']
        if str(image_id) not in id_list:
            continue
        segments_info = []
        for idx_segment in range(len(psg_data_info['segments_info'])):
            id = psg_data_info['segments_info'][idx_segment]['id']
            category_id = psg_data_info['segments_info'][idx_segment]['category_id']
            iscrowd = psg_data_info['segments_info'][idx_segment]['iscrowd']
            bbox = psg_data_info['annotations'][idx_segment]['bbox']
            area = psg_data_info['segments_info'][idx_segment]['area']
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            segment = dict(
                id=id,
                category_id=category_id+1,
                iscrowd=iscrowd,
                bbox=bbox,
                area=area,
            )
            segments_info.append(segment)
        annotation = dict(
            segments_info=segments_info,
            file_name=file_name,
            image_id=image_id,
            relations=relations,
        )
        out_json['annotations'].append(annotation)


    write_json(out_json, output_json)

    
# psg json for val instane map
def f3(psg_data, id_list, output_instance_json, coco_json_file):
    # coco_json_file = '/share/data/coco/annotations/instances_train2017_coco80.json'
    coco_json = load_json(coco_json_file)


    out_json = {}
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = coco_json['categories']
    
    # images
    use_coco_id_list = {}
    prog_bar = mmcv.ProgressBar(len(psg_data['data']))
    for idx in range(len(psg_data['data'])):
        prog_bar.update()
        psg_data_info = psg_data['data'][idx]
        coco_image_id = psg_data_info['coco_image_id']
        img_id = int(psg_data_info['image_id'])
        file_name = psg_data_info['file_name']
        height = psg_data_info['height']
        width = psg_data_info['width']
        if str(img_id) not in id_list:
            continue
        image = dict(
            file_name=file_name,
            height=height,
            width=width,
            id=img_id,
        )
        out_json['images'].append(image)
        use_coco_id_list[coco_image_id] = img_id

    # annotations
    prog_bar = mmcv.ProgressBar(len(psg_data['data']))
    for idx in range(len(coco_json['annotations'])):
        prog_bar.update()
        ann_info = coco_json['annotations'][idx]
        image_id = ann_info['image_id']
        if str(image_id) not in use_coco_id_list:
            continue
        ann_info['image_id'] = use_coco_id_list[str(image_id)]
        out_json['annotations'].append(ann_info)


    write_json(out_json, output_instance_json)

    

def f1(raw_psg_data, raw_psg_valtest_data, coco80_instance_val2017_json, output_tra_json, output_val_json, output_val_instance_json, ):
    # output_tra_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_tra.json'
    # output_val_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_val.json'
    # output_tra_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_tra.json'
    # output_val_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_val.json'
    # raw_psg_data = '/share/data/psg/dataset/for_participants/psg_train_val.json'

    # coco80_instance_train2017_json = '/share/data/coco/annotations/instances_train2017_coco80.json'
    # coco80_instance_val2017_json = '/share/data/coco/annotations/instances_val2017_coco80.json'

    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list(
        psg_tra_data_file=raw_psg_data, 
        psg_val_data_file=raw_psg_valtest_data,
    )

    psg_data = load_json(raw_psg_data)

    # psg数据改成coco pan格式，额外增加"relations_categories" 和 "relations"
    # 其中 "relations" 在 "annotations" 的元素中
    f2(psg_data, tra_id_list, output_tra_json)
    f2(psg_data, val_id_list, output_val_json)
    # psg数据改成coco instance 格式，用来计算psg val上 bbox map 和 segm map
    # f3(psg_data, tra_id_list, output_tra_instance_json, coco_json_file=coco80_instance_train2017_json)
    # f3(psg_data, val_id_list, output_val_instance_json, coco_json_file=coco80_instance_val2017_json)











if __name__ == '__main__':
    # raw data file
    raw_psg_traval_data='/share/data/psg/dataset/for_participants/psg_train_val.json'
    raw_psg_valtest_data='/share/data/psg/dataset/for_participants/psg_val_test.json'
    # raw_coco_val_json_file='/share/data/coco/annotations/instances_val2017.json'

    # output file
    # output_coco80_val_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/instances_val2017_coco80.json'
    # output_tra_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_tra.json'
    # output_val_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_val.json'
    # output_val_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_val.json'

    output_tra_json='/root/mfpsg/data/psg_tra.json'
    output_val_json='/root/mfpsg/data/psg_val.json'
    output_coco80_val_instance_json = None
    output_val_instance_json = None


    # coco90_to_coco80(
    #     raw_json_file=raw_coco_val_json_file,
    #     new_json_file=output_coco80_val_instance_json,
    # )
    f1(
        raw_psg_data=raw_psg_traval_data,
        raw_psg_valtest_data=raw_psg_valtest_data,
        coco80_instance_val2017_json=output_coco80_val_instance_json,
        output_val_instance_json=output_val_instance_json,
        output_tra_json=output_tra_json,
        output_val_json=output_val_json,
    )

