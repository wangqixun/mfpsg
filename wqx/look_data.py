import cv2
import numpy as np
import os
import random
from panopticapi.utils import rgb2id, id2rgb
import mmcv
from tqdm import tqdm

from IPython import embed
import json


def write_json(x_struct: dict, json_file: str):
    #json_str = json.dumps(x_struct,indent=2,ensure_ascii=False)
    with open(json_file, 'w') as fd:
        json.dump(x_struct, fd, indent=4, ensure_ascii=False)

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def look_zhanbi():
    json_file = '/share/data/psg/dataset/for_participants/psg_train_val.json'
    psg_data = load_json(json_file)
    img_dir = '/share/data/psg/dataset'
    res = {}

    for dddd in psg_data['data']:
        # pan_file = os.path.join(img_dir, dddd['pan_seg_file_name'])
        # pan_img = cv2.imread(pan_file)
        # pan_img = pan_img[..., ::-1]
        # id_img = rgb2id(pan_img)
        h, w = dddd['height'], dddd['width']
        
        sample = []

        for s_idx, o_idx, r_cls in dddd['relations']:
            s_cls = dddd['segments_info'][s_idx]['category_id']
            s_id = dddd['segments_info'][s_idx]['id']
            s_area = dddd['segments_info'][s_idx]['area']
            s_zhanbi = s_area / (1e-8+h*w)

            o_cls = dddd['segments_info'][o_idx]['category_id']
            o_id = dddd['segments_info'][o_idx]['id']
            o_area = dddd['segments_info'][o_idx]['area']
            o_zhanbi = o_area / (1e-8+h*w)

            r_cls = r_cls

            sample.append(
                dict(
                    s_cls=s_cls,
                    s_id=s_id,
                    s_zhanbi=s_zhanbi,
                    o_cls=o_cls,
                    o_id=o_id,
                    o_zhanbi=o_zhanbi,
                    r_cls=r_cls,
                )
            )

        res[''] = sample








