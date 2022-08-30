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












