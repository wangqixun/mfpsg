import sys
sys.path.append(".")
from mmdet.apis import init_detector, inference_detector
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


def get_model(cfg, ckp):
    cfg = mmcv.Config.fromfile(cfg)

    cfg['model']['type'] = 'Mask2FormerRelationForinfer'

    cfg['model']['relationship_head']['pretrained_transformers'] = cfg.model.relationship_head.pretrained_transformers
    cfg['model']['relationship_head']['cache_dir'] = cfg.model.relationship_head.cache_dir
    if 'entity_length' in cfg['model']['relationship_head'] and cfg['model']['relationship_head']['entity_length'] > 1:
        cfg['model']['relationship_head']['entity_part_encoder'] = cfg.model.relationship_head.cache_dir

    model = init_detector(cfg, ckp)
    return model


def get_tra_val_test_list():
    psg_tra_data_file = '/root/autodl-nas/for_participants/psg_train_val.json'
    psg_val_data_file = '/root/autodl-nas/for_participants/psg_val_test.json'
    psg_tra_data = load_json(psg_tra_data_file)
    psg_val_data = load_json(psg_val_data_file)

    tra_id_list = []
    val_id_list = []
    test_id_list = []

    for d in psg_tra_data['data']:
        if d['image_id'] in psg_tra_data['test_image_ids']:
            val_id_list.append(d['image_id'])
        else:
            tra_id_list.append(d['image_id'])

    for d in psg_val_data['data']:
        test_id_list.append(d['image_id'])
    
    tra_id_list = np.array(tra_id_list)
    val_id_list = np.array(val_id_list)
    test_id_list = np.array(test_id_list)
    print('tra', len(tra_id_list))
    print('val', len(val_id_list))
    print('test', len(test_id_list))

    # np.save('/share/wangqixun/workspace/bs/psg/psg/wqx/tra_id_list.npy', tra_id_list)
    # np.save('/share/wangqixun/workspace/bs/psg/psg/wqx/val_id_list.npy', val_id_list)
    # np.save('/share/wangqixun/workspace/bs/psg/psg/wqx/test_id_list.npy', test_id_list)
    
    return tra_id_list, val_id_list, test_id_list


def get_val_p(mode, cfg, ckp, val_mode_output_dir=None):
    jpg_output_dir = f'./submit/{mode}/submission'
    json_output_dir = f'./submit/{mode}/submission'

    if mode=='val':
        jpg_output_dir = os.path.join(val_mode_output_dir, 'submission/panseg')
        json_output_dir = os.path.join(val_mode_output_dir, 'submission')
        # jpg_output_dir = '/share/wangqixun/workspace/bs/psg/mfpsg/submit/val/submission/panseg'
        # json_output_dir = '/share/wangqixun/workspace/bs/psg/mfpsg/submit/val/submission'

    os.makedirs(jpg_output_dir, exist_ok=True)

    INSTANCE_OFFSET = 1000


    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list()
    psg_val_data_file = '/root/autodl-nas/for_participants/psg_val_test.json'
    psg_val_data = load_json(psg_val_data_file)

    img_dir = '/root/autodl-nas'
    model = get_model(cfg, ckp)

    cur_nb = -1
    nb_vis = None

    all_img_dicts = []
    for d in tqdm(psg_val_data['data']):
        cur_nb += 1
        if nb_vis is not None and cur_nb > nb_vis:
            continue

        image_id = d['image_id']

        if mode=='val' and image_id not in val_id_list:
            continue

        img_file = os.path.join(img_dir, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)

        pan_results = img_res['pan_results']
        # ins_results = img_res['ins_results']
        rela_results = img_res['rela_results']
        entityid_list = rela_results['entityid_list']
        relation = rela_results['relation']


        img_output = np.zeros_like(img)
        segments_info = []
        for instance_id in entityid_list:
            # instance_id == 133 background
            mask = pan_results == instance_id
            if instance_id == 133:
                continue
            r, g, b = random.choices(range(0, 255), k=3)
            
            mask = mask[..., None]
            mask = mask.astype(int)
            coloring_mask = np.concatenate([mask]*3, axis=-1)
            color = np.array([b,g,r]).reshape([1,1,3])
            coloring_mask = coloring_mask * color
            # coloring_mask = np.concatenate([mask[..., None]*1]*3, axis=-1)
            # for j, color in enumerate([b, g, r]):
            #     coloring_mask[:, :, j] = coloring_mask[:, :, j] * color
            img_output = img_output + coloring_mask
            idx_class = instance_id % INSTANCE_OFFSET + 1
            segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        img_output = img_output.astype(np.uint8)
        # mask = np.sum(img_output, axis=-1) > 0
        # img_output_2 = np.copy(img)
        # img_output_2[mask] = img_output_2[mask] * 0.5 + img_output[mask] * 0.5
        # img_output = np.concatenate([img_output_2, img_output], axis=1)
        cv2.imwrite(f'{jpg_output_dir}/{cur_nb}.png', img_output)

        if len(relation) == 0:
            relation = [[0, 0, 0]]
        if len(segments_info) == 0:
            r, g, b = random.choices(range(0, 255), k=3)
            segments_info = [dict(category_id=1, id=rgb2id((r, g, b)))]

        single_result_dict = dict(
            # image_id=image_id,
            relations=[[s, o, r + 1] for s, o, r in relation],
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % cur_nb,
        )
        all_img_dicts.append(single_result_dict)
    
    # write_json(all_img_dicts, f'{json_output_dir}/relation.json')
    with open(f'{json_output_dir}/relation.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)





if __name__ == '__main__':
    # get_tra_val_test_list()
    # get_test_p()
    get_val_p(
        mode='val',
        cfg='/root/mfpsg/configs/psg/v104.py',
        ckp='/root/autodl-tmp/output/swin-base-focal-loss-feature-merge-data-aug/epoch_1.pth',
        val_mode_output_dir='/root/autodl-tmp/output/swin-base-focal-loss-feature-merge-photo-distor/'
    )

    # landmark
    # best v1 ep30 31.3
    # v4 ep30 32.36
    # v5 ep30 31.94




































