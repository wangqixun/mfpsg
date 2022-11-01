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

import copy


def write_json(x_struct: dict, json_file: str):
    #json_str = json.dumps(x_struct,indent=2,ensure_ascii=False)
    with open(json_file, 'w') as fd:
        json.dump(x_struct, fd, indent=4, ensure_ascii=False)

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_model(cfg, ckp, test_pipeline_img_scale, transformers_model):
    cfg = mmcv.Config.fromfile(cfg)

    cfg['model']['type'] = 'Mask2FormerRelationForinfer'

    cfg['model']['relationship_head']['pretrained_transformers'] = transformers_model
    cfg['model']['relationship_head']['cache_dir'] = './'    
    if 'entity_length' in cfg['model']['relationship_head'] and cfg['model']['relationship_head']['entity_length'] > 1:
        cfg['model']['relationship_head']['entity_part_encoder'] = transformers_model

    test_pipeline = copy.deepcopy(cfg['data']['test']['pipeline'])
    test_pipeline[1]['img_scale'] = test_pipeline_img_scale
    cfg['data']['test']['pipeline'] = test_pipeline

    model = init_detector(cfg, ckp)
    return model


def get_test_id(psg_test_data_file):
    dataset = load_json(psg_test_data_file)
    test_id_list = [ 
        d['image_id'] for d in dataset['data'] if (d['image_id'] in dataset['test_image_ids']) and (len(d['relations']) != 0)
    ]
    return test_id_list

    # dataset['data'] = [
    #     d for d in dataset['data'] if len(d['relations']) != 0
    # ]



def get_val_p(test_pipeline_img_scale, cfg, ckp, psg_test_data_file, img_dir, test_mode_output_dir, transformers_model):

    jpg_output_dir = os.path.join(test_mode_output_dir,'submission/panseg')
    json_output_dir = os.path.join(test_mode_output_dir,'submission')

    os.makedirs(jpg_output_dir, exist_ok=True)
    INSTANCE_OFFSET = 1000

    test_id_list = get_test_id(psg_test_data_file)
    psg_test_data = load_json(psg_test_data_file)

    model = get_model(cfg, ckp, test_pipeline_img_scale, transformers_model=transformers_model)

    cur_nb = -1
    nb_vis = None

    prog_bar = mmcv.ProgressBar(len(test_id_list))
    all_img_dicts = []
    for d in psg_test_data['data']:
        image_id = d['image_id']
        if image_id not in test_id_list:
            continue
        cur_nb += 1
        prog_bar.update()


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
    # local 
    # get_val_p(
    #     mode='val',
    #     test_pipeline_img_scale=(1500, 1500),
    #     cfg='/share/wangqixun/workspace/bs/psg/mfpsg/configs/psg/v39-slurm.py',
    #     ckp='/share/wangqixun/workspace/bs/psg/mfpsg/output/v39/epoch_12.pth',
    #     val_mode_output_dir='/share/wangqixun/workspace/bs/psg/mfpsg/submit/val_v39_1500',
    #     test_mode_output_dir='/share/wangqixun/workspace/bs/psg/mfpsg/submit',

    #     psg_tra_data_file='/share/data/psg/dataset/for_participants/psg_train_val.json',
    #     psg_val_data_file='/share/data/psg/dataset/for_participants/psg_val_test.json',
    #     img_dir='/share/data/psg/dataset',
    #     transformers_model='/share/wangqixun/workspace/bs/tx_mm/code/model_dl/hfl/chinese-roberta-wwm-ext',
    # )

    # test submit
    # TODO 
    # needs to be modified
    # ==== start ========================================================================================
    submit_output_dir = '/share/wangqixun/workspace/bs/psg/OpenPSG/submit/new_norare'  # submit 输出地址
    psg_test_data_file = '/share/data/psg/dataset/for_participants/psg_test.json'
    img_dir = '/share/data/psg/dataset'  # 图像地址

    config_file = '/root/mfpsg/configs/psg/submit_cfg.py'  # 训练时候用的config
    checkpoint_file = '/root/checkpoint/epoch_12.pth'  # 训练得到的权重。默认的地址是我们训练出来的权重
    pretrained_transformers = '/root/test_submit/pretrain_model/chinese-roberta-wwm-ext'  # 训练时用的 pretrained_transformers
    # ==== end ==========================================================================================
    get_val_p(
        test_pipeline_img_scale=(1500, 1500),
        cfg=config_file,
        ckp=checkpoint_file,
        psg_test_data_file=psg_test_data_file,
        img_dir=img_dir,
        test_mode_output_dir=submit_output_dir,
        transformers_model=pretrained_transformers,
    )

































