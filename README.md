# Mask2Former PSG


整体框架还是基于[mmdet](https://github.com/open-mmlab/mmdetection)，其中relation部分借鉴[transformers](https://github.com/huggingface/transformers)




<br>

## Install
环境参考 [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) 
```
pip install -e .
```

[apex](https://github.com/NVIDIA/apex) 建议通过下面代码安装
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
panopticapi
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

<br>

## 数据准备

修改 `wqx/main.py` 中下列文件路径
```python
if __name__ == '__main__':
    # TODO 
    # need to be modified
    # ==== start ====
    psg_dataset_dir = '/share/data/psg/dataset'  # 原始psg数据地址
    data_dir = '/root/test_submit/data'  # 预处理数据输出地址
    # ==== end ====
```
执行
```
python wqx/main.py
```




<br>

## 需要的一些预训练权重
+ 分割部分:
[mask2former](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)，
[预训练权重](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)

+ 关系分类：
[transformers](https://github.com/huggingface/transformers)，
[预训练权重](https://huggingface.co/hfl/chinese-roberta-wwm-ext)



<br>

## 训练
+ 调整 `configs/psg/submit_cfg.py` 中预训练路径、输出路径、tra val 数据路径
```python
# TODO 
# need to be modified
# ==== start ========================================================================================
psg_dataset_dir = '/share/data/psg/dataset'  # 原始psg数据地址
data_dir = '/root/test_submit/data'  # 预处理数据地址
# weight from https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former
load_from = '/root/test_submit/pretrain_model/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic_20220329_230021-3bb8b482.pth'  # 预训练权重 mask2former 
# weight from https://huggingface.co/hfl/chinese-roberta-wwm-ext
pretrained_transformers = '/root/test_submit/pretrain_model/chinese-roberta-wwm-ext'  # 预训练权重 roberta
cache_dir = '/root/test_submit/output/tmp'  # cache地址，随便给一个就行，/tmp 就行 
work_dir = '/root/test_submit/output/v36'  # 输出地址
# ==== end ==========================================================================================
```
+ 训练
```
# 8卡训练
bash tools/dist_train.sh configs/psg/submit_cfg.py 8 
```

<br>

## Infer and Submit

+ 初赛、决赛方案区别

初赛、决赛均适用本repo代码，区别是决赛加了一个数量极少类别的过滤方案。具体代码在 ```mfpsg/mmdet/models/detectors/mask2formerrelation.py``` 的第```805-808``` 行增加了以下代码：

```python
# 数量极少丢弃
for ratio_th, weight in [[[0, 0.001/100], -9999],  ]:
    mask = ((self.rela_cls_ratio > ratio_th[0]) & (self.rela_cls_ratio < ratio_th[1])) * 1
    relationship_output = (relationship_output + weight) * mask + relationship_output
```

+ 调整 `wqx/infer_p.py` 中路径
```python
if __name__ == '__main__':
    # TODO 
    # need to be modified
    # ==== start ========================================================================================
    submit_output_dir = '/share/wangqixun/workspace/bs/psg/OpenPSG/submit/new_latest'  # submit 输出地址
    psg_test_data_file = '/share/data/psg/dataset/for_participants/psg_test.json'
    img_dir = '/share/data/psg/dataset'  # 图像地址

    config_file = '/root/mfpsg/configs/psg/submit_cfg.py'  # 训练时候用的config
    checkpoint_file = '/root/checkpoint/epoch_12.pth'  # 训练得到的权重。默认的地址是我们训练出来的权重
    pretrained_transformers = '/root/test_submit/pretrain_model/chinese-roberta-wwm-ext'  # 训练时用的 pretrained_transformers
    # ==== end ==========================================================================================
```
+ 推理
```
python wqx/infer_p.py
```
+ 打包

输出格式为初赛、决赛提交的zip格式
```
cd submit/
zip -r submission.zip submission/
```









