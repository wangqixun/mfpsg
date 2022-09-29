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

下载 [coco instance val 2017](https://cocodataset.org/#download)，用于验证 psg val 的instance map

修改 `wqx/main.py` 中下列文件路径
```python
if __name__ == '__main__':
    # raw data file
    raw_psg_data='/share/data/psg/dataset/for_participants/psg_train_val.json'
    raw_coco_val_json_file='/share/data/coco/annotations/instances_val2017.json'

    # output file
    output_coco80_val_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/instances_val2017_coco80.json'
    output_tra_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_tra.json'
    output_val_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_val.json'
    output_val_instance_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_val.json'

```
执行
```
python wqx/main.py
```




<br>

## (可能)需要的一些预训练权重
### 分割部分:
[mask2former](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)，
[预训练权重](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)

### 关系分类：
[transformers](https://github.com/huggingface/transformers)，
[预训练权重](https://huggingface.co/hfl/chinese-roberta-wwm-ext)



<br>

## 训练
+ 调整 `configs/psg/submit_cfg.py` 中预训练路径、输出路径、tra val 数据路径
```python
# 输出路径
work_dir = 'output/v0'
```
+ 训练咯
```
# 8卡训练
bash tools/dist_train.sh configs/psg/submit_cfg.py 8 
```

<br>

## Submit
+ 调整 `wqx/infer_p.py` 中 `cfg` 和 `ckp`
```python
if __name__ == '__main__':
    get_val_p(
        mode='v0',
        cfg='configs/psg/mask2former_r50.py',
        ckp='output/v0/latest.pth',
    )
```
+ 推理
```

python wqx/infer_p.py
```
+ 打包
```
cd submit/
zip -r submission.zip submission/
```









