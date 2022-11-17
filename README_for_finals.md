# 上帝说要有光 团队方案


## 方法模型设计/改进思路

Panoptic Scene Graph Generation 在全景分割的基础上，需要为其中具有位置、语义关系的两个“物体”（或者说是“实体”）进行“关系”的建模。

    我们认为，主、宾实体的检测，与关系的建立是两大核心步骤，并且前者很大程度上决定了后者的召回上限。
在本次比赛中，我们将主、宾实体的检测与关系的建立进行二阶段训练。
相比于一阶段端到端的训练方案，二阶段方案由于在关系GT的匹配上不需要使用匈牙利匹配，训练过程更加稳定，且全景分割的精度更高。
在获得每个实体的 mask 后，依据 mask 为每一个实体划分成 L 个 token，并额外增加一个 cls token，随后所有实体的 N * (L+1) 个token送入 transformer 进行全局建模，得到 [bs, N*(L+1), 768] 维张量。 随后在实体粒度上进行全局池化，将一个实体的多个 token 收紧为 1 个token，即得到 [bs, N, 768] 维张量。
在得到实体级别 embedding 后，只需用对应的 embedding 建模任意两个 token 的关系即可。
同样是对任意两个 token 进行建模，[GlobalPointer](https://kexue.fm/archives/8373) 给出了非常优秀的解决方案。
GlobalPointer 是为解决 NLP 任务中“实体抽取”问题提出的方案。它将两两关系的建模直接使用 self-attention 实现，不仅将嵌套问题离散化，而且精度、速度均有提升。
我们借鉴 GlobalPointer 方法，通过 self-attention layer 实现两两实体关系的建模。







## 实现细节
全景分割部分采用开源的[Mask2Former](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)预训练方案与权重，



## 训练流程
```bash
# 8卡训练
bash tools/dist_train.sh configs/psg/submit_cfg.py 8 
```


## 对模型性能有影响的训练/推理策略
todo




