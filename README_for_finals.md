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


![img](./imgs/mfpsg_model.jpg)






## 实现细节
全景分割部分采用开源的[Mask2Former](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)预训练方案与权重，
transformer 部分采用开源的[transformers](https://github.com/huggingface/transformers)




## 训练流程
```bash
# 8卡训练
bash tools/dist_train.sh configs/psg/submit_cfg.py 8 
```
更多环境安装、训练、推理细节可[参阅](./README.md)


## 对模型性能有影响的训练/推理策略
+ #### 常规的数据增强
常规的数据增强可以提升全景分割、R20和mR20的精度



+ #### Transformer
关系部分的建模使用到了 transformer 模型，经过多个消融实验，显示使用 [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large?text=%E5%B7%B4%E9%BB%8E%E6%98%AF%5BMASK%5D%E5%9B%BD%E7%9A%84%E9%A6%96%E9%83%BD%E3%80%82) 的前2层精度最高。

+ #### 置信度连乘

最终输出的关系 predict 不仅与关系模型的输出相关，而且与 mask 精度呈现正相关。
```python
最终输出的分数 = 主语 mask 置信度 * 谓语关系置信度 * 宾语 mask 置信度
```
可以提高模型精度

+ #### Mask抖动

在训练阶段，只有gt的mask，非常准确。但在infer阶段，会出现大量置信度很低的mask，由此会造成训练-推理阶段的差异。为此，在训练阶段，对于mask进行小幅度膨胀、腐蚀的增强，此外随机生成一些假的mask作为负样本。在不增加任何推理成本的情况下，可提升模型精度

+ #### 多标签分类loss

关系的预测本质上是一个多标签问题，存在 N 个实体的情况下，需要对 ```56 * N * N``` 个预测值进行监督，一般来说，真正“有关系”的正样本对不会超过20个，即存在极其严重的类别不均衡问题。为了解决这一问题，我们借鉴了[将“softmax+交叉熵”推广到多标签分类问题](https://kexue.fm/archives/7359)的方法，显著提升了训练稳定性和精度

+ #### Focal loss
focal loss 可以进一步缓解长尾问题，精度小幅度提升