from turtle import forward
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses.eqlv2 import EQLv2
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv.cnn.bricks.transformer import build_positional_encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Recall
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertModel, BertTokenizer

from mmdet.models.losses import accuracy
import numpy as np

from IPython import embed


@HEADS.register_module()
class BertTransformer(BaseModule):
    def __init__(
        self, 
        pretrained_transformers='/share/wangqixun/workspace/bs/tx_mm/code/model_dl/hfl/chinese-roberta-wwm-ext', 
        cache_dir='/share/wangqixun/workspace/bs/psg/psg/tmp',
        input_feature_size=256,
        layers_transformers=6,
        feature_size=768,
        num_classes=133,
        num_cls=56,
        cls_qk_size=512,
        loss_weight=1.,
        num_entity_max=30,
        positional_encoding=None,
        use_background_feature=False,
        entity_length=1,
        entity_part_encoder='/mnt/mmtech01/usr/guiwan/workspace/model_dl/hfl/chinese-roberta-wwm-ext',
        entity_part_encoder_layers=6,
        loss_mode='v1',
    ):
        '''
            loss_mode = 'v1'
                [bs, head, N, N] -> [bs*head, N*N]
            loss_mode = 'v2'
                [bs, head, N, N] -> [bs, head*N*N]
            loss_mode = 'v3'
                [bs, head, N, N] -> [bs, N, head, N] -> [bs*N, head*N]
        '''
        super().__init__()
        self.num_cls = num_cls
        self.cls_qk_size = cls_qk_size
        self.fc_input = nn.Sequential(
            nn.Linear(input_feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )
        self.fc_output = nn.Linear(feature_size, num_cls + 1)
        
        self.cache_dir = cache_dir
        self.model = AutoModel.from_pretrained(pretrained_transformers, cache_dir=cache_dir)
        self.model.encoder.layer = self.model.encoder.layer[:layers_transformers]
        self.model.embeddings.word_embeddings = None
        self.loss_weight = loss_weight
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.num_entity_max = num_entity_max
        self.num_classes = num_classes
        if positional_encoding is not None:
            self.positional_encoding_cfg = positional_encoding
            self.postional_encoding = build_positional_encoding(positional_encoding)
        self.use_background_feature = use_background_feature
        self.entity_length = entity_length
        self.entity_part_encoder = entity_part_encoder
        self.entity_part_encoder_layers = entity_part_encoder_layers
        self.loss_mode = loss_mode
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_cls, dtype=torch.float))

        
        # self.eqlv2_loss = EQLv2(num_classes=num_cls)
        self.get_recall_N = Recall(average='macro', num_classes=num_cls + 1, top_k=20)
        self.ce_loss = nn.CrossEntropyLoss()
        
        
    def forward(self,inputs_embeds, attention_mask=None):
        position_ids = torch.tensor(list(range(0, inputs_embeds.shape[1]))).to(inputs_embeds.device).to(torch.long)  # 先不加global feature
        if inputs_embeds.shape[-1] != self.feature_size:
            encode_inputs_embeds = self.fc_input(inputs_embeds)
        else:
            encode_inputs_embeds = inputs_embeds
        encode_res = self.model(inputs_embeds=encode_inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)
        encode_embedding = encode_res['last_hidden_state']
        pooling_embedding = encode_embedding.mean(dim=1)
        cls_pred = self.fc_output(pooling_embedding)

        return cls_pred


    def get_f1_p_r(self, y_pred, y_true, mask_attention, th=0):
        # y_pred     [bs, 56, N, N]
        # y_true     [bs, 56, N, N]
        # mask_attention   [bs, 56, N, N]
        res = []
        
        y_pred[y_pred > th] = 1
        y_pred[y_pred < th] = 0

        n1 = y_pred * y_true * mask_attention
        n2 = y_pred * mask_attention
        n3 = y_true * mask_attention

        p = 100 * n1.sum(dim=[1,2,3]) / (1e-8 + n2.sum(dim=[1,2,3]))
        r = 100 * n1.sum(dim=[1,2,3]) / (1e-8 + n3.sum(dim=[1,2,3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([f1.mean(), p.mean(), r.mean()])

        mask_mean = y_true.sum(dim=[0, 2, 3]) > 0
        p = 100 * n1.sum(dim=[0,2,3]) / (1e-8 + n2.sum(dim=[0,2,3]))
        r = 100 * n1.sum(dim=[0,2,3]) / (1e-8 + n3.sum(dim=[0,2,3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([
            torch.sum(f1 * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(p * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(r * mask_mean) / (torch.sum(mask_mean) + 1e-8),
        ])

        return res


    

    def loss(self, pred, target, mask_attention=None):
        # pred     [bs, 56, N, N]
        # target   [bs, 56, N, N]
        # mask_attention   [bs, N]
        losses = {}
        bs, nb_cls = pred.shape

        loss = self.ce_loss(pred, target)

        # eqlv2_loss = self.eqlv2_loss(input_tensor.T, target_tensor.T)
        
        # loss = category_loss # + focal_loss # + eqlv2_loss
        
        losses['loss_relationship'] = loss * self.loss_weight

        # f1, p, r
        # [f1, precise, recall], [f1_mean, precise_mean, recall_mean] = self.get_f1_p_r(pred, target, mask)
        # losses['rela.F1'] = f1
        # losses['rela.precise'] = precise
        # losses['rela.recall'] = recall
        # losses['rela.F1_mean'] = f1_mean
        # losses['rela.precise_mean'] = precise_mean
        # losses['rela.recall_mean'] = recall_mean

        # recall
        recall = self.get_recall_N(pred, target)
        losses['rela.recall@20'] = recall

        return losses


    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
            不用加激活函数，尤其是不能加sigmoid或者softmax！预测
            阶段则输出y_pred大于0的类。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss





def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # print(inputs, targets)
    inputs = inputs.float()  # (B, C)
    targets = targets.float()  # (B, C)
    p = torch.sigmoid(inputs)  # (B, C)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (B, C)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)  # (B, C)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # # (B, C)
        loss = alpha_t * loss # (B, C)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
