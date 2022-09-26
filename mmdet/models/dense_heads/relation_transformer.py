from turtle import forward
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv.cnn.bricks.transformer import build_positional_encoding

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        loss_alpha=None,
        train_add_noise_mask=False,
        embedding_add_cls=True,
        mask_shake=False,
        cls_embedding_mode='add',
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
        self.fc_output = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )
        self.cache_dir = cache_dir
        self.model = AutoModel.from_pretrained(pretrained_transformers, cache_dir=cache_dir)
        self.cls_q = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.cls_k = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.model.encoder.layer = self.model.encoder.layer[:layers_transformers]
        self.model.embeddings.word_embeddings = None
        self.loss_weight = loss_weight
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.num_entity_max = num_entity_max
        self.num_classes = num_classes
        if positional_encoding is not None:
            self.positional_encoding_cfg = positional_encoding
            self.postional_encoding_layer = build_positional_encoding(positional_encoding)
        self.use_background_feature = use_background_feature
        self.entity_length = entity_length
        self.entity_part_encoder = entity_part_encoder
        self.entity_part_encoder_layers = entity_part_encoder_layers
        self.loss_mode = loss_mode
        self.loss_alpha = loss_alpha
        self.train_add_noise_mask = train_add_noise_mask
        self.embedding_add_cls = embedding_add_cls
        self.mask_shake = mask_shake
        self.cls_embedding_mode = cls_embedding_mode
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_cls, dtype=torch.float))


    def forward(self,inputs_embeds, attention_mask=None):
        position_ids = torch.ones([1, inputs_embeds.shape[1]]).to(inputs_embeds.device).to(torch.long)
        if inputs_embeds.shape[-1] != self.feature_size:
            encode_inputs_embeds = self.fc_input(inputs_embeds)
        else:
            encode_inputs_embeds = inputs_embeds
        encode_res = self.model(inputs_embeds=encode_inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)
        encode_embedding = encode_res['last_hidden_state']
        encode_embedding = self.fc_output(encode_embedding)
        bs, N, c = encode_embedding.shape
        q_embedding = self.cls_q(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        k_embedding = self.cls_k(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        cls_pred = q_embedding @ torch.transpose(k_embedding, 2, 3) / self.cls_qk_size ** 0.5
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


    def get_recall_N(self, y_pred, y_true, mask_attention, th=0, N=20):
        # y_pred     [bs, 56, N, N]
        # y_true     [bs, 56, N, N]
        # mask_attention   [bs, 56, N, N]

        device = y_pred.device
        dtype = y_pred.dtype
        recall_list = []

        for idx in range(len(y_true)):
            sample_y_true = []
            sample_y_pred = []
            
            # find topk
            _, topk_indices = torch.topk(y_true[idx:idx+1].reshape([-1,]), k=N)
            for index in topk_indices:
                pred_cls = index // (y_true.shape[2] ** 2)
                index_subject_object = index % (y_true.shape[2] ** 2)
                pred_subject = index_subject_object // y_true.shape[2]
                pred_object = index_subject_object % y_true.shape[2]
                if y_true[idx, pred_cls, pred_subject, pred_object] == 0:
                    continue
                sample_y_true.append([pred_subject, pred_object, pred_cls])

            # find topk
            _, topk_indices = torch.topk(y_pred[idx:idx+1].reshape([-1,]), k=N)
            for index in topk_indices:
                pred_cls = index // (y_pred.shape[2] ** 2)
                index_subject_object = index % (y_pred.shape[2] ** 2)
                pred_subject = index_subject_object // y_pred.shape[2]
                pred_object = index_subject_object % y_pred.shape[2]
                sample_y_pred.append([pred_subject, pred_object, pred_cls])

            recall = len([x for x in sample_y_pred if x in sample_y_true]) / (len(sample_y_true) + 1e-8)
            recall_list.append(recall)

        mean_recall = torch.tensor(recall_list).to(device).mean() * 100
        return mean_recall


    def loss(self, pred, target, mask_attention):
        # pred     [bs, 56, N, N]
        # target   [bs, 56, N, N]
        # mask_attention   [bs, N]
        losses = {}
        bs, nb_cls, N, N = pred.shape
        
        mask = torch.zeros_like(pred).to(pred.device)
        for idx in range(bs):
            n = torch.sum(mask_attention[idx]).to(torch.int)
            mask[idx, :, :n, :n] = 1
        pred = pred * mask - 9999 * (1 - mask)

        if self.loss_mode == 'v1':
            input_tensor = pred.reshape([bs*nb_cls, -1])
            target_tensor = target.reshape([bs*nb_cls, -1])
            loss = self.multilabel_categorical_crossentropy(target_tensor, input_tensor)
        elif self.loss_mode == 'v2':
            input_tensor = pred.reshape([bs, -1])
            target_tensor = target.reshape([bs, -1])
            loss = self.multilabel_categorical_crossentropy(target_tensor, input_tensor)
        elif self.loss_mode == 'v3':
            input_tensor = pred.permute([0, 2, 1, 3]).reshape([bs*N, -1])
            target_tensor = target.permute([0, 2, 1, 3]).reshape([bs*N, -1])
            loss = self.multilabel_categorical_crossentropy(target_tensor, input_tensor)
        elif self.loss_mode == 'v4':
            assert pred.shape[0] == 1 and target.shape[0] == 1
            input_tensor = pred.reshape([nb_cls, -1])
            target_tensor = target.reshape([nb_cls, -1])
            loss = self.multilabel_categorical_crossentropy(target_tensor, input_tensor)
            # accumulate the samples for each category
            for u_l in range(self.num_cls):
                self.cum_samples[u_l] += target_tensor[u_l].sum()
            loss_weight = self.cum_samples.clamp(min=1).sum() / self.cum_samples.clamp(min=1)
            loss_weight = loss_weight.clamp(max=1000)
            loss = loss * loss_weight
        elif self.loss_mode == 'v5':
            assert pred.shape[0] == 1 and target.shape[0] == 1
            input_tensor = pred.reshape([nb_cls, -1])
            target_tensor = target.reshape([nb_cls, -1])
            loss = self.multilabel_categorical_crossentropy(target_tensor, input_tensor)
            weight = (loss / loss.max()) ** self.loss_alpha
            loss = loss * weight
        elif self.loss_mode == 'v6':
            assert pred.shape[0] == 1 and target.shape[0] == 1
            input_tensor = pred.reshape([nb_cls*N, -1])
            target_tensor = target.reshape([nb_cls*N, -1])
            loss = self.multilabel_categorical_crossentropy(target_tensor, input_tensor)
            weight = (loss / loss.max()) ** self.loss_alpha
            loss = loss * weight

        loss = loss.mean()
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
        recall = self.get_recall_N(pred, target, mask, N=20)
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



@HEADS.register_module()
class MultiHeadCls(BaseModule):
    def __init__(
        self, 
        pretrained_transformers=None, 
        cache_dir=None,
        input_feature_size=256,
        layers_transformers=6,
        feature_size=768,
        num_cls=56,
        cls_qk_size=512,
        loss_weight=1.,
        num_entity_max=30,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.cls_qk_size = cls_qk_size
        self.fc_input = nn.Sequential(
            nn.Linear(input_feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )
        self.cls_q = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.cls_k = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.loss_weight = loss_weight
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.num_entity_max = num_entity_max


    def forward(self,inputs_embeds, attention_mask=None):
        encode_embedding = self.fc_input(inputs_embeds)
        bs, N, c = encode_embedding.shape
        q_embedding = self.cls_q(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        k_embedding = self.cls_k(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        cls_pred = q_embedding @ torch.transpose(k_embedding, 2, 3) / self.cls_qk_size ** 0.5
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


    def loss(self, pred, target, mask_attention):
        # pred     [bs, 56, N, N]
        # target   [bs, 56, N, N]
        # mask_attention   [bs, N]
        losses = {}
        bs, nb_cls, N, N = pred.shape
        
        mask = torch.zeros_like(pred).to(pred.device)
        for idx in range(bs):
            n = torch.sum(mask_attention[idx]).to(torch.int)
            mask[idx, :, :n, :n] = 1
        pred = pred * mask - 9999 * (1 - mask)

        loss = self.multilabel_categorical_crossentropy(target.reshape([bs*nb_cls, -1]), pred.reshape([bs*nb_cls, -1]))
        loss = loss.mean()
        losses['loss_relationship'] = loss * self.loss_weight

        # f1, p, r
        [f1, precise, recall], [f1_mean, precise_mean, recall_mean] = self.get_f1_p_r(pred, target, mask)
        losses['rela.F1'] = f1
        losses['rela.precise'] = precise
        losses['rela.recall'] = recall
        losses['rela.F1_mean'] = f1_mean
        losses['rela.precise_mean'] = precise_mean
        losses['rela.recall_mean'] = recall_mean

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





