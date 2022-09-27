# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np

from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.core.visualization import imshow_det_bboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector

import torch.nn as nn
import torch
import torch.nn.functional as F
from IPython import embed
from mmdet.utils import AvoidCUDAOOM
import random

from transformers import AutoTokenizer, AutoModel, AutoConfig


@DETECTORS.register_module()
class MaskFormerRelation(SingleStageDetector):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 relationship_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 bs_size=64):
        super(SingleStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)

        panoptic_fusion_head_ = copy.deepcopy(panoptic_fusion_head)
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # BaseDetector.show_result default for instance segmentation
        if self.num_stuff_classes > 0:
            self.show_result = self._show_pan_result
        
        if relationship_head is not None:
            self.relationship_head = build_head(relationship_head)
            self.rela_cls_embed = nn.Embedding(self.relationship_head.num_classes + 1, self.relationship_head.input_feature_size)
            self.num_entity_max = self.relationship_head.num_entity_max
            self.use_background_feature = self.relationship_head.use_background_feature
            
            self.entity_length = self.relationship_head.entity_length
            self.entity_part_encoder = self.relationship_head.entity_part_encoder
            self.entity_part_encoder_layers = self.relationship_head.entity_part_encoder_layers
            print("entity length = {}".format(self.entity_length))
            if self.entity_length > 1:
                self.entity_model = AutoModel.from_pretrained(self.entity_part_encoder, cache_dir=self.relationship_head.cache_dir)
                self.entity_model.encoder.layer = self.entity_model.encoder.layer[:self.entity_part_encoder_layers]
                self.entity_model.embeddings.word_embeddings = None
                self.entity_encode_fc_input = nn.Sequential(
                    nn.Linear(self.relationship_head.input_feature_size, self.relationship_head.feature_size),
                    nn.LayerNorm(self.relationship_head.feature_size),
                )
                self.entity_encode_fc_output = nn.Sequential(
                    nn.Linear(self.relationship_head.feature_size, self.relationship_head.feature_size),
                    nn.LayerNorm(self.relationship_head.feature_size),
                )

        
        self.relu = nn.ReLU(inplace=True)
        self.bs_size = bs_size

    @property
    def with_relationship(self):
        """bool: whether the head has relationship head"""
        return hasattr(self, 'relationship_head') and self.relationship_head is not None

    def forward_dummy(self, img, img_metas):
        """Used for computing network flops. See
        `mmdetection/tools/analysis_tools/get_flops.py`

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        outs = self.panoptic_head(x, img_metas)
        return outs

    def _id_is_thing(self, old_idx, gt_thing_label):
        if old_idx < len(gt_thing_label):
            return True
        else:
            return False

    def _mask_pooling(self, feature, mask, output_size=1):
        '''
        feature [256, h, w]
        mask [1, h, w]
        output_size == 1: mean
        '''
        if mask.sum() <= 0:
            return feature.new_zeros([output_size, feature.shape[0]])
        
        # confid_thresh = np.random.uniform(0.35, 0.65)
        confid_thresh = 0.5
        mask_bool = (mask >= confid_thresh)[0]
        feats = feature[:, mask_bool]
        if feats.shape[1] < output_size:
            feats = torch.cat([feats] * int(np.ceil(output_size/feats.shape[1])), dim=1)
            feats = feats[:, :output_size]
        
        split_list = [feats.shape[1] // output_size] * output_size
        for idx in range(feats.shape[1] - sum(split_list)):
            split_list[idx] += 1
        feats_list = torch.split(feats, split_list, dim=1)
        feats_mean_list = [feat.mean(dim=1)[None] for feat in feats_list]
        feats_tensor = torch.cat(feats_mean_list, dim=0)
        # [output_size, 256]
        return feats_tensor

    def _thing_embedding(self, idx, feature, gt_thing_mask, gt_thing_label, meta_info):
        device = feature.device
        dtype = feature.dtype

        gt_mask = gt_thing_mask.to_ndarray()
        gt_mask = gt_mask[idx: idx + 1]
        gt_mask = torch.from_numpy(gt_mask).to(device).to(dtype)

        h_img, w_img = meta_info['img_shape'][:2]
        gt_mask = F.interpolate(gt_mask[:, None], size=(h_img, w_img))[:, 0]
        h_pad, w_pad = meta_info['pad_shape'][:2]
        gt_mask = F.pad(gt_mask[:, None], (0, w_pad-w_img, 0, h_pad-h_img))[:, 0]
        h_feature, w_feature = feature.shape[-2:]
        gt_mask = F.interpolate(gt_mask[:, None], size=(h_feature, w_feature))[:, 0]


        # feature_thing = feature[None] * gt_mask[:, None]
        # embedding_thing = feature_thing.sum(dim=[-2, -1]) / (gt_mask[:, None].sum(dim=[-2, -1]) + 1e-8)
        embedding_thing = self._mask_pooling(feature, gt_mask, output_size=self.entity_length)  # [output_size, 256]
        cls_feature_thing = self.rela_cls_embed(gt_thing_label[idx: idx + 1].reshape([-1, ]))  # [1, 256]

        embedding_thing = embedding_thing + cls_feature_thing

        if self.use_background_feature:
            # background_mask = 1 - gt_mask
            # background_feature = feature[None] * background_mask[:, None]
            # background_feature = background_feature.sum(dim=[-2, -1]) / (background_mask[:, None].sum(dim=[-2, -1]) + 1e-8)      
            # 这里面featue是已经找出了这个类的featue吗？？？           
            background_feature = self._mask_pooling(feature, torch.ones_like(feature), output_size=self.entity_length)  # [output_size, 256]
            # embedding_thing = embedding_thing + background_feature
            # 一种新的背景融合方式
            cls_feature_background = self.rela_cls_embed(torch.tensor([56]).to(device))  # [1, 256], 56代表背景

        # [output_size, 256]
        return embedding_thing, cls_feature_background

    def _staff_embedding(self, idx, feature, masks, gt_semantic_seg):
        device = feature.device
        dtype = feature.dtype

        category_staff = masks[idx]['category']
        mask_staff = gt_semantic_seg == category_staff
        mask_staff = mask_staff.to(dtype)
        mask_staff = F.interpolate(mask_staff[None], size=(feature.shape[1], feature.shape[2]))[0]
        label_staff = torch.tensor(category_staff).to(device).to(torch.long)

        # feature_staff = feature[None] * mask_staff[:, None]
        # embedding_staff = feature_staff.sum(dim=[-2, -1]) / (mask_staff[:, None].sum(dim=[-2, -1]) + 1e-8)
        embedding_staff = self._mask_pooling(feature, mask_staff, output_size=self.entity_length)  # [output_size, 256]
        cls_feature_staff = self.rela_cls_embed(label_staff.reshape([-1, ]))  # [1, 256]

        embedding_staff = embedding_staff + cls_feature_staff

        if self.use_background_feature:
            # background_mask = 1 - mask_staff
            # background_feature = feature[None] * background_mask[:, None]
            # background_feature = background_feature.sum(dim=[-2, -1]) / (background_mask[:, None].sum(dim=[-2, -1]) + 1e-8)
            # 这里面featue是已经找出了这个类的featue吗？？？           
            background_feature = self._mask_pooling(feature, torch.ones_like(feature), output_size=self.entity_length)  # [output_size, 256]
            # embedding_staff = embedding_staff + background_feature
            cls_feature_background = self.rela_cls_embed(torch.tensor([56]).to(device))  # [1, 256], 56代表背景
        
        # [output_size, 256]
        return embedding_staff, cls_feature_background

    def _entity_encode(self, inputs_embeds):
        '''
        inputs_embeds [1, n * self.entity_length, 256]
        '''
        num_entity = inputs_embeds.shape[1] // self.entity_length

        position_ids = torch.ones([1, inputs_embeds.shape[1]]).to(inputs_embeds.device).to(torch.long)
        encode_inputs_embeds = self.entity_encode_fc_input(inputs_embeds)
        encode_res = self.entity_model(inputs_embeds=encode_inputs_embeds, attention_mask=None, position_ids=position_ids)
        encode_embedding = encode_res['last_hidden_state']
        encode_embedding = self.entity_encode_fc_output(encode_embedding)

        split_list = [self.entity_length] * num_entity
        encode_embedding_list = torch.split(encode_embedding, split_list, dim=1)

        encode_embedding = [e.mean(dim=1) for e in encode_embedding_list]
        encode_embedding = torch.cat(encode_embedding, dim=0)
        encode_embedding = encode_embedding[None]
        # [1, n, 256]
        return encode_embedding

    def _get_entity_embedding_and_target(self, feature, meta_info, gt_thing_mask, gt_thing_label, gt_semantic_seg):
        # feature: [256, h, w]
        # meta_info: dict
        # gt_thing_mask: bitmap, n
        # gt_thing_label: 

        masks = meta_info['masks']
        device = feature.device

        embedding_list = []
        for idx in range(len(masks)):
            if self._id_is_thing(idx, gt_thing_label):
                embedding, cls_feature_background = self._thing_embedding(idx, feature, gt_thing_mask, gt_thing_label, meta_info)
            else:
                embedding, cls_feature_background = self._staff_embedding(idx, feature, masks, gt_semantic_seg)
            embedding_list.append(embedding)
        
        # idx2label映射关系建立
        idx2label = dict()
        for ii, jj, cls_relationship in meta_info['gt_relationship'][0]:
            idx2label[(ii, jj)] = cls_relationship

        # 改结构，就要改输入
        combine_embedding = []
        target_relationship = []
        for idx_i in range(len(embedding_list)):
            for idx_j in range(len(embedding_list)):
                if idx_i == idx_j:
                    continue
                
                if (idx_i, idx_j) in idx2label:
                    target_relationship.append(idx2label[(idx_i, idx_j)])
                else:
                    target_relationship.append(56)


                comb = torch.stack([embedding_list[idx_i], embedding_list[idx_j], cls_feature_background], dim=1)
                if self.entity_length > 1:
                    comb2 = self._entity_encode(comb)
                combine_embedding.append(comb)


        # embedding [1, n, 256]
        # target_relationship [x, 57]
        target_relationship = torch.tensor(target_relationship, dtype=torch.long).to(device)
        combine_embedding = torch.cat(combine_embedding, dim=0)

        return combine_embedding, target_relationship




    # @AvoidCUDAOOM.retry_if_cuda_oom
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor]): semantic segmentation mask for
                images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses, mask_features = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_masks,
                                                  gt_semantic_seg,
                                                  gt_bboxes_ignore)

        device = mask_features.device
        dtype = mask_features.dtype

        if self.with_relationship:

            relationship_input_embedding = []
            relationship_target = []
            bg_feature_list = []

            num_imgs = len(img_metas)
            
            for idx in range(num_imgs):
                embedding, target_relationship = self._get_entity_embedding_and_target(
                    mask_features[idx],
                    img_metas[idx],
                    gt_masks[idx],
                    gt_labels[idx],
                    gt_semantic_seg[idx],
                )
                relationship_input_embedding.append(embedding)
                relationship_target.append(target_relationship)

            

            relationship_input_embedding = torch.cat(relationship_input_embedding, dim=0)
            rand_idx = torch.randint(relationship_input_embedding.shape[0], (self.bs_size,))
            relationship_input_embedding = relationship_input_embedding[rand_idx]
            relationship_target = torch.cat(relationship_target, dim=0)[rand_idx]

            relationship_output = self.relationship_head(relationship_input_embedding)
            loss_relationship = self.relationship_head.loss(relationship_output, relationship_target)
            losses.update(loss_relationship)



        torch.cuda.empty_cache()
        return losses

    # @AvoidCUDAOOM.retry_if_cuda_oom
    def simple_test(self, imgs, img_metas, **kwargs):
        """Test without augmentation.

        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.

        Returns:
            list[dict[str, np.array | tuple[list]] | tuple[list]]:
                Semantic segmentation results and panoptic segmentation \
                results of each image for panoptic segmentation, or formatted \
                bbox and mask results of each image for instance segmentation.

            .. code-block:: none

                [
                    # panoptic segmentation
                    {
                        'pan_results': np.array, # shape = [h, w]
                        'ins_results': tuple[list],
                        # semantic segmentation results are not supported yet
                        'sem_results': np.array
                    },
                    ...
                ]

            or

            .. code-block:: none

                [
                    # instance segmentation
                    (
                        bboxes, # list[np.array]
                        masks # list[list[np.array]]
                    ),
                    ...
                ]
        """
        feats = self.extract_feat(imgs)
        need_mask_features = False
        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(
            feats, img_metas, need_mask_features, **kwargs)
        results = self.panoptic_fusion_head.simple_test(
            mask_cls_results, mask_pred_results, img_metas, **kwargs)
        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach(
                ).cpu().numpy()

            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i][
                    'ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image,
                                           self.num_things_classes)
                mask_results = [[] for _ in range(self.num_things_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = bbox_results, mask_results

            assert 'sem_results' not in results[i], 'segmantic segmentation '\
                'results are not supported yet.'

        if self.num_stuff_classes == 0:
            results = [res['ins_results'] for res in results]

        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    def _show_pan_result(self,
                         img,
                         result,
                         score_thr=0.3,
                         bbox_color=(72, 101, 241),
                         text_color=(72, 101, 241),
                         mask_color=None,
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=False,
                         wait_time=0,
                         out_file=None):
        """Draw `panoptic result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes  # for VOID label
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            segms=segms,
            labels=labels,
            class_names=self.CLASSES,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img


@DETECTORS.register_module()
class Mask2FormerRelation(MaskFormerRelation):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 relationship_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            relationship_head=relationship_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)


@DETECTORS.register_module()
class Mask2FormerRelationForinfer(MaskFormerRelation):

    def get_entity_embedding(self, pan_result, entity_id_list, entity_score_list, feature_map, meta):
        device = feature_map.device
        dtype = feature_map.dtype

        ori_height, ori_width = meta['ori_shape'][:2]
        resize_height, resize_width = meta['img_shape'][:2]
        pad_height, pad_width = meta['pad_shape'][:2]

        mask_list = []
        class_mask_list = []
        instance_id_all = entity_id_list
        for idx_instance, instance_id in enumerate(instance_id_all):
            if instance_id == 133:
                continue
            mask = pan_result == instance_id
            class_mask = instance_id % INSTANCE_OFFSET
            # class_score = entity_score_list[idx_instance]
            mask_list.append(mask)
            class_mask_list.append(class_mask)

        if len(mask_list) == 0:
            return None
        
        class_mask_tensor = torch.tensor(class_mask_list).to(device).to(torch.long)[None]
        cls_entity_embedding = self.rela_cls_embed(class_mask_tensor)

        mask_tensor = torch.stack(mask_list)[None]
        mask_tensor = (mask_tensor * 1).to(dtype)
        h_img, w_img = resize_height, resize_width
        mask_tensor = F.interpolate(mask_tensor, size=(h_img, w_img))
        h_pad, w_pad = pad_height, pad_width
        mask_tensor = F.pad(mask_tensor, (0, w_pad-w_img, 0, h_pad-h_img))
        h_feature, w_feature = feature_map.shape[-2:]
        mask_tensor = F.interpolate(mask_tensor, size=(h_feature, w_feature))
        mask_tensor = mask_tensor[0][:, None]

        # feature_map [bs, 256, h, w]
        # mask_tensor [n, 1, h, w]
        combine_embedding = []
        pred_idx2mask_idx = dict()
        if self.entity_length > 1:
            embedding_list = []
            for idx in range(len(mask_list)):
                # embedding [self.entity_length, 256]
                embedding = self._mask_pooling(feature_map[0], mask_tensor[idx], self.entity_length)
                embedding = embedding + cls_entity_embedding[0, idx:idx+1]
                if self.use_background_feature:
                    background_embedding = self._mask_pooling(feature_map[0], 1 - mask_tensor[idx], self.entity_length)
                    # embedding = embedding + background_embedding
                    cls_feature_background = self.rela_cls_embed(torch.tensor([56]).to(device))  # [1, 256], 56代表背景

                embedding_list.append(embedding)

            # 改结构，就要改输入
            
            for idx_i in range(len(embedding_list)):
                for idx_j in range(len(embedding_list)):
                    if idx_i == idx_j:
                        continue
                    
                    
                    comb = torch.stack([embedding_list[idx_i], embedding_list[idx_j], cls_feature_background], dim=1)
                    if self.entity_length > 1:
                        comb2 = self._entity_encode(comb)
                    pred_idx2mask_idx[len(combine_embedding)] = (idx_i, idx_j)
                    combine_embedding.append(comb)
            # entity_embedding [1, n, 256]
        else:
            entity_embedding = (feature_map * mask_tensor).sum(dim=[2, 3]) / (mask_tensor.sum(dim=[2, 3]) + 1e-8)
            embedding_list = entity_embedding + cls_entity_embedding
            embedding_list = embedding_list[0]

            if self.use_background_feature:
                # entity_embedding = entity_embedding + background_feature
                cls_feature_background = self.rela_cls_embed(torch.tensor([56]).to(device))  # [1, 256], 56代表背景
            for idx_i in range(len(embedding_list)):
                for idx_j in range(len(embedding_list)):
                    if idx_i == idx_j:
                        continue

                    comb = torch.stack([embedding_list[idx_i].unsqueeze(0), embedding_list[idx_j].unsqueeze(0), cls_feature_background], dim=1)
                    pred_idx2mask_idx[len(combine_embedding)] = (idx_i, idx_j)
                    combine_embedding.append(comb)

        if len(combine_embedding) == 0:
            print("cannot get feature for rel predict, metainfo: {}".format(meta))
            return None

        combine_embedding = torch.cat(combine_embedding, dim=0)

        return combine_embedding, entity_id_list, entity_score_list, pred_idx2mask_idx


    def simple_test(self, imgs, img_metas, **kwargs):
        feats = self.extract_feat(imgs)
        need_mask_features = True
        mask_cls_results, mask_pred_results, mask_features = self.panoptic_head.simple_test(
            feats, img_metas, need_mask_features, **kwargs)
        results = self.panoptic_fusion_head.simple_test(
            mask_cls_results, mask_pred_results, img_metas, **kwargs)

        device = mask_features.device
        dtype = mask_features.dtype

        res = results[0]
        pan_results = res['pan_results']
        entityid_list = res['entityid_list']
        entity_score_list = res['entity_score_list']

        entity_res = self.get_entity_embedding(
            pan_result=pan_results,
            entity_id_list=entityid_list,
            entity_score_list=entity_score_list,
            feature_map=mask_features,
            meta=img_metas[0]
        )

        if not hasattr(self, 'rela_cls_ratio'):
            rela_cls_ratio = [
                14.90037819733397, 4.264230886861559, 17.104291458111103, 20.223487024176283, 3.7487785046659257, 
                7.82018096779728, 1.4676592792618066, 0.054454448480870075, 0.003729756745265074, 0.044011129594127875, 
                0.07161132950908942, 2.7063114943643374, 0.43787344189411964, 0.076460013277934, 6.994785800070119, 
                0.5661770739312382, 2.030479572122306, 0.3241158611635349, 0.0302110296366471, 0.06676264574024482, 
                1.112213461438045, 3.903563409594426, 0.88954698374572, 1.9957928343913411, 0.03356781070738566, 
                0.006340586466950625, 0.478527790417509, 0.043638153919601366, 0.02610829721685552, 0.0604220592732942, 
                0.050351716061078494, 0.005594635117897611, 0.13166041310785712, 0.014546051306533789, 0.004475708094318089, 
                0.008205464839583163, 0.7213349545342653, 0.36514318536145074, 0.029092102613067577, 0.1204711428720619, 
                0.0029838053962120592, 0.00708653781600364, 0.15963358869734517, 0.06825454843835084, 0.05184361875918452, 
                0.22975301550832855, 0.7687028651991317, 2.5343697084076173, 2.1576642771358454, 0.21334208582916223, 
                0.02797317558948805, 0.27712092617319495, 0.020513662098957906, 0.010443318886742206, 0.2237854047159044, 
                0.3099427855315276
            ]
            rela_cls_ratio = torch.tensor(rela_cls_ratio, dtype=dtype, device=device) / 100.
            self.rela_cls_ratio = rela_cls_ratio.reshape([-1, 1, 1])

        
        relation_res = []
        if entity_res is not None:
            entity_embedding, entityid_list, entity_score_list, pred_idx2mask_idx = entity_res
            relationship_output = self.relationship_head(entity_embedding, attention_mask=None)
            relationship_output = torch.nn.functional.softmax(relationship_output, dim=1)

            """
            # relationship weight
            for ratio_th, weight in [
                [[0, 0.01], 800], 
            #     # [[0, 0.01], 300],
            #     # [[0.001, 0.01], 300],
            ]:
                mask = ((self.rela_cls_ratio > ratio_th[0]) & (self.rela_cls_ratio < ratio_th[1])) * 1
                relationship_output = relationship_output * mask * weight + relationship_output * (1 - mask)
            """
            # for idx_rela in [8, 11, 19, 25, 26, 29, 31, 32, 34, 35, 36, 39, 40, 41, 42, 51, 53, 54]:
            # for idx_rela in [50, 25, 40, 29]:  # 0.5+
            #     relationship_output[idx_rela] = relationship_output[idx_rela] * 300
            # for idx_rela in [9, 27, 39, 8, 26, ]:  # 0.3+
            #     relationship_output[idx_rela] = relationship_output[idx_rela] * 150
            # for idx_rela in [18, 42, 13, 37, 43, 32]:  # 0.2+
            #     relationship_output[idx_rela] = relationship_output[idx_rela] * 150
            # for idx_rela in [12, 24, 55, 17, 44, ]:  # 0.1+
            #     relationship_output[idx_rela] = relationship_output[idx_rela] * 150

            # find topk
            label_pred = torch.argmax(relationship_output, dim=1)

            # subject, object, cls
            for idx in range(len(relationship_output)):
                tag = label_pred[idx].item()
                confid = relationship_output[idx][tag].item()
                if tag == 56:
                    continue
                
                pred_subj, pred_obj = pred_idx2mask_idx[idx]
                
                pred = [pred_subj, pred_obj, tag, confid]
                relation_res.append(pred)
            
            relation_res = sorted(relation_res, key=lambda x:x[-1], reverse=True)[:20]
            relation_res = [i[:3] for i in relation_res]
            
        rl = dict(
            entityid_list=[eid.item() for eid in entityid_list],
            relation=relation_res,
        )

        res['rela_results'] = rl
        res['pan_results'] = res['pan_results'].detach().cpu().numpy()

        return [res]
        
