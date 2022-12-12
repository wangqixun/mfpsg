# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmdet.core import INSTANCE_OFFSET, bbox2roi, multiclass_nms
from mmdet.core.visualization import imshow_det_bboxes
from ..builder import DETECTORS, build_head
from ..roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from .two_stage import TwoStageDetector

import torch.nn as nn
import torch
import torch.nn.functional as F
from IPython import embed
from mmdet.utils import AvoidCUDAOOM
import random

from transformers import AutoTokenizer, AutoModel, AutoConfig


@DETECTORS.register_module()
class PanopticFPNRelation(TwoStageDetector):
    """Base class of Two-stage Panoptic Segmentor.

    As well as the components in TwoStageDetector, Panoptic Segmentor has extra
    semantic_head and panoptic_fusion_head.
    """

    def __init__(
            self,
            backbone,
            neck=None,
            rpn_head=None,
            roi_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None,
            # for panoptic segmentation
            semantic_head=None,
            panoptic_fusion_head=None,
            relationship_head=None,
    ):
        super(PanopticFPNRelation,
              self).__init__(backbone, neck, rpn_head, roi_head, train_cfg,
                             test_cfg, pretrained, init_cfg)
        if semantic_head is not None:
            self.semantic_head = build_head(semantic_head)
        if panoptic_fusion_head is not None:
            panoptic_cfg = test_cfg.panoptic if test_cfg is not None else None
            panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
            panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
            self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

            self.num_things_classes = self.panoptic_fusion_head.\
                num_things_classes
            self.num_stuff_classes = self.panoptic_fusion_head.\
                num_stuff_classes
            self.num_classes = self.panoptic_fusion_head.num_classes

        
        if relationship_head is not None:
            self.relationship_head = build_head(relationship_head)
            self.rela_cls_embed = nn.Embedding(self.relationship_head.num_classes, self.relationship_head.input_feature_size)
            self.num_entity_max = self.relationship_head.num_entity_max
            self.use_background_feature = self.relationship_head.use_background_feature
            
            self.entity_length = self.relationship_head.entity_length
            self.entity_part_encoder = self.relationship_head.entity_part_encoder
            self.entity_part_encoder_layers = self.relationship_head.entity_part_encoder_layers
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
            if hasattr(self.relationship_head, 'postional_encoding_layer') and self.relationship_head.postional_encoding_layer is not None:
                self.add_postional_encoding = True
            else:
                self.add_postional_encoding = False
            self.train_add_noise_mask = self.relationship_head.train_add_noise_mask
            self.embedding_add_cls = self.relationship_head.embedding_add_cls
            self.mask_shake = self.relationship_head.mask_shake   
            self.cls_embedding_mode = self.relationship_head.cls_embedding_mode         

    @property
    def with_relationship(self):
        """bool: whether the head has relationship head"""
        return hasattr(self, 'relationship_head') and self.relationship_head is not None


    @property
    def with_semantic_head(self):
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    @property
    def with_panoptic_fusion_head(self):
        return hasattr(self, 'panoptic_fusion_heads') and \
               self.panoptic_fusion_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        raise NotImplementedError(
            f'`forward_dummy` is not implemented in {self.__class__.__name__}')


    def _get_noise_embedding(self, feature):
        '''
        feature [bs, 256, h, w]
        '''
        c, h, w = feature.shape
        device = feature.device
        dtype = feature.dtype

        x_center = 0.1 + np.random.rand() * 0.8
        y_center = 0.1 + np.random.rand() * 0.8
        half_w_max = min(x_center, 1-x_center)
        half_w = np.random.rand() * half_w_max
        half_h_max = min(y_center, 1-y_center)
        half_h = np.random.rand() * half_h_max

        x1 = (x_center - half_w) * w
        x2 = (x_center + half_w) * w
        y1 = (y_center - half_h) * h
        y2 = (y_center + half_h) * h

        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)

        x1 = np.floor(x1).astype(int)
        x2 = np.ceil(x2).astype(int)
        y1 = np.floor(y1).astype(int)
        y2 = np.ceil(y2).astype(int)

        mask = feature.new_zeros([1, h, w])
        mask[:, y1:y2, x1:x2] = 1
        embedding = self._mask_pooling(feature, mask, output_size=1)
        # [self.entity_length, 256]
        return embedding

    def _id_is_thing(self, old_idx, gt_thing_label):
        if old_idx < len(gt_thing_label):
            return True
        else:
            return False

    def _tensor_mask_shake(self, mask):
        '''
        mask [1, h, w]
        '''
        mode = 'min' if np.random.rand() < 0.5 else 'max'
        ksize = np.random.randint(3, 15)
        if ksize % 2 == 0:
            ksize += 1

        bin_img = mask[None]
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)

        if np.random.rand() < 0.5:
            res_mask, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        else:
            res_mask, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
        res_mask = res_mask[0]
        return res_mask

    def _mask_pooling(self, feature, mask, output_size=1):
        '''
        feature [256, h, w]
        mask [1, h, w]
        output_size == 1: mean
        '''
        if mask.sum() <= 0:
            return feature.new_zeros([output_size, feature.shape[0]])
        mask_bool = (mask >= 0.5)[0]
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

        if self.mask_shake:
            gt_mask = self._tensor_mask_shake(gt_mask)

        embedding_thing = self._mask_pooling(feature, gt_mask, output_size=self.entity_length)  # [output_size, 256]
        cls_feature_thing = self.rela_cls_embed(gt_thing_label[idx: idx + 1].reshape([-1, ]))  # [1, 256]            

        if self.embedding_add_cls:
            if self.cls_embedding_mode == 'cat':
                embedding_thing = torch.cat([embedding_thing, cls_feature_thing], dim=-1)
            elif self.cls_embedding_mode == 'add':
                embedding_thing = embedding_thing + cls_feature_thing

        if self.add_postional_encoding:
            # [1, h, w]
            pos_embed_zeros = feature.new_zeros((1, ) + feature.shape[-2:])
            # [1, 256, h, w]
            pos_embed = self.relationship_head.postional_encoding_layer(pos_embed_zeros)
            pos_embed_mask_pooling = self._mask_pooling(pos_embed[0], gt_mask, output_size=self.entity_length)
            embedding_thing = embedding_thing + pos_embed_mask_pooling

        if self.use_background_feature:
            # background_mask = 1 - gt_mask
            # background_feature = feature[None] * background_mask[:, None]
            # background_feature = background_feature.sum(dim=[-2, -1]) / (background_mask[:, None].sum(dim=[-2, -1]) + 1e-8)
            background_feature = self._mask_pooling(feature, 1 - gt_mask, output_size=self.entity_length)  # [output_size, 256]
            embedding_thing = embedding_thing + background_feature

        # [output_size, 256]
        return embedding_thing

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

        if self.mask_shake:
            mask_staff = self._tensor_mask_shake(mask_staff)

        embedding_staff = self._mask_pooling(feature, mask_staff, output_size=self.entity_length)  # [output_size, 256]
        cls_feature_staff = self.rela_cls_embed(label_staff.reshape([-1, ]))  # [1, 256]

        if self.embedding_add_cls:
            if self.cls_embedding_mode == 'cat':
                embedding_staff = torch.cat([embedding_staff, cls_feature_staff], dim=-1)
            elif self.cls_embedding_mode == 'add':
                embedding_staff = embedding_staff + cls_feature_staff

        if self.add_postional_encoding:
            # [1, h, w]
            pos_embed_zeros = feature.new_zeros((1, ) + feature.shape[-2:])
            # [1, 256, h, w]
            pos_embed = self.relationship_head.postional_encoding_layer(pos_embed_zeros)
            pos_embed_mask_pooling = self._mask_pooling(pos_embed[0], mask_staff, output_size=self.entity_length)
            embedding_staff = embedding_staff + pos_embed_mask_pooling


        if self.use_background_feature:
            # background_mask = 1 - mask_staff
            # background_feature = feature[None] * background_mask[:, None]
            # background_feature = background_feature.sum(dim=[-2, -1]) / (background_mask[:, None].sum(dim=[-2, -1]) + 1e-8)
            background_feature = self._mask_pooling(feature, 1 - mask_staff, output_size=self.entity_length)  # [output_size, 256]
            embedding_staff = embedding_staff + background_feature
        
        # [output_size, 256]
        return embedding_staff

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
        device = feature.device
        dtype = feature.dtype

        masks = meta_info['masks']
        num_keep = min(self.num_entity_max, len(masks))
        keep_idx_list = random.sample(list(range(len(masks))), num_keep)
        old2new_dict = {old: new for new, old in enumerate(keep_idx_list)}

        embedding_list = []
        for idx, old in enumerate(keep_idx_list):
            if self._id_is_thing(old, gt_thing_label):
                embedding = self._thing_embedding(old, feature, gt_thing_mask, gt_thing_label, meta_info)
            else:
                embedding = self._staff_embedding(old, feature, masks, gt_semantic_seg)
            embedding_list.append(embedding[None])
        # [1, n * self.entity_length, 256]
        embedding = torch.cat(embedding_list, dim=1)

        if self.train_add_noise_mask:
            noise_embedding_list = []
            noise_class_list = []
            for idx, old in enumerate(keep_idx_list):
                if self._id_is_thing(old, gt_thing_label):
                    noise_class = gt_thing_label[old] # tensor
                else:
                    noise_class = torch.tensor(masks[old]['category']).to(device).to(torch.long)
                noise_embedding = self._get_noise_embedding(feature)  # [self.entity_length, 256]
                noise_class_list.append(noise_class)
                noise_embedding_list.append(noise_embedding[None])
            noise_class_tensor = torch.tensor(noise_class_list).to(device).reshape([-1, ])
            noise_class_embedding = self.rela_cls_embed(noise_class_tensor)  # [n, 256]
            noise_embedding_tensor = torch.cat(noise_embedding_list, dim=1)  # [1, n * self.entity_length, 256]

            noise_class_embedding = torch.repeat_interleave(noise_class_embedding, self.entity_length, dim=0)[None]
            noise_embedding_tensor = noise_embedding_tensor + noise_class_embedding

            embedding = torch.cat([embedding, noise_embedding_tensor], dim=1) # [1, 2*n*self.entity_length, 256]


        if self.entity_length > 1:
            embedding = self._entity_encode(embedding)

        target_relationship = feature.new_zeros([1, self.relationship_head.num_cls, embedding.shape[1], embedding.shape[1]])
        for ii, jj, cls_relationship in meta_info['gt_relationship'][0]:
            if not (ii in old2new_dict and jj in old2new_dict):
                continue
            new_ii, new_jj = old2new_dict[ii], old2new_dict[jj]
            target_relationship[0][cls_relationship][new_ii][new_jj] = 1

        # embedding [1, n, 256]
        # target_relationship [1, 56, n, n]
        return embedding, target_relationship




    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        semantic_loss = self.semantic_head.forward_train(x, gt_semantic_seg)
        losses.update(semantic_loss)


        embed()
        xxxxxxxxx
        mask_features = x[0]


        if self.with_relationship:

            relationship_input_embedding = []
            relationship_target = []

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

            max_length = max([e.shape[1] for e in relationship_input_embedding])
            mask_attention = mask_features.new_zeros([num_imgs, max_length])
            for idx in range(num_imgs):
                mask_attention[idx, :relationship_input_embedding[idx].shape[1]] = 1.
            relationship_input_embedding = [
                F.pad(e, [0, 0, 0, max_length-e.shape[1]])
                for e in relationship_input_embedding
            ]
            relationship_target = [
                F.pad(t, [0, max_length-t.shape[3], 0, max_length-t.shape[2]])
                for t in relationship_target
            ]
            relationship_input_embedding = torch.cat(relationship_input_embedding, dim=0)
            relationship_target = torch.cat(relationship_target, dim=0)
            relationship_output = self.relationship_head(relationship_input_embedding, mask_attention)
            loss_relationship = self.relationship_head.loss(relationship_output, relationship_target, mask_attention)
            losses.update(loss_relationship)



        torch.cuda.empty_cache()
        return losses

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        img_shapes = tuple(meta['ori_shape']
                           for meta in img_metas) if rescale else tuple(
                               meta['pad_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            masks = []
            for img_shape in img_shapes:
                out_shape = (0, self.roi_head.bbox_head.num_classes) \
                            + img_shape[:2]
                masks.append(det_bboxes[0].new_zeros(out_shape))
            mask_pred = det_bboxes[0].new_zeros((0, 80, 28, 28))
            mask_results = dict(
                masks=masks, mask_pred=mask_pred, mask_feats=None)
            return mask_results

        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        if rescale:
            if not isinstance(scale_factors[0], float):
                scale_factors = [
                    det_bboxes[0].new_tensor(scale_factor)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                _bboxes[i] * scale_factors[i] for i in range(len(_bboxes))
            ]

        mask_rois = bbox2roi(_bboxes)
        mask_results = self.roi_head._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

        # resize the mask_preds to (K, H, W)
        masks = []
        for i in range(len(_bboxes)):
            det_bbox = det_bboxes[i][:, :4]
            det_label = det_labels[i]

            mask_pred = mask_preds[i].sigmoid()

            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, det_label][:, None]

            img_h, img_w, _ = img_shapes[i]
            mask_pred, _ = _do_paste_mask(
                mask_pred, det_bbox, img_h, img_w, skip_empty=False)
            masks.append(mask_pred)

        mask_results['masks'] = masks

        return mask_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without Augmentation."""
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        bboxes, scores = self.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, None, rescale=rescale)

        pan_cfg = self.test_cfg.panoptic
        # class-wise predictions
        det_bboxes = []
        det_labels = []
        for bboxe, score in zip(bboxes, scores):
            det_bbox, det_label = multiclass_nms(bboxe, score,
                                                 pan_cfg.score_thr,
                                                 pan_cfg.nms,
                                                 pan_cfg.max_per_img)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        mask_results = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)
        masks = mask_results['masks']

        seg_preds = self.semantic_head.simple_test(x, img_metas, rescale)

        results = []
        for i in range(len(det_bboxes)):
            pan_results = self.panoptic_fusion_head.simple_test(
                det_bboxes[i], det_labels[i], masks[i], seg_preds[i])
            pan_results = pan_results.int().detach().cpu().numpy()
            result = dict(pan_results=pan_results)
            results.append(result)
        return results

    def show_result(self,
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
        """Draw `result` over `img`.

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
