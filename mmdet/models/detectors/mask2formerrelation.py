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
                 init_cfg=None):
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
            self.rela_cls_embed = nn.Embedding(self.relationship_head.num_classes, self.relationship_head.input_feature_size)
            self.num_entity_max = self.relationship_head.num_entity_max
            self.use_background_feature = self.relationship_head.use_background_feature


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

            num_imgs = len(img_metas)
            for idx in range(num_imgs):                
                meta_info = img_metas[idx]

                masks = meta_info['masks']
                if len(masks) > self.num_entity_max:
                    masks = masks[:self.num_entity_max]

                # feature
                feature = mask_features[idx]

                # thing mask
                gt_mask = gt_masks[idx].to_ndarray()
                gt_mask = gt_mask[:self.num_entity_max]
                gt_mask = torch.from_numpy(gt_mask).to(device).to(dtype)
                gt_label = gt_labels[idx]
                if gt_mask.shape[0] != 0:
                    h_img, w_img = meta_info['img_shape'][:2]
                    gt_mask = F.interpolate(gt_mask[:, None], size=(h_img, w_img))[:, 0]
                    h_pad, w_pad = meta_info['pad_shape'][:2]
                    gt_mask = F.pad(gt_mask[:, None], (0, w_pad-w_img, 0, h_pad-h_img))[:, 0]
                    h_feature, w_feature = mask_features.shape[-2:]
                    gt_mask = F.interpolate(gt_mask[:, None], size=(h_feature, w_feature))[:, 0]

                    # thing feature
                    feature_thing = feature[None] * gt_mask[:, None]
                    embedding_thing = feature_thing.sum(dim=[-2, -1]) / (gt_mask[:, None].sum(dim=[-2, -1]) + 1e-8)
                    cls_feature_thing = self.rela_cls_embed(gt_label.reshape([-1, ]))
                    embedding_thing = embedding_thing + cls_feature_thing
                    if self.use_background_feature:
                        background_mask = 1 - gt_mask
                        background_feature = feature[None] * background_mask[:, None]
                        background_feature = background_feature.sum(dim=[-2, -1]) / (background_mask[:, None].sum(dim=[-2, -1]) + 1e-8)                 
                        embedding_thing = embedding_thing + background_feature
                else:
                    embedding_thing = None


                # staff mask
                mask_staff = []
                label_staff = []
                gt_semantic_seg_idx = gt_semantic_seg[idx]
                for idx_stuff in range(len(gt_masks[idx]), len(masks), 1):
                    category_staff = masks[idx_stuff]['category']
                    mask = gt_semantic_seg_idx == category_staff
                    mask_staff.append(mask)
                    label_staff.append(category_staff)
                if len(mask_staff) != 0:
                    mask_staff = torch.cat(mask_staff, dim=0)
                    mask_staff = mask_staff.to(dtype)
                    mask_staff = F.interpolate(mask_staff[None], size=(feature.shape[1], feature.shape[2]))[0]
                    label_staff = torch.tensor(label_staff).to(device).to(torch.long)
                    # staff feature
                    feature_staff = feature[None] * mask_staff[:, None]
                    cls_feature_staff = self.rela_cls_embed(label_staff.reshape([-1, ]))
                    embedding_staff = feature_staff.sum(dim=[-2, -1]) / (mask_staff[:, None].sum(dim=[-2, -1]) + 1e-8)
                    embedding_staff = embedding_staff + cls_feature_staff
                    if self.use_background_feature:
                        background_mask = 1 - mask_staff
                        background_feature = feature[None] * background_mask[:, None]
                        background_feature = background_feature.sum(dim=[-2, -1]) / (background_mask[:, None].sum(dim=[-2, -1]) + 1e-8)
                        embedding_staff = embedding_staff + background_feature
                else:
                    embedding_staff = None

                # final embedding
                embedding_list = []
                if embedding_thing is not None:
                    embedding_list.append(embedding_thing)
                if embedding_staff is not None:
                    embedding_list.append(embedding_staff)
                if len(embedding_list) != 0:
                    embedding = torch.cat(embedding_list, dim=0)
                    embedding = embedding[None]
                else:
                    embedding = None
                
                if embedding is not None:
                    relationship_input_embedding.append(embedding)
                    target_relationship = torch.zeros([1, self.relationship_head.num_cls, embedding.shape[1], embedding.shape[1]]).to(device)
                    for ii, jj, cls_relationship in meta_info['gt_relationship'][0]:
                        if not (ii < embedding.shape[1] and jj < embedding.shape[1]):
                            continue
                        target_relationship[0][cls_relationship][ii][jj] = 1
                    relationship_target.append(target_relationship)
                else:
                    continue

            if len(relationship_input_embedding) != 0:

                max_length = max([e.shape[1] for e in relationship_input_embedding])
                mask_attention = torch.zeros([num_imgs, max_length]).to(device)
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

                relationship_input_embedding = relationship_input_embedding[:, :self.num_entity_max, :]
                relationship_target = relationship_target[:, :, :self.num_entity_max, :self.num_entity_max]
                mask_attention = mask_attention[:, :self.num_entity_max]

                relationship_output = self.relationship_head(relationship_input_embedding, mask_attention)
                loss_relationship = self.relationship_head.loss(relationship_output, relationship_target, mask_attention)
                losses.update(loss_relationship)

        return losses

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

        entity_embedding = (feature_map * mask_tensor).sum(dim=[2, 3]) / (mask_tensor.sum(dim=[2, 3]) + 1e-8)
        entity_embedding = entity_embedding[None]
        entity_embedding = entity_embedding + cls_entity_embedding

        return entity_embedding, entity_id_list, entity_score_list


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

        relation_res = []
        if entity_res is not None:
            entity_embedding, entityid_list, entity_score_list = entity_res
            relationship_output = self.relationship_head(entity_embedding, attention_mask=None)
            relationship_output = relationship_output[0]
            for idx_i in range(relationship_output.shape[1]):
                relationship_output[:, idx_i, idx_i] = -9999
            relationship_output = torch.exp(relationship_output)

            # relationship_output x subject score x object score
            entity_score_tensor = torch.tensor(entity_score_list, device=device, dtype=dtype)
            relationship_output = relationship_output * entity_score_tensor[None, :, None]
            relationship_output = relationship_output * entity_score_tensor[None, None, :]

            # find topk
            _, topk_indices = torch.topk(relationship_output.reshape([-1,]), k=20)

            # subject, object, cls
            for index in topk_indices:
                pred_cls = index // (relationship_output.shape[1] ** 2)
                index_subject_object = index % (relationship_output.shape[1] ** 2)
                pred_subject = index_subject_object // relationship_output.shape[1]
                pred_object = index_subject_object % relationship_output.shape[1]
                pred = [pred_subject.item(), pred_object.item(), pred_cls.item()]
                relation_res.append(pred)
            
        rl = dict(
            entityid_list=[eid.item() for eid in entityid_list],
            relation=relation_res,
        )

        res['rela_results'] = rl
        res['pan_results'] = res['pan_results'].detach().cpu().numpy()

        return [res]
        
        # for i in range(len(results)):
        #     if 'pan_results' in results[i]:
        #         results[i]['pan_results'] = results[i]['pan_results'].detach(
        #         ).cpu().numpy()

        #     if 'ins_results' in results[i]:
        #         labels_per_image, bboxes, mask_pred_binary = results[i][
        #             'ins_results']
        #         bbox_results = bbox2result(bboxes, labels_per_image,
        #                                    self.num_things_classes)
        #         mask_results = [[] for _ in range(self.num_things_classes)]
        #         for j, label in enumerate(labels_per_image):
        #             mask = mask_pred_binary[j].detach().cpu().numpy()
        #             mask_results[label].append(mask)
        #         results[i]['ins_results'] = bbox_results, mask_results

        #     assert 'sem_results' not in results[i], 'segmantic segmentation '\
        #         'results are not supported yet.'

        # if self.num_stuff_classes == 0:
        #     results = [res['ins_results'] for res in results]

        # return results
