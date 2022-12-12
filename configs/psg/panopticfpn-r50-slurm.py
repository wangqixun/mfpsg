find_unused_parameters=True

num_relation = 56
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes


pretrained_transformers = '/mnt/mmtech01/usr/guiwan/workspace/model_dl/hfl/chinese-roberta-wwm-ext'
cache_dir = '/mnt/mmtech01/usr/guiwan/workspace/mfpsg_output/tmp'



# model settings
model = dict(
    type='PanopticFPNRelation',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.5)
    ),
    panoptic_fusion_head=dict(
        type='HeuristicFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),

    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5),
        panoptic=dict(
                    score_thr=0.6,
                    max_per_img=100,
                    mask_thr_binary=0.5,
                    mask_overlap=0.5,
                    nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
                    stuff_area_limit=4096)
    )
)



# dataset settings
dataset_type = 'CocoPanopticDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024//2, 1024//2)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        with_rela=True,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    # dict(
    #     type='Resize',
    #     img_scale=image_size,
    #     ratio_range=(0.1, 2.0),
    #     multiscale_mode='range',
    #     keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1500, 400), (1500, 1400)],
        # img_scale=[(960, 540), (640, 180)],
        multiscale_mode='range',
        keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_size=image_size,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip','flip_direction', 'img_norm_cfg', 'masks', 'gt_relationship'),

    ),
        
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        # img_scale=(1500, 1500),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

















































data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_tra.json',
        img_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        seg_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_val.json',
        img_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        seg_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        # ins_ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_instance_val.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_val.json',
        img_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        seg_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        # ins_ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_instance_val.json',
        pipeline=test_pipeline))
evaluation = dict(metric=['pq'], classwise=True)


backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
depths = [2, 2, 18, 2]
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0))


optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))


runner = dict(type='EpochBasedRunner', max_epochs=12)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 10])
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'




load_from = '/mnt/mmtech01/usr/guiwan/workspace/model_dl/panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155-5650f98b.pth'
resume_from = None
work_dir = '/mnt/mmtech01/usr/guiwan/workspace/mfpsg_output/panopticfpn-r50'
