find_unused_parameters=True

num_relation = 56
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
depths = [2, 2, 18, 2]


model = dict(
    type='Mask2FormerRelation',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        use_abs_pos_embed=True,
    ),
    panoptic_head=dict(
        type='Mask2FormerRelationHead',
        in_channels=[128, 256, 512, 1024],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)
    ),
    relationship_head=dict(
        type='BertTransformer',
        pretrained_transformers='/mnt/mmtech01/usr/guiwan/workspace/model_dl/hfl/chinese-roberta-wwm-ext', 
        cache_dir='/mnt/mmtech01/usr/guiwan/workspace/mfpsg_output/tmp',
        input_feature_size=256,
        layers_transformers=6,
        feature_size=768,
        num_classes=num_classes,
        num_cls=num_relation,
        cls_qk_size=512,
        loss_weight=50,
        num_entity_max=50,
        use_background_feature=False,
    ),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        object_mask_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)

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
        img_scale=[(1500, 400), (1500, 1350)],
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
        ins_ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_instance_val.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_val.json',
        img_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        seg_prefix='/mnt/mmtech01/dataset/v_cocomask/psg/',
        ins_ann_file='/mnt/mmtech01/dataset/v_cocomask/psg/ann/psg_instance_val.json',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm', 'pq'], classwise=True)


backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
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


runner = dict(type='EpochBasedRunner', max_epochs=36)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'



# max_iters = 368750
# runner = dict(type='IterBasedRunner', max_iters=max_iters)
# learning policy
# lr_config = dict(
#     policy='step',
#     gamma=0.1,
#     by_epoch=False,
#     step=[327778, 355092],
#     warmup='linear',
#     warmup_by_epoch=False,
#     warmup_ratio=1.0,  # no warmup
#     warmup_iters=10)
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         dict(type='TensorboardLoggerHook', by_epoch=False)
#     ])
# custom_hooks = [dict(type='NumClassCheckHook')]
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# opencv_num_threads = 0
# mp_start_method = 'fork'
# auto_scale_lr = dict(enable=False, base_batch_size=12)
# interval = 100
# workflow = [('train', interval)]
# checkpoint_config = dict(
#     by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=3)
# dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
# evaluation = dict(
#     interval=interval,
#     dynamic_intervals=dynamic_intervals,
#     metric=['PQ', 'bbox', 'segm'])

load_from = '/mnt/mmtech01/usr/guiwan/workspace/model_dl/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic_20220329_230021-3bb8b482.pth'
resume_from = None
work_dir = '/mnt/mmtech01/usr/guiwan/workspace/mfpsg_output/v9'
