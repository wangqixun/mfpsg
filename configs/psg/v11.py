_base_ = ['./mask2former_r50.py']
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

depths = [2, 2, 6, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
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
    relationship_head=dict(
        num_entity_max=50,
        use_background_feature=True,
        # entity_length=8,
        # entity_part_encoder='/share/wangqixun/workspace/bs/tx_mm/code/model_dl/hfl/chinese-roberta-wwm-ext',
        # entity_part_encoder_layers=6,
        # train_add_noise_mask=True,
        mask_shake=True,
    ),
    panoptic_head=dict(
        in_channels=[96, 192, 384, 768]),
    init_cfg=None,
    train_cfg=dict(
        freeze_layers = [
            # 'backbone',
            # 'panoptic_head',
            # 'panoptic_fusion_head',
        ]
    ),
)

find_unused_parameters = True

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
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
    paramwise_cfg=dict(custom_keys={}, norm_decay_mult=0.0))


load_from = '/share/wangqixun/workspace/github_project/mmdetection_checkpoint/model_dl/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic_20220326_224553-fc567107.pth'
resume_from = None
work_dir = '/share/wangqixun/workspace/bs/psg/mfpsg/output/v11'
