_base_ = [
    '../../../_base_/models/vitclip_base.py', '../../../_base_/default_runtime.py'
]
# model settings
num_frames=16
model = dict(
    backbone=dict(type='AIM_FLASH_WIN',drop_path_rate=0.2, adapter_scale=0.5, num_frames=num_frames,pretrained='openaiclip',
                use_flash_attn=True,checkpoint=False,prompt=True,wind_attn=True,window_size= (16,7,7),not_shift=True),
    cls_head=dict(num_classes=51),
    # test_cfg=dict(max_testing_views=8)
)

module_hooks = [
    dict(
        type='GPUNormalize',
        hooked_module='backbone',
        hook_pos='forward_pre',
        input_format='NCTHW',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321])
]

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/hmdb51/videos'
data_root_val = 'data/hmdb51/videos'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_videos.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_videos.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_videos.txt'
img_norm_cfg = dict(
    mean=[122.769, 116.74, 104.04], std=[68.493, 66.63, 70.321], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='FusedDecordInit',fast_rrc=True,rrc_params=(224, (0.4, 1.0)),hflip_prob=0.5),
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=1,frame_uniform=True),
    dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    # dict(type='Imgaug', transforms=[dict(type='RandAugment', n=4, m=7)]),
    # dict(
    #     type='PytorchVideoWrapper',
    #     op='RandAugment',
    #     magnitude=7,
    #     num_layers=4),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='RandomErasing', probability=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(type='FusedDecordInit',fast_rcc=True,cc_params=(224,)),
    
    dict(
        type='SampleFrames',
        clip_len=num_frames,
        frame_interval=4,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=num_frames,
        frame_interval=4,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

batchsize=16
data = dict(
    videos_per_gpu=batchsize,
    workers_per_gpu=2,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1,
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])


base_lr=3e-4

# actual_lr=base_lr*batchsize/64

# optimizer
optimizer = dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-2,
                 paramwise_cfg=dict(custom_keys={'class_embedding': dict(decay_mult=0.),
                                                 'positional_embedding': dict(decay_mult=0.),
                                                 'ln_1': dict(decay_mult=0.),
                                                 'ln_2': dict(decay_mult=0.),
                                                 'ln_pre': dict(decay_mult=0.),
                                                 'ln_post': dict(decay_mult=0.),}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)

total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1,max_keep_ckpts=1)

find_unused_parameters = False


project='vitclip_hmdb51'
name='aim_flash_tcls_7x16_gpunorm_16f'

work_dir = f'./work_dirs/hmdb51/{project}/{name}'


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True,
            ),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project=project, name=name,
                
        #         ),
        #     ),
        dict(type='TensorboardLoggerHook')
        ]
    )

# do not use mmdet version fp16
fp16 = None

optimizer_config = dict(
        type='GradientCumulativeFp16OptimizerHook',
        cumulative_iters=2
)


# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=2,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

# workflow = [('train', 1), ('val', 1)]
