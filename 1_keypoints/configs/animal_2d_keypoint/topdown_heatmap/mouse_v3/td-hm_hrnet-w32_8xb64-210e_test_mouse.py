_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=2000, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=5e-4,
    )
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False
    ),  # warm-up
    dict(
        type="MultiStepLR",
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best="coco/AP", rule="greater"))

# codec settings
codec = dict(type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(48, 96),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
            ),
        ),
        init_cfg=dict(
            type="Pretrained",
            # checkpoint="https://download.openmmlab.com/mmpose/"
            # "pretrain_models/hrnet_w48-8ef0771d.pth",
            prefix='backbone.',
            checkpoint="/media/vrlab/disk/skx_mouse/mmpose_mouse/v5_all/epoch_36.pth"
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=48,
        out_channels=25,
        deconv_out_channels=None,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=True,
    ),
)

dataset_type = "mouse_v4"
dataset_type2 = 'mouse_v4' ##test
data_mode = 'topdown'
data_root2 = '/media/vrlab/disk/skx_mouse/mmpose_mouse/data/mouse'

dataset2=dict(
        type=dataset_type2,
        data_root=data_root2,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='imgs/'),
        pipeline=[
            dict(
            type='KeypointConverter',
            num_keypoints=25,
            # mouse_v4 22pts -> mouse_v5 25pts
            mapping=[
                (0, 0),     # EarL
                (1, 1),     # EarR
                (2, 2),     # Nose
                (3, 3),     # Neck
                (4, 4),     # SpineM
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
            ])
        ],
    )

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders ##64
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CombinedDataset",
        metainfo=dict(from_file="configs/_base_/datasets/mouse_v5.py"),
        sample_ratio_factor=[0.5],
        datasets=[dataset2],
        pipeline=train_pipeline,
        test_mode=False,
    ),
)

mouse_converter = dict(
            type='KeypointConverter',
            num_keypoints=25,
            # mouse_v4 22pts -> mouse_v5 25pts
            mapping=[
                (0, 0),     # EarL
                (1, 1),     # EarR
                (2, 2),     # Nose
                (3, 3),     # Neck
                (4, 4),     # SpineM
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
            ])

mouse_test_dataset = dict(
    type=dataset_type2,
    data_root=data_root2,
    data_mode=data_mode,
    ann_file="annotations/test.json",
    data_prefix=dict(img="imgs/"),
    test_mode=True,
    pipeline=[],
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="CombinedDataset",
        metainfo=dict(from_file="configs/_base_/datasets/mouse_v5.py"),
        datasets=[mouse_test_dataset],
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    # type="MultiDatasetEvaluator",
    # metrics=[
    #     dict(
            type="CocoMetric",
            ann_file=data_root2 + "/annotations/test.json",
            # data_prefix=dict(img="imgs/"),
            gt_converter=mouse_converter,
    #     ),
    # ],
    # datasets=[rat_test_dataset],
)
test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='CocoMetric',
#     # dataset=dict(
#     #     type='CombinedDataset',
#     #     metainfo=dict(from_file='configs/_base_/datasets/mouse_v3.py'),
#     #     datasets=[dataset2_test,dataset1_test],
#     #     pipeline=val_pipeline,
#     #     test_mode=True,
#     # )
# )
