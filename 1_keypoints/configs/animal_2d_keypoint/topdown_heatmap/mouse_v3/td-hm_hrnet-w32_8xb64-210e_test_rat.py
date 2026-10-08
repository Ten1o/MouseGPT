_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=1000, val_interval=10)

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
            checkpoint="https://download.openmmlab.com/mmpose/"
            "pretrain_models/hrnet_w48-8ef0771d.pth",
            # checkpoint="./v5_all/epoch_36.pth"
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

dataset_type = "rat"
dataset_type3 = "rat"  ##test 16
data_mode = "topdown"
data_root3 = "/home/vrlab/Documents/mmpose_mouse/data/rat_sync"

dataset3 = dict(
    type=dataset_type3,
    data_root=data_root3,
    data_mode=data_mode,
    ann_file="annotations/train.json",
    data_prefix=dict(img="imgs/"),
    pipeline=[
        dict(
            type="KeypointConverter",
            num_keypoints=25,
            # rat 20pts -> mouse_v5 25pts
            mapping=[
                (0, 0),  # EarL
                (1, 1),  # EarR
                (2, 2),  # Nose
                (3, 3),  # SpineF
                (4, 22),
                (5, 4),
                (8, 23),
                (9, 24),
                (10, 10),
                (11, 9),
                (12, 11),
                (13, 15),
                (14, 14),
                (15, 13),
                (16, 21),
                (17, 18),
                (18, 17),
                (19, 20),
            ],
        )
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
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CombinedDataset",
        metainfo=dict(from_file="configs/_base_/datasets/mouse_v5.py"),
        sample_ratio_factor=[1/256],
        datasets=[dataset3],
        pipeline=train_pipeline,
        test_mode=False,
    ),
)

test_dataset = dict(
    type=dataset_type3,
    data_root=data_root3,
    data_mode=data_mode,
    ann_file="annotations/test.json",
    data_prefix=dict(img="imgs/"),
    test_mode=True,
    pipeline=[
        dict(
            type="KeypointConverter",
            num_keypoints=25,
            # rat 20pts -> mouse_v5 25pts
            mapping=[
                (0, 0),  # EarL
                (1, 1),  # EarR
                (2, 2),  # Nose
                (3, 3),  # SpineF
                (4, 22),
                (5, 4),
                (8, 23),
                (9, 24),
                (10, 10),
                (11, 9),
                (12, 11),
                (13, 15),
                (14, 14),
                (15, 13),
                (16, 21),
                (17, 18),
                (18, 17),
                (19, 20),
            ],
        )
    ],
)

rat_converter = dict(
    type="KeypointConverter",
    num_keypoints=25,
    # rat 20pts -> mouse_v5 25pts
    mapping=[
            (0, 0),  # EarL
            (1, 1),  # EarR
            (2, 2),  # Nose
            (3, 3),  # SpineF
            (4, 22),
            (5, 4),
            (8, 23),
            (9, 24),
            (10, 10),
            (11, 9),
            (12, 11),
            (13, 15),
            (14, 14),
            (15, 13),
            (16, 21),
            (17, 18),
            (18, 17),
            (19, 20),
        ],
)

rat_test_dataset = dict(
    type=dataset_type3,
    data_root=data_root3,
    data_mode=data_mode,
    ann_file="annotations/test.json",
    data_prefix=dict(img="imgs/"),
    test_mode=True,
    pipeline=[],
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="CombinedDataset",
        metainfo=dict(from_file="configs/_base_/datasets/mouse_v5.py"),
        sample_ratio_factor=[0.0625],
        datasets=[rat_test_dataset],
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
            ann_file=data_root3 + "/annotations/test.json",
            # data_prefix=dict(img="imgs/"),
            gt_converter=rat_converter,
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
