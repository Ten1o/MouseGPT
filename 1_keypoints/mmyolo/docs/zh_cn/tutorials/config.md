# 学习 YOLOv5 配置文件

MMYOLO 和其他 OpenMMLab 仓库使用 [MMEngine 的配置文件系统](https://mmengine.readthedocs.io/zh_cn/latest/tutorials/config.md)。 配置文件使用了模块化和继承设计，以便于进行各类实验。

## 配置文件的内容

MMYOLO 采用模块化设计，所有功能的模块都可以通过配置文件进行配置。 以 [yolov5_s-v61_syncbn_8xb16-300e_coco.py](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py) 为例，我们将根据不同的功能模块介绍配置文件中的各个字段：

### 重要参数

如下参数是修改训练配置时经常需要修改的参数。例如缩放因子 `deepen_factor` 和 `widen_factor`，MMYOLO 中的网络基本都使用它们来控制模型的大小。所以我们推荐在配置文件中单独定义这些参数。

```python
img_scale = (640, 640)
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 300
save_epoch_intervals = 10
train_batch_size_per_gpu = 16
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2
```

### 模型配置

在 MMYOLO 的配置中，我们使用 `model` 字段来配置检测算法的组件。 除了 `backbone`、`neck` 等神经网络组件外，还需要 `data_preprocessor`、`train_cfg` 和 `test_cfg`。 `data_preprocessor` 负责对 dataloader 输出的每一批数据进行预处理。 模型配置中的 `train_cfg` 和 `test_cfg` 用于设置训练和测试组件的超参数。

```python
anchors = [[(10, 13), (16, 30), (33, 23)],
           [(30, 61), (62, 45), (59, 119)],
           [(116, 90), (156, 198), (373, 326)]]
strides = [8, 16, 32]

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=80,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms',
                 iou_threshold=0.65),
        max_per_img=300))
```

### 数据集和评测器配置

在使用 [执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html) 进行训练、测试、验证时，我们需要配置 [Dataloader](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader) 。构建数据 dataloader 需要设置数据集（dataset）和数据处理流程（data pipeline）。 由于这部分的配置较为复杂，我们使用中间变量来简化 dataloader 配置的编写。由于 MMYOLO 中各类轻量目标检测算法使用了更加复杂的数据增强方法，因此会比 MMDetection 中的其他模型拥有更多样的数据集配置。

YOLOv5 的训练与测试的数据流存在一定差异，这里我们分别进行介绍。

```python
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

pre_transform = [
    dict(
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True)
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))
```

YOLOv5 测试阶段采用 [Letter Resize](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/transforms/transforms.py#L116) 的方法来将所有的测试图像统一到相同尺度，进而有效保留了图像的长宽比。因此我们在验证和评测时，都采用相同的数据流进行推理。

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize',
         scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        ann_file='annotations/instances_val2017.json',
        pipeline=test_pipeline,
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=val_batch_size_per_gpu,
            img_size=img_scale[0],
            size_divisor=32,
            extra_pad_ratio=0.5)))

test_dataloader = val_dataloader
```

[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html) 用于计算训练模型在验证和测试数据集上的指标。评测器的配置由一个或一组评价指标（Metric）配置组成：

```python
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
)
test_evaluator = val_evaluator
```

由于测试数据集没有标注文件，因此 MMYOLO 中的 `test_dataloader` 和 `test_evaluator` 配置通常等于 `val`。 如果要保存在测试数据集上的检测结果，则可以像这样编写配置：

```python


test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='./work_dirs/coco_detection/test')
```

### 训练和测试的配置

MMEngine 的 Runner 使用 Loop 来控制训练，验证和测试过程。
用户可以使用这些字段设置最大训练轮次和验证间隔。

```python
max_epochs = 300
save_epoch_intervals = 10

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

MMEngine 也支持动态评估间隔，例如你可以在前面 280 epoch 训练阶段中，每间隔 10 个 epoch 验证一次，到最后 20 epoch 训练中每隔 1 个 epoch 验证一次，则配置写法为：

```python
max_epochs = 300
save_epoch_intervals = 10

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(280, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

### 优化相关配置

`optim_wrapper` 是配置优化相关设置的字段。优化器封装（OptimWrapper）不仅提供了优化器的功能，还支持梯度裁剪、混合精度训练等功能。更多内容请看[优化器封装教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html).

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    clip_grad=None,
    constructor='YOLOv5OptimizerConstructor')

```

`param_scheduler` 字段用于配置参数调度器（Parameter Scheduler）来调整优化器的超参数（例如学习率和动量）。 用户可以组合多个调度器来创建所需的参数调整策略。 在[参数调度器教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html) 和参数调度器 API 文档 中查找更多信息。在 YOLOv5 中，参数调度实现比较复杂，难以通过  `param_scheduler` 实现。所以我们采用了 `YOLOv5ParamSchedulerHook` 来实现（见下节），这样做更简单但是通用性较差。

```python
param_scheduler = None
```

### 钩子配置

用户可以在训练、验证和测试循环上添加钩子，以便在运行期间插入一些操作。配置中有两种不同的钩子字段，一种是 `default_hooks`，另一种是 `custom_hooks`。

`default_hooks` 是一个字典，用于配置运行时必须使用的钩子。这些钩子具有默认优先级，如果未设置，runner 将使用默认值。如果要禁用默认钩子，用户可以将其配置设置为 `None`。

```python
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=3))
```

`custom_hooks` 是一个列表。用户可以在这个字段中加入自定义的钩子，例如 `EMAHook`。

```python
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
```

### 运行相关配置

```python
default_scope = 'mmyolo'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
```

## 配置文件继承

在 `config/_base_` 文件夹目前有运行时的默认设置(default runtime)。由 `_base_` 下的组件组成的配置，被我们称为 _原始配置(primitive)_。

对于同一文件夹下的所有配置，推荐**只有一个**对应的**原始配置**文件。所有其他的配置文件都应该继承自这个**原始配置**文件。这样就能保证配置文件的最大继承深度为 3。

为了便于理解，我们建议贡献者继承现有方法。例如，如果在 YOLOv5s 的基础上做了一些修改，比如修改网络深度，用户首先可以通过指定 `_base_ = ./yolov5_s-v61_syncbn_8xb16-300e_coco.py` 来集成基础的 YOLOv5 结构，然后修改配置文件中的必要参数以完成继承。

如果你在构建一个与任何现有方法不共享结构的全新方法，那么可以在 `configs` 文件夹下创建一个新的例如 `yolov100` 文件夹。

更多细节请参考 [MMEngine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html)。

通过设置 `_base_` 字段，我们可以设置当前配置文件继承自哪些文件。

当 `_base_` 为文件路径字符串时，表示继承一个配置文件的内容。

```python
_base_ = '../_base_/default_runtime.py'
```

当 `_base_` 是多个文件路径的列表时，表示继承多个文件。

```python
_base_ = [
    './yolov5_s-v61_syncbn_8xb16-300e_coco.py',
    '../_base_/default_runtime.py'
]
```

如果需要检查配置文件，可以通过运行 `mim run mmdet print_config /PATH/TO/CONFIG` 来查看完整的配置。

### 忽略基础配置文件里的部分内容

有时，您也许会设置 `_delete_=True` 去忽略基础配置文件里的一些域内容。 您也许可以参照 [MMEngine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html) 来获得一些简单的指导。

在 MMYOLO 里，例如为了改变 RTMDet 的主干网络的某些内容：

```python
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(...),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(...),
    bbox_head=dict(...))
```

如果想把 RTMDet 主干网络的 `CSPNeXt` 改成 `YOLOv6EfficientRep`，因为 `CSPNeXt` 和 `YOLOv6EfficientRep` 中有不同的字段(`channel_attention` 和 `expand_ratio`)，这时候就需要使用 `_delete_=True` 将新的键去替换 `backbone` 域内所有老的键。

```python
_base_ = '../rtmdet/rtmdet_l_syncbn_8xb32-300e_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(...),
    bbox_head=dict(...))
```

### 使用配置文件里的中间变量

配置文件里会使用一些中间变量，例如数据集里的 `train_pipeline`/`test_pipeline`。我们在定义新的 `train_pipeline`/`test_pipeline` 之后，需要将它们传递到 `data` 里。例如，我们想在训练或测试时，改变 YOLOv5 网络的 `img_scale` 训练尺度并在训练时添加 `YOLOv5MixUp` 数据增强，`img_scale/train_pipeline/test_pipeline` 是我们想要修改的中间变量。

**注**：使用 `YOLOv5MixUp` 数据增强时，需要将 `YOLOv5MixUp` 之前的训练数据处理流程定义在其 `pre_transform`  中。详细过程和图解可参见 [YOLOv5 原理和实现全解析](../recommended_topics/algorithm_descriptions/yolov5_description.md)。

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

img_scale = (1280, 1280)
affine_scale = 0.9

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=0.1,
        pre_transform=[*pre_transform,*mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

我们首先定义新的 `train_pipeline`/`test_pipeline` 然后传递到 `data` 里。

同样的，如果我们想从 `SyncBN` 切换到 `BN` 或者 `MMSyncBN`，我们需要修改配置文件里的每一个  `norm_cfg`。

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    ...)
```

### 复用 \_base\_ 文件中的变量

如果用户希望在当前配置中复用 `_base_` 文件中的变量，则可以通过使用 `{{_base_.xxx}}` 的方式来获取对应变量的拷贝。而在新版 MMEngine 中，还支持省略 `{{}}` 的写法。例如：

```python
_base_ = '../_base_/default_runtime.py'

pre_transform = _base_.pre_transform
```

## 通过脚本参数修改配置

当运行 `tools/train.py` 和 `tools/test.py` 时，可以通过 `--cfg-options` 来修改配置文件。

- 更新字典链中的配置

  可以按照原始配置文件中的 dict 键顺序地指定配置预选项。例如，使用 `--cfg-options model.backbone.norm_eval=False` 将模型主干网络中的所有 BN 模块都改为 `train` 模式。

- 更新配置列表中的键

  在配置文件里，一些字典型的配置被包含在列表中。例如，数据训练流程 `data.train.pipeline` 通常是一个列表，比如 `[dict(type='LoadImageFromFile'), ...]`。如果需要将 `'LoadImageFromFile'` 改成 `'LoadImageFromNDArray'`，需要写成下述形式：`--cfg-options data.train.pipeline.0.type=LoadImageFromNDArray`.

- 更新列表或元组的值

  如果要更新的值是列表或元组。例如，配置文件通常设置 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`。如果需要改变这个键，可以通过 `--cfg-options model.data_preprocessor.mean="[127,127,127]"` 来重新设置。需要注意，引号 `"` 是支持列表或元组数据类型所必需的，并且在指定值的引号内**不允许**有空格。

## 配置文件名称风格

我们遵循以下样式来命名配置文件。建议贡献者遵循相同的风格。

```
{algorithm name}_{model component names [component1]_[component2]_[...]}-[version id]_[norm setting]_[data preprocessor type]_{training settings}_{training dataset information}_[testing dataset information].py
```

文件名分为 8 个部分，其中 4 个必填部分、4 个可选部分。 每个部分用 `_` 连接，每个部分内的单词应该用 `-` 连接。`{}` 表示必填部分，`[]` 表示选填部分。

- `{algorithm name}`：算法的名称。 它可以是检测器名称，例如 `yolov5`, `yolov6`, `yolox` 等。
- `{component names}`：算法中使用的组件名称，如 backbone、neck 等。例如 yolov5_s代表其深度缩放因子`deepen_factor=0.33` 以及其宽度缩放因子 `widen_factor=0.5`。
- `[version_id]` (可选)：由于 YOLO 系列算法迭代速度远快于传统目标检测算法，因此采用 `version id` 来区分不同子版本之间的差异。例如 YOLOv5 的 3.0 版本采用 `Focus` 层作为第一个下采样层，而 6.0 以后的版本采用 `Conv` 层作为第一个下采样层。
- `[norm_setting]` (可选)：`bn` 表示 `Batch Normalization`， `syncbn` 表示 `Synchronized Batch Normalization`。
- `[data preprocessor type]` (可选)：`fast` 表示调用 [YOLOv5DetDataPreprocessor](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/models/data_preprocessors/data_preprocessor.py#L9) 并配合 [yolov5_collate](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/utils.py#L12) 进行数据预处理，训练速度比默认的 `mmdet.DetDataPreprocessor` 更快，但是对多任务处理的灵活性较低。
- `{training settings}`：训练设置的信息，例如 batch 大小、数据增强、损失、参数调度方式和训练最大轮次/迭代。 例如：`8xb16-300e_coco` 表示使用 8 个 GPU 每个 GPU 16 张图，并训练 300 个 epoch。
  缩写介绍:
  - `{gpu x batch_per_gpu}`：GPU 数和每个 GPU 的样本数。例如 `4x4b` 是 4 个 GPU 每个 GPU 4 张图的缩写。
  - `{schedule}`：训练方案，MMYOLO 中默认为 300 个 epoch。
- `{training dataset information}`：训练数据集，例如 `coco`, `cityscapes`, `voc-0712`, `wider-face`, `balloon`。
- `[testing dataset information]` (可选)：测试数据集，用于训练和测试在不同数据集上的模型配置。 如果没有注明，则表示训练和测试的数据集类型相同。
