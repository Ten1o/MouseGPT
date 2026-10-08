# 配置文件

MMPose 使用 Python 文件作为配置文件，将模块化设计和继承设计结合到配置系统中，便于进行各种实验。

## 简介

MMPose 拥有一套强大的配置系统，在注册器的配合下，用户可以通过一个配置文件来定义整个项目需要用到的所有内容，以 Python 字典形式组织配置信息，传递给注册器完成对应模块的实例化。

下面是一个常见的 Pytorch 模块定义的例子：

```Python

Class Loss_A(nn.Module):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    def forward(self, x):
        return x


loss = Loss_A(param1=1.0, param2=True)
```

只需要通过一行代码对这个类进行注册：

```Python

from mmpose.registry import MODELS

@MODELS.register_module()
Class Loss_A(nn.Module):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    def forward(self, x):
        return x
```

并在对应目录下的 `__init__.py` 中进行 `import`：

```Python
# __init__.py of mmpose/models/losses
from .loss_a.py import Loss_A

__all__ = ['Loss_A']
```

我们就可以通过如下方式来从配置文件定义并进行实例化：

```Python

loss_cfg = dict(
    type='Loss_A',
    param1=1.0,
    param2=True
)


loss = MODELS.build(loss_cfg)
```

MMPose 预定义的 Registry 在 `$MMPOSE/mmpose/registry.py` 中，目前支持的有：

- `DATASETS`：数据集

- `TRANSFORMS`：数据变换

- `MODELS`：模型模块（Backbone、Neck、Head、Loss等）

- `VISUALIZERS`：可视化工具

- `VISBACKENDS`：可视化后端

- `METRICS`：评测指标

- `KEYPOINT_CODECS`：编解码器

- `HOOKS`：钩子类

```{note}
需要注意的是，所有新增的模块都需要使用注册器（Registry）进行注册，并在对应目录的 `__init__.py` 中进行 `import`，以便能够使用配置文件构建其实例。
```

## 配置系统

具体而言，一个配置文件主要包含如下五个部分：

- 通用配置：与训练或测试无关的通用配置，如时间统计，模型存储与加载，可视化等相关 Hook，以及一些分布式相关的环境配置

- 数据配置：数据增强策略，Dataset和Dataloader相关配置

- 训练配置：断点恢复、模型权重加载、优化器、学习率调整、训练轮数和测试间隔等

- 模型配置：模型模块、参数、损失函数等

- 评测配置：模型性能评测指标

你可以在 `$MMPOSE/configs` 下找到我们提供的配置文件，配置文件之间通过继承来避免冗余。为了保持配置文件简洁易读，我们将一些必要但不常改动的配置存放到了 `$MMPOSE/configs/_base_` 目录下，如果希望查阅完整的配置信息，你可以运行如下指令：

```Bash
python tools/analysis/print_config.py /PATH/TO/CONFIG
```

### 通用配置

通用配置指与训练或测试无关的必要配置，主要包括：

- **默认Hook**：迭代时间统计，训练日志，参数更新，checkpoint 等

- **环境配置**：分布式后端，cudnn，多进程配置等

- **可视化器**：可视化后端和策略设置

- **日志配置**：日志等级，格式，打印和记录间隔等

下面是通用配置的样例说明：

```Python

default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='coco/AP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
```

通用配置一般单独存放到`$MMPOSE/configs/_base_`目录下，通过如下方式进行继承：

```Python
_base_ = ['../../../_base_/default_runtime.py']
```

```{note}
CheckpointHook:

- save_best: `'coco/AP'` 用于 `CocoMetric`, `'PCK'` 用于 `PCKAccuracy`
- max_keep_ckpts: 最大保留ckpt数量，默认为-1，代表不限制

样例:

`default_hooks = dict(checkpoint=dict(save_best='PCK', rule='greater', max_keep_ckpts=1))`
```

### 数据配置

数据配置指数据处理相关的配置，主要包括：

- **数据后端**：数据供给后端设置，默认为本地硬盘，我们也支持从 LMDB，S3 Bucket 等加载

- **数据集**：图像与标注文件路径

- **加载**：加载策略，批量大小等

- **流水线**：数据增强策略

- **编码器**：根据标注生成特定格式的监督信息

下面是数据配置的样例说明：

```Python
backend_args = dict(backend='local')
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args,
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='GenerateTarget',

        encoder=codec,
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type ,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type ,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader
```

```{note}

常用功能可以参考以下教程:
- [恢复训练](../common_usages/resume_training.md)
- [自动混合精度训练](../common_usages/amp_training.md)
- [设置随机种子](../common_usages/set_random_seed.md)

```

### 训练配置

训练配置指训练策略相关的配置，主要包括：

- 从断点恢复训练

- 模型权重加载

- 训练轮数和测试间隔

- 学习率调整策略，如 warmup，scheduler

- 优化器和学习率

- 高级训练策略设置，如自动学习率缩放

下面是训练配置的样例说明：

```Python
resume = False
load_from = None
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict( # scheduler
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
auto_scale_lr = dict(base_batch_size=512)
```

### 模型配置

模型配置指模型训练和推理相关的配置，主要包括：

- 模型结构

- 损失函数

- 数据解码策略

- 测试时增强策略

下面是模型配置的样例说明，定义了一个基于 HRNetw32 的 Top-down Heatmap-based 模型：

```Python

codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose'
            '/pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))
```

### 评测配置

评测配置指公开数据集中关键点检测任务常用的评测指标，主要包括：

- AR, AP and mAP

- PCK, PCKh, tPCK

- AUC

- EPE

- NME

下面是评测配置的样例说明，定义了一个COCO指标评测器：

```Python
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
```

## 配置文件命名规则

MMPose 配置文件命名风格如下：

```Python
{{算法信息}}_{{模块信息}}_{{训练信息}}_{{数据信息}}.py
```

文件名总体分为四部分：算法信息，模块信息，训练信息和数据信息。不同部分的单词之间用下划线 `'_'` 连接，同一部分有多个单词用短横线 `'-'` 连接。

- **算法信息**：算法名称，如 `topdown-heatmap`，`topdown-rle` 等

- **模块信息**：按照数据流的顺序列举一些中间的模块，其内容依赖于算法任务，如 `res101`，`hrnet-w48`等

- **训练信息**：训练策略的一些设置，包括 `batch size`，`schedule` 等，如 `8xb64-210e`

- **数据信息**：数据集名称、模态、输入尺寸等，如 `ap10k-256x256`，`zebra-160x160` 等

有时为了避免文件名过长，会省略模型信息中一些强相关的模块，只保留关键信息，如RLE-based算法中的`GAP`，Heatmap-based算法中的 `deconv` 等。

如果你希望向MMPose添加新的方法，你的配置文件同样需要遵守该命名规则。

## 常见用法

### 配置文件的继承

该用法常用于隐藏一些必要但不需要修改的配置，以提高配置文件的可读性。假如有如下两个配置文件：

`optimizer_cfg.py`:

```Python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

`resnet50.py`:

```Python
_base_ = ['optimizer_cfg.py']
model = dict(type='ResNet', depth=50)
```

虽然我们在 `resnet50.py` 中没有定义 optimizer 字段，但由于我们写了 `_base_ = ['optimizer_cfg.py']`，会使这个配置文件获得 `optimizer_cfg.py` 中的所有字段：

```Python
cfg = Config.fromfile('resnet50.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

### 继承字段的修改

对于继承过来的已经定义好的字典，可以直接指定对应字段进行修改，而不需要重新定义完整的字典：

`resnet50_lr0.01.py`:

```Python
_base_ = ['optimizer_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(lr=0.01)
```

这个配置文件只修改了对应字段`lr`的信息：

```Python
cfg = Config.fromfile('resnet50_lr0.01.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

### 删除字典中的字段

如果不仅是需要修改某些字段，还需要删除已定义的一些字段，需要在重新定义这个字典时指定`_delete_=True`，表示将没有在新定义中出现的字段全部删除：

`resnet50.py`:

```Python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(_delete_=True, type='SGD', lr=0.01)
```

此时字典中除了 `type` 和 `lr` 以外的内容（`momentum`和`weight_decay`）将被全部删除：

```Python
cfg = Config.fromfile('resnet50_lr0.01.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.01)
```

```{note}
如果你希望更深入地了解配置系统的高级用法，可以查看 [MMEngine 教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html)。
```
