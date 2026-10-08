# 旋转目标检测

旋转目标检测（Rotated Object Detection），又称为有向目标检测（Oriented Object Detection），试图在检测出目标位置的同时得到目标的方向信息。它通过重新定义目标表示形式，以及增加回归自由度数量的操作，实现旋转矩形、四边形甚至任意形状的目标检测。旋转目标检测在人脸识别、场景文字、遥感影像、自动驾驶、医学图像、机器人抓取等领域都有广泛应用。

关于旋转目标检测的详细介绍请参考文档 [MMRotate 基础知识](https://mmrotate.readthedocs.io/zh_CN/1.x/overview.html)

MMYOLO 中的旋转目标检测依赖于 MMRotate 1.x，请参考文档 [开始你的第一步](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) 安装 MMRotate 1.x。

本教程将介绍如何在 MMYOLO 中训练和使用旋转目标检测模型，目前支持了 RTMDet-R。

## 数据集准备

对于旋转目标检测数据集，目前最常用的数据集是 DOTA 数据集，由于DOTA数据集中的图像分辨率较大，因此需要进行切片处理，数据集准备请参考 [Preparing DOTA Dataset](https://github.com/open-mmlab/mmyolo/tools/dataset_converters/dota_split).

处理后的数据集结构如下：

```none
mmyolo
├── data
│   ├── split_ss_dota
│   │   ├── trainval
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
│   ├── split_ms_dota
│   │   ├── trainval
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

其中 `split_ss_dota` 是单尺度切片，`split_ms_dota` 是多尺度切片，可以根据需要选择。

对于自定义数据集，我们建议将数据转换为 DOTA 格式并离线进行转换，如此您只需在数据转换后修改 config 的数据标注路径和类别即可。

为了方便使用，我们同样提供了基于 COCO 格式的旋转标注格式，将多边形检测框储存在 COCO 标注的 segmentation 标签中，示例如下：

```json
{
    "id": 131,
    "image_id": 72,
    "bbox": [123, 167, 11, 37],
    "area": 271.5,
    "category_id": 1,
    "segmentation": [[123, 167, 128, 204, 134, 201, 132, 167]],
    "iscrowd": 0,
}
```

## 配置文件

这里以 RTMDet-R 为例介绍旋转目标检测的配置文件，其中大部分和水平检测模型相同，主要介绍它们的差异，包括数据集和评测器配置、检测头、可视化等。

得益于 MMEngine 的配置文件系统，大部分模块都可以调用 MMRotate 中的模块。

### 数据集和评测器配置

关于配置文件的基础请先阅读 [学习 YOLOV5 配置文件](./config.md). 下面介绍旋转目标检测的一些必要设置。

```python
dataset_type = 'YOLOv5DOTADataset'
data_root = 'data/split_ss_dota/'

angle_version = 'le90'

train_pipeline = [

    dict(
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True,
         box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType',
         box_type_mapping=dict(gt_bboxes='rbox')),


    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip',
         prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmrotate.RandomRotate',
         prob=0.5,
         angle_range=180,
         rotate_type='mmrotate.Rotate',
         rect_obj_labels=[9, 11]),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RegularizeRotatedBox',
         angle_version=angle_version),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
```

RTMDet-R 保持论文内的配置，默认仅采用随机旋转增强，得益于 BoxType 设计，在数据增强阶段，大部分增强无需改动代码即可直接支持，例如 MixUp 和 Mosaic 等，可以直接在 pipeline 中使用。

```{Warning}
目前已知 Albu 数据增强仅支持水平框，在使用其他的数据增强时建议先使用 可视化数据集脚本 `browse_dataset.py` 验证数据增强是否正确。
```

RTMDet-R 测试阶段仅采用 Resize 和 Pad，在验证和评测时，都采用相同的数据流进行推理。

```python
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),

    dict(
        type='LoadAnnotations',
        with_bbox=True,
        box_type='qbox',
        _scope_='mmdet'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=val_pipeline))
```

[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html) 用于计算训练模型在验证和测试数据集上的指标。评测器的配置由一个或一组评价指标（Metric）配置组成：

```python
val_evaluator = dict(
    type='mmrotate.DOTAMetric',
    metric='mAP'
)
test_evaluator = val_evaluator
```

由于 DOTA 测试数据集没有标注文件， 如果要保存在测试数据集上的检测结果，则可以像这样编写配置：

```python


test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='mmrotate.DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/dota_detection/submission')
```

如果使用基于 COCO 格式的旋转框标注，只需要修改 pipeline 中数据读取流程和训练数据集的配置，以训练数据为例：

```python

dataset_type='YOLOv5CocoDataset'

train_pipeline = [

    dict(
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         poly2mask=False)
    dict(type='ConvertMask2BoxType',
         box_type='rbox'),

    ...
]

metainfo = dict(
    classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field',
             'roundabout', 'harbor', 'swimming-pool', 'helicopter'))

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline),
)
```

### 模型配置

对于旋转目标检测器，在模型配置中 backbone 和 neck 的配置和其他模型是一致的，主要差异在检测头上。目前仅支持 RTMDet-R 旋转目标检测，下面介绍新增的参数：

1. `angle_version` 角度范围，用于在训练时限制角度的范围，可选的角度范围有 `le90`, `le135` 和 `oc`。

2. `angle_coder` 角度编码器，和 bbox coder 类似，用于编码和解码角度。

   默认使用的角度编码器是 `PseudoAngleCoder`，即”伪角度编码器“，并不进行编解码，直接回归角度参数。这样设计的目标是能更好的自定义角度编码方式，而无需重写代码，例如 CSL，DCL，PSC 等方法。

3. `use_hbbox_loss` 是否使用水平框 loss。考虑到部分角度编码解码过程不可导，直接使用旋转框的损失函数无法学习角度，因此引入该参数用于将框和角度分开训练。

4. `loss_angle` 角度损失函数。在设定`use_hbbox_loss=True` 时必须设定，而使用旋转框损失时可选，此时可以作为回归损失的辅助。

通过组合 `use_hbbox_loss` 和 `loss_angle` 可以控制旋转框训练时的回归损失计算方式，共有三种组合方式：

- `use_hbbox_loss=False` 且 `loss_angle` 为 None.

  此时框预测和角度预测进行合并，直接对旋转框预测进行回归，此时 `loss_bbox` 应当设定为旋转框损失，例如 `RotatedIoULoss`。
  这种方案和水平检测模型的回归方式基本一致，只是多了额外的角度编解码过程。

  ```
  bbox_pred────(tblr)───┐
                        ▼
  angle_pred          decode──►rbox_pred──(xywha)─►loss_bbox
      │                 ▲
      └────►decode──(a)─┘
  ```

- `use_hbbox_loss=False`，同时设定 `loss_angle`.

  此时会增加额外的角度回归和分类损失，具体的角度损失类型需要根据角度编码器 `angle_code` 进行选择。

  ```
  bbox_pred────(tblr)───┐
                        ▼
  angle_pred          decode──►rbox_pred──(xywha)─►loss_bbox
      │                 ▲
      ├────►decode──(a)─┘
      │
      └───────────────────────────────────────────►loss_angle
  ```

- `use_hbbox_loss=True` 且 `loss_angle` 为 None.

  此时框预测和角度预测完全分离，将两个分支视作两个任务进行训练。
  此时 `loss_bbox` 要设定为水平框的损失函数，例如 `IoULoss` 。

  ```
  bbox_pred──(tblr)──►decode──►hbox_pred──(xyxy)──►loss_bbox

  angle_pred──────────────────────────────────────►loss_angle
  ```

除了检测头中的参数，在test_cfg中还增加了 `decoded_with_angle` 参数用来控制推理时角度的处理逻辑，默认设定为 True 。
设计这个参数的目标是让训练过程和推理过程的逻辑对齐，该参数会影响最终的精度。

当 `decoded_with_angle=True` 时，将框和角度同时送入 `bbox_coder` 中。
此时要使用旋转框的编解码器，例如`DistanceAnglePointCoder`。

```
bbox_pred────(tblr)───┐
                      ▼
angle_pred          decode──(xywha)──►rbox_pred
    │                 ▲
    └────►decode──(a)─┘
```

当 `decoded_with_angle=False` 时，首先解码出水平检测框，之后将角度 concat 到检测框。
此时要使用水平框的编解码器，例如`DistancePointBBoxCoder`。

```
bbox_pred──(tblr)─►decode
                      │ (xyxy)
                      ▼
                    format───(xywh)──►concat──(xywha)──►rbox_pred
                                       ▲
angle_pred────────►decode────(a)───────┘
```

### 可视化器

由于旋转框和水平框的差异，旋转目标检测模型需要使用 MMRotate 中的 `RotLocalVisualizer`，配置如下：

```python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmrotate.RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

## 实用工具

目前测试可用的工具包括：

[可视化数据集](../useful_tools/browse_dataset.md)
