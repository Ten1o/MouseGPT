# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmpose.datasets import build_dataset
from mmpose.apis import init_model


cfg_path = r'configs\animal_2d_keypoint\topdown_heatmap\mouse_test\td-hm_hrnet-w48_8xb64-210e_mouse_test-256x256.py'
cfg = Config.fromfile(cfg_path)
if 'preprocess_cfg' in cfg:
    cfg.model.setdefault('data_preprocessor',
                        cfg.get('preprocess_cfg', {}))

cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
pose_estimator = init_model(
    cfg_path,
    r'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
    device='cuda:0',
    cfg_options=cfg_options
)

