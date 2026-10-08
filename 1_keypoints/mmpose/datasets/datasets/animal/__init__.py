# Copyright (c) OpenMMLab. All rights reserved.
from .animalpose_dataset import AnimalPoseDataset
from .ap10k_dataset import AP10KDataset
from .atrw_dataset import ATRWDataset
from .fly_dataset import FlyDataset
from .horse10_dataset import Horse10Dataset
from .locust_dataset import LocustDataset
from .macaque_dataset import MacaqueDataset
from .zebra_dataset import ZebraDataset
from .mouse_test_dataset import mouse_test_dataset

__all__ = [
    "AnimalPoseDataset",
    "AP10KDataset",
    "Horse10Dataset",
    "MacaqueDataset",
    "FlyDataset",
    "LocustDataset",
    "ZebraDataset",
    "ATRWDataset",
    "mouse_test_dataset",
    "mouse_v3_dataset",
    "mouse_v4_dataset",
    "mouse_v5_dataset",
    "mouse_v6_dataset",
    "rat_dataset",
]
