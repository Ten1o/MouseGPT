# Copyright (c) OpenMMLab. All rights reserved.
from .mmpose_inferencer import MMPoseInferencer
from .bmmpose_inferencer import BMMPoseInferencer
from .rmmpose_inferencer import RMMPoseInferencer
from .pose2d_inferencer import Pose2DInferencer
from .bpose2d_inferencer import BPose2DInferencer
from .rpose2d_inferencer import RPose2DInferencer
from .utils import get_model_aliases

__all__ = ['Pose2DInferencer', 'BPose2DInferencer', 'RPose2DInferencer', 'MMPoseInferencer', 'BMMPoseInferencer', 'RMMPoseInferencer', 'get_model_aliases']
