# MouseGPT changes to MMYOLO

Distribution documented on 2026-10-09. This directory is based on [OpenMMLab MMYOLO `8c4d9dc503dc8e327bec8147e8dc97124052f693`](https://github.com/open-mmlab/mmyolo/tree/8c4d9dc503dc8e327bec8147e8dc97124052f693). The following changes distinguish this distribution from that upstream revision.

This component, including MouseGPT modifications and additions, is distributed under [GPLv3](LICENSE), with existing third-party notices preserved. The root MouseGPT CC BY-NC-ND license does not apply to this component.

Customizations include animal-specific detector configurations, bounding-box export and tracking/reconstruction helpers, configurable demo paths, and documentation/comment edits. Notebook outputs and selected demo media are omitted. The upstream `mmyolo/` runtime package is unchanged.

## Modified files

- [demo/15_minutes_instance_segmentation.ipynb](demo/15_minutes_instance_segmentation.ipynb)
- [demo/15_minutes_object_detection.ipynb](demo/15_minutes_object_detection.ipynb)
- [demo/image_demo.py](demo/image_demo.py)
- [demo/video_demo.py](demo/video_demo.py)
- [docs/zh_cn/common_usage/freeze_layers.md](docs/zh_cn/common_usage/freeze_layers.md)
- [docs/zh_cn/common_usage/mim_usage.md](docs/zh_cn/common_usage/mim_usage.md)
- [docs/zh_cn/common_usage/module_combination.md](docs/zh_cn/common_usage/module_combination.md)
- [docs/zh_cn/common_usage/ms_training_testing.md](docs/zh_cn/common_usage/ms_training_testing.md)
- [docs/zh_cn/common_usage/multi_necks.md](docs/zh_cn/common_usage/multi_necks.md)
- [docs/zh_cn/common_usage/set_random_seed.md](docs/zh_cn/common_usage/set_random_seed.md)
- [docs/zh_cn/common_usage/single_multi_channel_applications.md](docs/zh_cn/common_usage/single_multi_channel_applications.md)
- [docs/zh_cn/get_started/15_minutes_instance_segmentation.md](docs/zh_cn/get_started/15_minutes_instance_segmentation.md)
- [docs/zh_cn/get_started/15_minutes_object_detection.md](docs/zh_cn/get_started/15_minutes_object_detection.md)
- [docs/zh_cn/get_started/installation.md](docs/zh_cn/get_started/installation.md)
- [docs/zh_cn/notes/code_style.md](docs/zh_cn/notes/code_style.md)
- [docs/zh_cn/notes/compatibility.md](docs/zh_cn/notes/compatibility.md)
- [docs/zh_cn/notes/conventions.md](docs/zh_cn/notes/conventions.md)
- [docs/zh_cn/recommended_topics/algorithm_descriptions/rtmdet_description.md](docs/zh_cn/recommended_topics/algorithm_descriptions/rtmdet_description.md)
- [docs/zh_cn/recommended_topics/algorithm_descriptions/yolov6_description.md](docs/zh_cn/recommended_topics/algorithm_descriptions/yolov6_description.md)
- [docs/zh_cn/recommended_topics/algorithm_descriptions/yolov8_description.md](docs/zh_cn/recommended_topics/algorithm_descriptions/yolov8_description.md)
- [docs/zh_cn/recommended_topics/application_examples/ionogram_detection.md](docs/zh_cn/recommended_topics/application_examples/ionogram_detection.md)
- [docs/zh_cn/recommended_topics/complexity_analysis.md](docs/zh_cn/recommended_topics/complexity_analysis.md)
- [docs/zh_cn/recommended_topics/contributing.md](docs/zh_cn/recommended_topics/contributing.md)
- [docs/zh_cn/recommended_topics/labeling_to_deployment_tutorials.md](docs/zh_cn/recommended_topics/labeling_to_deployment_tutorials.md)
- [docs/zh_cn/recommended_topics/replace_backbone.md](docs/zh_cn/recommended_topics/replace_backbone.md)
- [docs/zh_cn/recommended_topics/visualization.md](docs/zh_cn/recommended_topics/visualization.md)
- [docs/zh_cn/tutorials/config.md](docs/zh_cn/tutorials/config.md)
- [docs/zh_cn/tutorials/custom_installation.md](docs/zh_cn/tutorials/custom_installation.md)
- [docs/zh_cn/tutorials/rotated_detection.md](docs/zh_cn/tutorials/rotated_detection.md)
- [docs/zh_cn/useful_tools/log_analysis.md](docs/zh_cn/useful_tools/log_analysis.md)
- [docs/zh_cn/useful_tools/print_config.md](docs/zh_cn/useful_tools/print_config.md)
- [projects/easydeploy/deepstream/README_zh-CN.md](projects/easydeploy/deepstream/README_zh-CN.md)
- [projects/easydeploy/docs/model_convert.md](projects/easydeploy/docs/model_convert.md)
- [tools/analysis_tools/browse_coco_json.py](tools/analysis_tools/browse_coco_json.py)

## Added files

- [configs/anchor_optimize_result.json](configs/anchor_optimize_result.json)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse.py)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2.py)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v2.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v2.py)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v3.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v3.py)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v3_rich.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v3_rich.py)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_rat.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_rat.py)
- [configs/yolov5_s-v61_syncbn_fast_1xb32-100e_rat_pre.py](configs/yolov5_s-v61_syncbn_fast_1xb32-100e_rat_pre.py)
- [demo/batch_track.py](demo/batch_track.py)
- [demo/com_from_bbox.py](demo/com_from_bbox.py)
- [demo/save_bbox.py](demo/save_bbox.py)
- [demo/track_demo.py](demo/track_demo.py)
- [demo/tracker/SLM.py](demo/tracker/SLM.py)
- [demo/tracker/basetrack.py](demo/tracker/basetrack.py)
- [demo/tracker/bot_sort.py](demo/tracker/bot_sort.py)
- [demo/tracker/gmc.py](demo/tracker/gmc.py)
- [demo/tracker/kalman_filter.py](demo/tracker/kalman_filter.py)
- [demo/tracker/matching.py](demo/tracker/matching.py)
- [demo/tracker/matching_backup.py](demo/tracker/matching_backup.py)
- [demo/tracker/mc_SMILEtrack.py](demo/tracker/mc_SMILEtrack.py)
- [demo/tracker/mc_bot_sort.py](demo/tracker/mc_bot_sort.py)
- [demo/tracker/resnet.py](demo/tracker/resnet.py)
- [demo/tracker/tracking_utils/evaluation.py](demo/tracker/tracking_utils/evaluation.py)
- [demo/tracker/tracking_utils/io.py](demo/tracker/tracking_utils/io.py)
- [demo/tracker/tracking_utils/timer.py](demo/tracker/tracking_utils/timer.py)
- [demo/triangulate_all.py](demo/triangulate_all.py)
- [demo/triangulate_com.py](demo/triangulate_com.py)
- [td-hm_hrnet-w32_8xb64-210e_test_mixap10k_onap10k.py](td-hm_hrnet-w32_8xb64-210e_test_mixap10k_onap10k.py)

This change notice is also an addition to the distribution.

## Omitted upstream files

- `demo/demo.jpg`
- `demo/demo.mp4`
- `demo/dog.jpg`
- `demo/large_image.jpg`
