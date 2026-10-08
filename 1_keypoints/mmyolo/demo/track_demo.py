# Copyright (c) OpenMMLab. All rights reserved.
"""Perform MMYOLO inference on a video as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth # noqa: E501, E261.

python demo/video_demo.py \
    demo/video_demo.mp4 \
    configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
    checkpoint/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
    --out demo_result.mp4
```
"""
import argparse

import os
import cv2
import mmcv
import numpy as np
from mmcv.transforms import Compose
from mmdet.apis import inference_detector, init_detector
from mmengine.utils import track_iter_progress

from mmyolo.registry import VISUALIZERS

from tracker.mc_SMILEtrack import SMILEtrack

from tqdm import tqdm

import random

from save_bbox import convert_to_serializable, write_json

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO video demo')
    parser.add_argument('--video', help='Video file', default='/mnt/nas/MutiView_3DVideo/20231117/231117_G1/videos/data1.mp4')
    parser.add_argument('--config', help='Config file', default='/home/vrlab/Documents/mmpose_mouse/work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v2/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v2.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='/home/vrlab/Documents/mmpose_mouse/work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_mouse_2mouse_v2/best_coco_bbox_mAP_epoch_49.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    
    # zhoutt: add
    parser.add_argument('--json', type=str,default = None, help='Output bbox json file')
    
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')

    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument("--track_high_thresh", type=float, default=0.7, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.5, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=300, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.85, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    
    args = parser.parse_args()
    
    args.ablation = False
    args.jde = False
    
    return args


def main():
    args = parse_args()
    
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')
        
    # zhoutt: add
    if args.json == None:
        json_path = args.video.split('/')[-1].split('.')[0] + '.json'
    else:
        json_path = args.json
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
        video_writer_track = cv2.VideoWriter(
            args.out.replace('.mp4', '-t.mp4'), fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    # zhoutt: add
    bbox = {}
    
    track_model = SMILEtrack(args)

    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    
    for frame in tqdm(video_reader):
    # for frame in track_iter_progress(video_reader):
        result = inference_detector(model, frame, test_pipeline=test_pipeline)

        t_img = frame.copy()

        #TODO trans result to pred for track
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()

        pred = np.zeros((len(bboxes), 6))
        pred[:, :4] = bboxes
        pred[:, 4] = scores
        pred[:, 5] = labels
        
        online_targets = track_model.update(pred, t_img)

        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            if tlwh[2] * tlwh[3] > args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_cls.append(t.cls)

                label = f'{tid}, {int(tcls)}'
                plot_one_box(tlbr, t_img, label=label, color=colors[int(tid) % len(colors)], line_thickness=3)

        # Save results (image with detections)
        video_writer_track.write(t_img)


        # zhoutt: add
        data = convert_to_serializable(result.to_dict())
        bbox[len(bbox) + 1] = data
        # print(len(bbox), len(bbox.keys()))
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr)
        frame = visualizer.get_image()
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)
            
    # zhoutt: add
    write_json(json_path, bbox)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
