#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D关键点投影到2D程序
将triangulated_keypoints_3d.npy投影到各个相机视角
"""

import numpy as np
import json
import scipy.io as sio
import cv2
import os
from pathlib import Path
import argparse
import subprocess
import re
from tqdm import tqdm

class KeypointProjector:
    def __init__(self, calib_folder):
        """
        初始化投影器

        Args:
            calib_folder: 相机标定文件夹路径，包含Camera1_params.mat到Camera8_params.mat
        """
        self.calib_folder = Path(calib_folder)
        self.cameras = {}
        self.load_camera_parameters()

    def load_camera_parameters(self):
        """加载相机参数"""
        print(f"正在加载相机参数从: {self.calib_folder}")

        # 查找所有Camera*_params.mat文件
        mat_files = sorted(self.calib_folder.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"在 {self.calib_folder} 中未找到Camera*_params.mat文件")

        print(f"找到 {len(mat_files)} 个相机标定文件")

        for mat_file in mat_files:
            try:
                # 从文件名提取相机ID
                camera_name = mat_file.stem.replace('_params', '')
                match=re.search(r'(?:Camera|data)([1-8])(?:_|$)',mat_file.stem,re.I)
                camera_id=int(match.group(1)) if match else mat_files.index(mat_file)+1
                if camera_id in self.cameras:raise ValueError('Duplicate camera ID in calibration filenames.')

                # 加载.mat文件
                mat_data = sio.loadmat(str(mat_file))

                # 提取相机参数
                self.cameras[camera_id] = {
                    'K': mat_data['K'].T,  # 内参矩阵（转置以匹配OpenCV格式）
                    'R': mat_data['r'].T,  # 旋转矩阵
                    't': mat_data['t'].T,  # 平移向量
                    'RDistort': mat_data.get('RDistort',np.zeros(3)).flatten(),  # 径向畸变
                    'TDistort': mat_data.get('TDistort',np.zeros(2)).flatten(),  # 切向畸变
                }

                print(f"OK 加载相机 {camera_name}: K={self.cameras[camera_id]['K'].shape}")

            except Exception as e:
                print(f"ERROR 加载相机 {mat_file.name} 失败: {e}")
                raise

        if not self.cameras:
            raise RuntimeError("没有成功加载任何相机参数")

        print(f"成功加载 {len(self.cameras)} 个相机参数")

    def project_3d_to_2d(self, points_3d, camera_id, apply_distortion=True):
        """
        将3D点投影到2D

        Args:
            points_3d: 形状为 (N, 3) 的3D点
            camera_id: 相机ID (1-8)
            apply_distortion: 是否应用畸变校正

        Returns:
            points_2d: 形状为 (N, 2) 的2D点
        """
        if camera_id not in self.cameras:
            raise ValueError(f"相机 {camera_id} 未找到")

        camera = self.cameras[camera_id]
        K = camera['K']
        R = camera['R']
        t = camera['t']

        # 将3D点转换为齐次坐标
        if points_3d.shape[1] == 3:
            points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        else:
            points_3d_h = points_3d

        # 构建投影矩阵 P = K * [R|t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt

        # 投影到2D
        points_2d_h = (P @ points_3d_h.T).T

        # 转换为非齐次坐标
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]

        # 应用畸变（如果需要）
        if apply_distortion:
            points_2d = self.apply_distortion(points_2d, camera)

        return points_2d

    def apply_distortion(self, points_2d, camera):
        """应用畸变"""
        K = camera['K']
        RDistort = camera['RDistort']
        TDistort = camera['TDistort']

        radial=np.pad(RDistort,(0,max(0,3-len(RDistort))))
        tangent=np.pad(TDistort,(0,max(0,2-len(TDistort))))
        coefficients=np.array([radial[0],radial[1],tangent[0],tangent[1],radial[2]],dtype=float)
        # Raw videos require forward distortion, not undistortPoints.
        pixels=np.asarray(points_2d,dtype=float)
        homogeneous=np.c_[pixels,np.ones(len(pixels))]
        rays=(np.linalg.inv(K)@homogeneous.T).T
        projected,_=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),K,coefficients)
        return projected.reshape(-1,2)

    def process_keypoints_file(self, keypoints_3d_path, output_folder, reference_json_folder=None):
        """
        处理3D关键点文件并生成各相机视角的2D投影

        Args:
            keypoints_3d_path: triangulated_keypoints_3d.npy文件路径
            output_folder: 输出文件夹
            reference_json_folder: 参考JSON文件夹（用于确定instance数量）
        """
        print(f"加载3D关键点文件: {keypoints_3d_path}")
        keypoints_3d = np.load(keypoints_3d_path)
        print(f"3D关键点形状: {keypoints_3d.shape}")

        num_frames, num_keypoints, _ = keypoints_3d.shape

        # 如果提供了参考JSON文件夹，读取instance数量
        instances_per_frame = 1  # 默认每帧1个instance
        if reference_json_folder:
            reference_json_folder = Path(reference_json_folder)
            ref_files = list(reference_json_folder.glob("*.json"))
            if ref_files:
                print(f"从参考JSON文件分析instance数量: {ref_files[0]}")
                with open(ref_files[0], 'r') as f:
                    ref_data = json.load(f)
                if len(ref_data) > 0:
                    instances_per_frame = len(ref_data[0].get('instances', [1]))
                    print(f"检测到每帧 {instances_per_frame} 个instance")

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # 为每个相机生成投影
        for camera_id in sorted(self.cameras.keys()):
            print(f"\n正在处理相机 {camera_id}...")

            camera_data = []

            for frame_idx in tqdm(range(num_frames), desc=f"Camera{camera_id}"):
                # 获取当前帧的3D关键点
                points_3d = keypoints_3d[frame_idx]  # (24, 3)

                # 投影到2D
                points_2d = self.project_3d_to_2d(points_3d, camera_id)

                # 构建帧数据
                frame_data = {
                    "frame_id": frame_idx,
                    "instances": []
                }

                # 根据instance数量复制关键点
                for instance_idx in range(instances_per_frame):
                    instance_data = {
                        "keypoints": points_2d.tolist(),
                        "keypoint_scores": [1.0] * num_keypoints,  # 默认高置信度
                        "bbox": self.calculate_bbox(points_2d),
                        "bbox_score": 0.99
                    }
                    frame_data["instances"].append(instance_data)

                camera_data.append(frame_data)

            # 保存JSON文件
            output_file = output_folder / f"data{camera_id}.json"
            with open(output_file, 'w') as f:
                json.dump(camera_data, f, indent=2)

            print(f"OK 保存相机 {camera_id} 的投影结果到: {output_file}")

    def calculate_bbox(self, points_2d):
        """根据关键点计算边界框"""
        x_coords = points_2d[:, 0]
        y_coords = points_2d[:, 1]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # 添加一些边距
        margin = 20
        return [
            max(0, x_min - margin),
            max(0, y_min - margin),
            x_max + margin,
            y_max + margin
        ]

UNITS={'m':1.,'cm':.01,'mm':.001}


def render_reprojection(keypoints, calibration, videos, destination, input_unit, calibration_unit):
    """Reproject all source frames and export eight browser-playable overlays."""
    from video_utils import ffmpeg_executable
    from batch_vis import bones
    if input_unit not in UNITS or calibration_unit not in UNITS:raise ValueError('Choose input and calibration translation units.')
    data=np.load(keypoints,allow_pickle=False)
    if data.ndim!=3 or data.shape[1:]!=(24,3) or not len(data) or not np.isfinite(data).all():raise ValueError('Expected finite N x 24 x 3 keypoints. Use the filtered NPY.')
    points=data.astype(float)*(UNITS[input_unit]/UNITS[calibration_unit])
    projector=KeypointProjector(calibration)
    if set(projector.cameras)!=set(range(1,9)):raise ValueError('Exactly eight camera calibrations are required, mapped to data1...data8.')
    sources=[Path(videos)/f'data{i}.mp4' for i in range(1,9)]
    metadata=[]
    for source in sources:
        cap=cv2.VideoCapture(str(source))
        try:
            if not cap.isOpened():raise ValueError(f'Cannot open video: {source}')
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if count!=len(points):raise ValueError(f'{source.name}: {count} video frames versus {len(points)} NPY frames; frame 0 must correspond to frame 0.')
            if not np.isfinite(fps) or fps<=0 or width<=0 or height<=0:raise ValueError(f'Invalid video metadata: {source}')
            metadata.append((fps,width,height))
        finally:cap.release()
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    targets=[dest/f'data{i}.{suffix}' for i in range(1,9) for suffix in ('json','mp4')]
    created=[];files=[];views=[]
    try:
        for target in targets:
            with target.open('xb'):pass
            created.append(target)
        for camera,source in enumerate(sources,1):
            fps,width,height=metadata[camera-1]
            print(f'Reprojecting camera {camera}/8: {len(points)} frames, {fps:g} FPS',flush=True)
            cap=cv2.VideoCapture(str(source));video=dest/f'data{camera}.mp4'
            command=[ffmpeg_executable(),'-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            frames=[]
            try:
                for index,frame_points in enumerate(points):
                    ok,frame=cap.read()
                    if not ok:raise ValueError(f'{source.name}: cannot decode frame {index}.')
                    projected=projector.project_3d_to_2d(frame_points,camera)
                    if not np.isfinite(projected).all():raise ValueError(f'Invalid projection in camera {camera}, frame {index}. Check calibration and units.')
                    frames.append({'frame_id':index,'instances':[{'keypoints':projected.tolist(),'keypoint_scores':[1.]*24,'bbox':projector.calculate_bbox(projected),'bbox_score':.99}]})
                    visible=np.isfinite(projected).all(axis=1)&(projected[:,0]>=0)&(projected[:,0]<width)&(projected[:,1]>=0)&(projected[:,1]<height)
                    for a,b in bones:
                        if visible[a-1] and visible[b-1]:cv2.line(frame,tuple(projected[a-1].astype(int)),tuple(projected[b-1].astype(int)),(255,0,0),2)
                    for xy in projected[visible]:cv2.circle(frame,tuple(xy.astype(int)),3,(0,0,255),-1)
                    cv2.putText(frame,f'Frame {index}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
                    encoder.stdin.write(frame.tobytes())
                encoder.stdin.close()
                error=encoder.stderr.read().decode('utf-8',errors='replace')
                if encoder.wait():raise RuntimeError('FFmpeg failed: '+error[-2000:])
                with (dest/f'data{camera}.json').open('w',encoding='utf-8') as stream:json.dump(frames,stream,allow_nan=False)
            finally:
                cap.release()
                if encoder.poll() is None:encoder.kill();encoder.wait()
                if encoder.stdin and not encoder.stdin.closed:encoder.stdin.close()
                encoder.stderr.close()
            files.extend([str(dest/f'data{camera}.json'),str(video)])
            views.append({'camera':camera,'index':len(files)-1})
        return {'files':files,'videos':views,'viewer':'reprojection-videos','frames':len(points),'scripts':['1_keypoints/demo/project_3d_keypoints.py (adapted from MouseIMU)']}
    except Exception:
        for target in created:target.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='3D关键点投影到2D程序')
    parser.add_argument('--keypoints_3d', type=str, required=True,
                        help='triangulated_keypoints_3d.npy文件路径')
    parser.add_argument('--calib_folder', type=str, required=True,
                        help='相机标定文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='输出文件夹路径')
    parser.add_argument('--reference_json_folder', type=str, default=None,
                        help='参考JSON文件夹路径（用于确定instance数量）')

    parser.add_argument('--videos',help='Original data1.mp4 ... data8.mp4 directory; enables video overlays.')
    parser.add_argument('--input_unit',choices=UNITS,default='m')
    parser.add_argument('--calibration_unit',choices=UNITS,default='m')
    parser.add_argument('--result',help='Optional result manifest for the UI worker.')
    args = parser.parse_args()

    try:
        if args.videos:
            result=render_reprojection(args.keypoints_3d,args.calib_folder,args.videos,args.output_folder,args.input_unit,args.calibration_unit)
            if args.result:Path(args.result).write_text(json.dumps(result,indent=2),encoding='utf-8')
            return 0
        # 创建投影器
        projector = KeypointProjector(args.calib_folder)

        # 处理关键点文件
        projector.process_keypoints_file(
            args.keypoints_3d,
            args.output_folder,
            args.reference_json_folder
        )

        print("\nDone 投影完成！")

    except Exception as e:
        print(f"ERROR 错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("示例用法:")
        print("python project_3d_keypoints.py \\")
        print("  --keypoints_3d '/mnt/nas/exports/Export 06-08-2025 15-21-53_output/triangulation/triangulated_keypoints_3d.npy' \\")
        print("  --calib_folder '/mnt/nas/exports/Export 06-08-2025 15-21-53_output/calib/metashape/calibration' \\")
        print("  --output_folder './projected_keypoints' \\")
        print("  --reference_json_folder '/mnt/nas/exports/Export 06-08-2025 15-22-14_output/pred/Export 06-08-2025 15-22-14'")
    else:
        exit(main())
