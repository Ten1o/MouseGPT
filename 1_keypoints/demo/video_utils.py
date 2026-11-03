"""Video codec utilities for keypoint visualization outputs."""
from pathlib import Path
import subprocess

def ffmpeg_executable():
    import shutil
    import sys
    executable=shutil.which('ffmpeg')
    bundled=Path(sys.prefix)/'Library/bin/ffmpeg.exe'
    if not executable and bundled.is_file():executable=str(bundled)
    if not executable:
        try:
            import imageio_ffmpeg
            executable=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:raise ValueError('FFmpeg is missing from this environment. Select the managed core environment on the Environments page.') from None
    return executable

def pose_videos(folder,output_folder):
    """Reuse MMPose overlays, converting its mp4v codec for browser playback."""
    folder=Path(folder);dest=Path(output_folder)
    sources=[folder/f'data{i}.mp4' for i in range(1,9)]
    if not all(p.is_file() for p in sources):
        raise ValueError('Select the pose preview directory containing data1.mp4 ... data8.mp4.')
    dest.mkdir(parents=True,exist_ok=True)
    files=[]
    for view,source in enumerate(sources,1):
        target=dest/f'data{view}.mp4'
        if target.exists():raise ValueError(f'Output already exists: {target}')
        print(f'Preparing 2D overlay video {view}/8: {source}',flush=True)
        subprocess.run([ffmpeg_executable(),'-nostdin','-v','error','-n','-i',str(source),'-an','-c:v','libx264','-preset','veryfast','-crf','23','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
        files.append(str(target))
    return {'files':files,'viewer':'pose-videos','videos':[{'camera':i,'index':i-1} for i in range(1,9)],'scripts':['1_keypoints/demo/inferencer_demo.py (existing overlay videos)','1_keypoints/demo/video_utils.py:pose_videos (browser codec conversion)']}
