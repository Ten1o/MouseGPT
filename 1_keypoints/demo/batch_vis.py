"""Filter reconstructed NPY keypoints and render their 3D skeleton animation.

This script consumes triangulation output; it does not run triangulation.
"""
import argparse
import json
import os
from pathlib import Path
import math
import numpy as np
from scipy.signal import savgol_filter


colors = [[1.,         0.,         0.,         1       ],
            [1.         ,0.26086957 ,0.         ,1       ],
            [1.         ,0.52173913 ,0.         ,1       ],
            [1.         ,0.7826087  ,0.         ,1       ],
            [0.95652174 ,1.         ,0.         ,1       ],
            [0.69565217 ,1.         ,0.         ,1       ],
            [0.43478261 ,1.         ,0.         ,1       ],
            [0.17391304 ,1.         ,0.         ,1       ],
            [0.         ,1.         ,0.08695652 ,1       ],
            [0.         ,1.         ,0.34782609 ,1       ],
            [0.         ,1.         ,0.60869565 ,1       ],
            [0.         ,1.         ,0.86956522 ,1       ],
            [0.         ,0.86956522 ,1.         ,1       ],
            [0.         ,0.60869565 ,1.         ,1       ],
            [0.         ,0.34782609 ,1.         ,1       ],
            [0.         ,0.08695652 ,1.         ,1       ],
            [0.17391304 ,0.         ,1.         ,1       ],
            [0.43478261 ,0.         ,1.         ,1       ],
            [0.69565217 ,0.         ,1.         ,1       ],
            [0.95652174 ,0.         ,1.         ,1       ],
            [1.         ,0.         ,0.52173913 ,1       ],
            [1.         ,0.         ,0.26086957 ,1       ],
            [1.         ,0.         ,1.         ,1       ],
            [1.         ,0.         ,1.         ,1       ],
            [1.         ,0.         ,0.         ,1       ],
            [1.         ,0.         ,0.         ,1       ]]

bones = [[ 1,  2],
        [ 2,  3],
        [ 1,  3],
        [ 1,  4],
        [ 2,  4],
        [ 3,  4],
        [ 4,  5],
        [ 7,  8],
        [ 8,  9],
        [ 9, 10],
        [11, 12],
        [12, 13],
        [13, 14],
        [14,  4],
        [15, 16],
        [16, 17],
        [17, 18],
        [18,  4],
        [19, 20],
        [20, 21],
        [22, 23],
        [23, 24],
        [ 6,  5],
        [ 6,  7],
        [ 7, 21],
        [ 7, 24]]


def filter(data_3d):
    """The original batch_vis Savitzky-Golay settings: window 15, order 5."""
    return savgol_filter(data_3d,15,5,axis=0)


def prepare_coordinates(source,unit):
    a=np.load(source,allow_pickle=False)
    if a.ndim!=3 or a.shape[1:]!=(24,3) or len(a)<15:
        raise ValueError('Expected at least 15 frames with shape (frames, 24, 3).')
    units={'m':1.,'cm':.01,'mm':.001}
    if unit not in units:raise ValueError('Select the source spatial unit: m, cm or mm.')
    missing=~np.isfinite(a).all(axis=2)|(a==0).all(axis=2)
    a=a.astype(float)
    for point in range(24):
        good=np.flatnonzero(~missing[:,point])
        if len(good)<2 or good[0]!=0 or good[-1]!=len(a)-1:
            raise ValueError(f'Keypoint {point}: missing endpoint or insufficient observations. Correct the source before filtering.')
        for dim in range(3):a[:,point,dim]=np.interp(np.arange(len(a)),good,a[good,point,dim])
    return filter(a)*units[unit],int(missing.sum())


def render_animation(data,target,start,end,fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,FFMpegWriter
    from video_utils import ffmpeg_executable
    fig=plt.figure(figsize=(6.4,4.8));ax=fig.add_subplot(111,projection='3d')
    try:
        points=data[start:end].reshape(-1,3)
        low=points.min(axis=0);high=points.max(axis=0);span=np.maximum(high-low,.01)
        for dim,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):setter(low[dim]-span[dim]*.1,high[dim]+span[dim]*.1)
        ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
        ax.set_box_aspect(span)
        scatter,=ax.plot([],[],[],'ro',markersize=3)
        lines=[ax.plot([],[],[],color='b',linewidth=1.5)[0] for _ in bones]
        title=ax.set_title('')
        def update(index):
            frame=data[index];scatter.set_data_3d(frame[:,0],frame[:,1],frame[:,2])
            for line,(a,b) in zip(lines,bones):
                segment=frame[[a-1,b-1]];line.set_data_3d(segment[:,0],segment[:,1],segment[:,2])
            title.set_text(f'Frame {index}')
            if (index-start)%30==0:print(f'3D animation: {index-start+1}/{end-start}',flush=True)
            return [scatter,*lines,title]
        animation=FuncAnimation(fig,update,frames=range(start,end),interval=1000/fps,cache_frame_data=False)
        with matplotlib.rc_context({'animation.ffmpeg_path':ffmpeg_executable()}):
            writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart'])
            animation.save(str(target),writer=writer,dpi=100)
    finally:plt.close(fig)


def playback_fps(source,reference_video=None,fps=None):
    if fps is not None:
        if not math.isfinite(fps) or fps<=0:raise ValueError('Playback FPS must be positive.')
        return float(fps)
    source=Path(source)
    reference=Path(reference_video).expanduser() if reference_video else source.parent.parent/'videos/data1.mp4'
    if reference_video and not reference.is_file():raise ValueError(f'Reference video not found: {reference}')
    if reference.is_file():
        import cv2
        cap=cv2.VideoCapture(str(reference))
        try:
            value=cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(value) or value<=0:raise ValueError(f'Cannot read FPS from {reference}')
            return float(value)
        finally:cap.release()
    return 60.


def process_file(source,output_dir=None,unit='m',start=0,frames=0,fps=None,reference_video=None):
    source=Path(source).expanduser().resolve()
    if not source.is_file():raise ValueError(f'NPY file not found: {source}')
    dest=Path(output_dir).expanduser().resolve() if output_dir else source.parent
    dest.mkdir(parents=True,exist_ok=True)
    npy=dest/'tri_filter.npy';video=dest/'tri_filter.mp4'
    for target in (npy,video):
        if target.exists():raise ValueError(f'Output already exists; choose a new output directory: {target}')
    data,filled=prepare_coordinates(source,unit)
    if not 0<=start<len(data) or frames<0:raise ValueError('Start frame must be in range; preview frames must be >= 0 (0 = all).')
    fps=playback_fps(source,reference_video,fps)
    end=min(len(data),start+frames) if frames else len(data)
    print(f'Filtered {len(data)} frames, filled {filled} keypoints; converted {unit} to metres.',flush=True)
    # Write directly in the selected directory. Some SMB shares restrict writes
    # inside newly created hidden/dot-prefixed directories.
    created=[]
    try:
        with npy.open('xb') as handle:
            created.append(npy)
            np.save(handle,data)
        # Reserve the video name without overwriting an existing result. FFmpeg
        # writes into this file; no temporary output subdirectory is used.
        with video.open('xb'):created.append(video)
        render_animation(data,video,start,end,fps)
    except BaseException as exc:
        for target in reversed(created):
            try:target.unlink(missing_ok=True)
            except OSError as cleanup:print(f'Cannot remove incomplete output {target}: {cleanup}',flush=True)
        if isinstance(exc,PermissionError):
            raise PermissionError(f'Cannot write output files in {dest}. Check file creation/write permissions on this directory. No temporary subdirectory is used.') from exc
        raise
    return {'files':[str(npy),str(video)],'frames':len(data),'filled_points':filled,'unit':'m','viewer':'3d-video',
            'videos':[{'index':1}],'preview_start':start,'preview_frames':end-start,'fps':fps,
            'filter':{'window':15,'polynomial_order':5},'scripts':['1_keypoints/demo/batch_vis.py']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Existing triangulated coordinates, usually tri_new.npy.')
    parser.add_argument('--output-dir',type=Path,help='Default: beside the input NPY.')
    parser.add_argument('--unit',choices=['m','cm','mm'],default='m',help='Input spatial unit; output is metres.')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--frames',type=int,default=0,help='Animation length; default 0 = all frames.')
    parser.add_argument('--fps',type=float,default=None,help='Override playback FPS. Default: original video FPS, otherwise 60.')
    parser.add_argument('--reference-video',type=Path,help='Original video used to read playback FPS.')
    parser.add_argument('--result',type=Path,help='Optional result manifest JSON for callers.')
    args=parser.parse_args()
    if args.input:
        result=process_file(args.input,args.output_dir,args.unit,args.start,args.frames,args.fps,args.reference_video)
    else:
        from glob import glob
        pattern=os.environ.get('MOUSEGPT_DATE_GLOB')
        if not pattern:parser.error('Provide --input (tri_new.npy) or MOUSEGPT_DATE_GLOB for batch processing.')
        sources=[Path(group)/'tri_new.npy' for date in sorted(glob(pattern)) for group in sorted(glob(str(Path(date)/'*')))]
        result={'sessions':[process_file(source,None,args.unit,args.start,args.frames,args.fps,args.reference_video) for source in sources if source.is_file()]}
    if args.result:args.result.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
