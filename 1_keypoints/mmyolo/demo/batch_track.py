import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='MMYOLO video demo')
parser.add_argument('-i', type=int, default=0,)
args = parser.parse_args()

video_folder = '/home/vrlab/Documents/Track-Anything/demo'
out_folder = '/home/vrlab/Documents/mmpose_mouse/mmyolo/demo/output/v2'
os.makedirs(out_folder, exist_ok=True)

script_path = '/home/vrlab/Documents/mmpose_mouse/mmyolo/demo/track_demo.py'

for id, video in enumerate(sorted(glob(f'{video_folder}/*min.mp4'))):
    
    # if id % 2 == args.i:
    #     continue
    
    cmd = f'python {script_path} --video {video} --out {out_folder}/data{id+1}.mp4'
    print(cmd)
    
    os.system(cmd)