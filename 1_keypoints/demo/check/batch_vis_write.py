import os
import subprocess
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
label_folder = os.environ['MOUSEGPT_CALIBRATION_DIR']
demo_sample_folder = os.environ['MOUSEGPT_DATA_ROOT']
output_folder = os.environ['MOUSEGPT_OUTPUT_DIR']

labels = sorted(glob(f'{label_folder}/*.mat'))
samples = sorted(glob(f'{demo_sample_folder}/*/'))

for mat in tqdm(labels):
    name = os.path.basename(mat).split('.')[0]
    date, group = name.split('_')[:2]

    demo_folder = f'{demo_sample_folder}/20{date}/{date}_{group}/demo/{name}'
    command = [
        sys.executable, str(Path(__file__).resolve().with_name('vis_annotation_write.py')),
        '--label_path', mat, '--data_folder', demo_folder, '--ofolder', output_folder,
    ]
    subprocess.run(command, check=True)
