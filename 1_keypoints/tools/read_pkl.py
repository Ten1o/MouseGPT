import pickle
import numpy as np
import shutil
import os
f = open('230814_test.pkl','rb')
data = pickle.load(f)
svdir = r'D:\yuyh\mmpose\animal\utils\filter_ap10k_train'
low_path = []
count = 0
for da in data:
    img_path = da['img_path']
    pred_instances = da['pred_instances']
    score = pred_instances['keypoint_scores']
    score_mean = np.mean(score[score>0.1])
    if score_mean < 0.9:
        low_path.append(img_path)
        count += 1
        root,name = os.path.split(img_path)
        name = name.split('_',)[0]
        subsvdir = os.path.join(svdir,name)
        os.makedirs(subsvdir,exist_ok=True)
        og_path = os.path.join(root,f'{name}.png')
        shutil.copy(og_path,subsvdir)
        shutil.copy(img_path,subsvdir)

print(count)