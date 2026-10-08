import pickle
import numpy as np
import shutil
import os
import glob

path1 = r'D:\yuyh\mmpose\animal\utils\test_sd_out_ap10k-train-split1\imgs'
path2 = r'D:\yuyh\mmpose\animal\utils\test_sd_out_ap10k-train-split1\imgs - 副本'
check_file = r'D:\yuyh\mmpose\animal\utils\filter_ap10k_train'

files1 = set(os.listdir(path1))
files2 = set(os.listdir(path2))
diffs = files1.difference(files2) 
count = 0
names = []
for diff in diffs:
    name = diff.split('_',)[0]
    path = os.path.join(check_file,name,diff)
    find = glob.glob(path)
    if(len(find)>0):
        names.append(diff)
    count += len(find)
print(names)
print(count)




# print(data.shape)