import pandas as pd
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
import random
import os
import json

annotation = r'data\mouse_v2\annotations\v2.json'
coco = COCO(annotation)
image_ids = coco.getImgIds()
random.shuffle(image_ids)
train_size = int(0.8 * len(image_ids))
train_ids = image_ids[:train_size]
test_ids = image_ids[train_size:]
test_ids = [955,959,963,967,968, 970,975,979,987,990,991,998,1009,1013,1018,1026]


# train_folder = r'D:\yuyh\mmpose\data\mouse_v2\train'
# test_folder = r'D:\yuyh\mmpose\data\mouse_v2\test'
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(test_folder, exist_ok=True)


train_images = coco.loadImgs(train_ids)
test_images = coco.loadImgs(test_ids)


train_annotations = coco.loadAnns(coco.getAnnIds(imgIds=train_ids))
test_annotations = coco.loadAnns(coco.getAnnIds(imgIds=test_ids))

train_data = {
    # "info": coco.dataset["info"],
    # "licenses": coco.dataset["licenses"],
    "images": train_images,
    "annotations": train_annotations,
    "categories": coco.dataset["categories"]
}

test_data = {
    # "info": coco.dataset["info"],
    # "licenses": coco.dataset["licenses"],
    "images": test_images,
    "annotations": test_annotations,
    "categories": coco.dataset["categories"]
}

# savdir = r'D:\yuyh\mmpose_change\data\mose_onlytop\annotations'
savdir =  '.'
# save_train = os.path.join(savdir, 'train.json')
# with open(save_train, 'w') as f:
#     json.dump(train_anns, f)
save_test = os.path.join(savdir, 'test_dlc.json')
# with open(save_test, 'w') as f:
#     json.dump(test_anns, f)

# with open(save_train, "w") as f:
#     json.dump(train_data, f)

with open(save_test, "w") as f:
    json.dump(test_data, f)

# train_anns = []
# test_anns = []
# for img_id in image_ids:
#     img_info = coco.loadImgs(img_id)[0]
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)

#     if img_id in train_ids:
#         save_dir = train_folder
#         # train_anns.extend(anns)
#     elif img_id in test_ids:
#         save_dir = test_folder
#         # test_anns.extend(anns)
#     else:
#         continue
#     save_path = os.path.join(save_dir, f'{img_info["file_name"]}.json')
#     with open(save_path, 'w') as f:
#         json.dump(anns, f)
