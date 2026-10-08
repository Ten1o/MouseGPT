import pandas as pd
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
import random
import os
import json

# annotation = r'data\mose_onlytop\annotations\v3.json'
annotation = r'animal\test_newjson\dataset_test.json'
annotation = r'Z:\aspect_fixed\dataset_json\sddataset_test_v2.json'
annotation = r'Y:\mouse_0630\annotations\0630_10kpts.json'
annotation = r'D:\yuyh\mmpose\data\mouse_v2\annotations\v2.json'
annotation = r'D:\yuyh\mmpose\animal\utils\test_sd_out_ap10k-train-split2\annotations\ap10k_onlydog.json' ###
coco = COCO(annotation)
image_ids = coco.getImgIds()
random.seed(123456)
random.shuffle(image_ids)
train_size = int(0.9 * len(image_ids))
train_ids = image_ids[:train_size]
test_ids = image_ids[train_size:]

train_images = coco.loadImgs(train_ids)
test_images = coco.loadImgs(test_ids)


train_annotations = coco.loadAnns(coco.getAnnIds(imgIds=train_ids))
test_annotations = coco.loadAnns(coco.getAnnIds(imgIds=test_ids))

train_data = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": coco.dataset["categories"]
}

test_data = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": coco.dataset["categories"]
}


savdir = r'Z:\aspect_fixed\dataset_json'
savdir = r'Y:\mouse_0630\annotations'
savdir = r'D:\yuyh\mmpose\data\mouse_v2\annotations'
savdir = r'D:\yuyh\mmpose\animal\utils\test_sd_out_ap10k-train-split2\annotations' ##
save_train = os.path.join(savdir, 'train_dog.json')
save_test = os.path.join(savdir, 'test_dog.json')
with open(save_train, "w") as f:
    json.dump(train_data, f)
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
