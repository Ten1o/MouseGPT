from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import scipy.io as sio
import pickle

# Load ground truth annotations
ann_file = 'test_dlc.json'
coco_gt = COCO(ann_file)
# Load predicted keypoints
data = sio.loadmat('dlc_result/dlc_result.mat')
pred_keypoints = [] 
gt_keypoints = []
img_ids = [955,959,963,967,968, 970,975,979,987,990,991,998,1009,1013,1018,1026]
for key in data.keys():
    if 'jpg' in key:
        pred = np.asarray(data[key],dtype = np.float32)
        pred = pred.reshape(-1, 3)
        pred_keypoints.append(pred)
for img_id in img_ids:
    gt_kp = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))[0]['keypoints']
    gt_kp = np.asarray(gt_kp, dtype = np.float32)
    gt_kp = gt_kp.reshape(-1, 3)
    gt_keypoints.append(gt_kp)
    
gt_keypoints

gt_keypoints = np.array(gt_keypoints)
pred_keypoints = np.array(pred_keypoints)
loss = 0
for i in range(len(pred_keypoints)):
    gt = gt_keypoints[i]
    pred = pred_keypoints[i]
    valid = 0
    loss_item = 0
    for j in range(len(gt)):
        if gt[j][2] > 0 :
            valid += 1
            one_ = np.mean((gt[j][:2] - pred[j][:2]) ** 2)
            if(one_ > 1000):
                loss_item = loss_item
                valid -= 1
            else:
                loss_item += one_
            # if(loss_item > 1000):
            #     print(j)
    loss_item = loss_item/valid
    loss += loss_item
loss = loss / len(pred_keypoints)
print(loss)
 # List of predicted keypoints for each image in the dataset

# Prepare results in COCO evaluation format
# coco_dt = []
# for i, keypoints in zip(img_ids, pred_keypoints):
#     image_id = i  # Image IDs start from 1 in COCO annotations
#     keypoints = np.asarray(keypoints, dtype=np.float32)
#     keypoints = keypoints.reshape(-1, 3)
#     sc = np.mean(keypoints[:,2:])
#     for i in range(len(keypoints)):
#         if keypoints[i][2]>=0.2:
#             keypoints[i][2] = 2
#         else:
#             keypoints[i][2] = 0    
#     kp = keypoints.flatten().tolist()
#     for j, keypoint in enumerate(keypoints):
#         coco_dt.append({
#             'image_id': image_id,
#             'category_id': 5,  # Assuming a single category for keypoints
#             'keypoints': kp,
#             'score': sc # Confidence score for the keypoint
#         })

# coco = COCO()
# coco.dataset['annotations'] = coco_dt
# coco.createIndex()
# # Load ground truth annotations and predicted keypoints into COCO evaluation object
# coco_eval = COCOeval(coco_gt, coco, 'keypoints')

# # Run evaluation
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()

# # Get OKS metric
# oks_metric = coco_eval.stats[9]  # OKS metric is the 10th element in the stats array
# print('OKS Metric:', oks_metric)