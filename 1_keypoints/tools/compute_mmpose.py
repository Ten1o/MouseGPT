from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import scipy.io as sio
import pickle

# Load ground truth annotations
ann_file = r'D:\yuyh\mmpose\data\mouse_v2\annotations\test.json'
coco_gt = COCO(ann_file)
# Load predicted keypoints
result_path = r'D:\yuyh\mmpose\230621_test.pkl'
with open(result_path, 'rb') as f:
    data = pickle.load(f)
pred_keypoints = []
pair = []
img_ids = coco_gt.getImgIds()
for i in range(len(data)):
    gt_kp = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=data[i]['img_id']))[0]['keypoints']
    gt_kp = np.asarray(gt_kp, dtype = np.float32)
    gt_kp = gt_kp.reshape(-1, 3)
    one_ = {'kpts':data[i]['pred_instances']['keypoints'], 'gt':gt_kp}
    pair.append(one_)

loss = 0
for i in range(len(pair)):
    gt = pair[i]['gt']
    pred = pair[i]['kpts'].squeeze()
    valid = 0
    loss_item = 0
    for j in range(len(gt)):
        if gt[j][2] > 0 :
            valid += 1
            loss_item += np.mean((gt[j][:2] - pred[j][:2]) ** 2)
            # if(loss_item > 1000):
            #     print(j)
    loss_item = loss_item/valid
    loss += loss_item
loss = loss / len(pair)
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