import os
import json
import numpy as np
import json
import cv2

class DatasetHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist.")
        self.load_kpts()
        self.load_velocity()

    def load_kpts(self):
        kpt_path = os.path.join(self.dataset_path, 'kpts/tri_filter.npy')
        self.kpts = np.load(kpt_path)
    
    def load_velocity(self):
        vel_path = os.path.join(self.dataset_path, 'velocity.json')
        with open(vel_path, 'r') as file:
            data = json.load(file)
        self.velocity = data

    def acquire_kpts(self, frame_id):
        return self.kpts[frame_id]

    def acquire_velocity(self, frame_id):
        return self.velocity[str(frame_id)]

    def acquire_description(self, frame_id):
        desc_path = os.path.join(self.dataset_path, 'descriptions/{}.json'.format(str(frame_id).zfill(8)))
        with open(desc_path, 'r') as file:
            data = json.load(file)
        description = data["choices"][0]["message"]["content"]
        return description
    
    def acquire_img(self, frame_id):
        img_path = os.path.join(self.dataset_path, 'concat_images/{}.jpg'.format(str(frame_id).zfill(8)))
        img = cv2.imread(img_path)
        return img

