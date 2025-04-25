import os
import cv2
import numpy as np
import albumentations as A
from tensorflow.keras.utils import Sequence

class FireDataGenerator(Sequence):
    def __init__(self, image_paths, label_paths, batch_size=8, image_size=(416, 416), augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomRotate90(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.CLAHE(p=0.1),
                A.HueSaturationValue(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return max(1, len(self.image_paths) // self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        labels = []
        
        for img_path, label_path in zip(batch_x, batch_y):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
                img = img / 255.0
                
                label_mask = np.zeros((*self.image_size, 1), dtype=np.float32)
                
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    img_h, img_w = self.image_size
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id, x_center, y_center, width, height = map(float, parts)
                            if cls_id == 1:
                                x1 = int((x_center - width/2) * img_w)
                                y1 = int((y_center - height/2) * img_h)
                                x2 = int((x_center + width/2) * img_w)
                                y2 = int((y_center + height/2) * img_h)
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(img_w, x2), min(img_h, y2)
                                label_mask[y1:y2, x1:x2, 0] = 1.0
                
                if self.transform and self.augment:
                    augmented = self.transform(image=img, mask=label_mask)
                    img = augmented['image']
                    label_mask = augmented['mask']
                
                images.append(img)
                labels.append(label_mask)
            except Exception as e:
                continue
        
        if len(images) == 0:
            return np.zeros((1, *self.image_size, 3)), np.zeros((1, *self.image_size, 1))
        
        return np.array(images), np.array(labels)
    
    def get_labels(self):
        labels = []
        for label_path in self.label_paths:
            label_mask = np.zeros((*self.image_size, 1), dtype=np.float32)
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                img_h, img_w = self.image_size
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, x_center, y_center, width, height = map(float, parts)
                        if cls_id == 1:
                            x1 = int((x_center - width/2) * img_w)
                            y1 = int((y_center - height/2) * img_h)
                            x2 = int((x_center + width/2) * img_w)
                            y2 = int((y_center + height/2) * img_h)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_w, x2), min(img_h, y2)
                            label_mask[y1:y2, x1:x2, 0] = 1.0
            labels.append(label_mask)
        return np.array(labels)