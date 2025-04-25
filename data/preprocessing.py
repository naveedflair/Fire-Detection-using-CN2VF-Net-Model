import os
import numpy as np
from sklearn.model_selection import train_test_split

def oversample_dataset(image_paths, label_paths):
    fire_indices = [i for i, label_path in enumerate(label_paths) if os.path.exists(label_path) and os.path.getsize(label_path) > 0]
    no_fire_indices = [i for i, label_path in enumerate(label_paths) if not (os.path.exists(label_path) and os.path.getsize(label_path) > 0)]
    
    oversampled_fire_indices = np.random.choice(fire_indices, size=len(no_fire_indices), replace=True)
    all_indices = np.concatenate([no_fire_indices, oversampled_fire_indices])
    
    return [image_paths[i] for i in all_indices], [label_paths[i] for i in all_indices]

def load_dataset(dataset_path):
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    
    test_images_dir = os.path.join(test_dir, 'images')
    test_labels_dir = os.path.join(test_dir, 'labels')
    
    train_image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    train_image_paths = [os.path.join(train_images_dir, f) for f in train_image_files]
    train_label_paths = [os.path.join(train_labels_dir, f"{os.path.splitext(f)[0]}.txt") for f in train_image_files]
    
    test_image_paths = [os.path.join(test_images_dir, f) for f in test_image_files]
    test_label_paths = [os.path.join(test_labels_dir, f"{os.path.splitext(f)[0]}.txt") for f in test_image_files]
    
    train_image_paths, train_label_paths = oversample_dataset(train_image_paths, train_label_paths)
    
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_image_paths, train_label_paths, test_size=0.2, random_state=42
    )
    
    return (train_imgs, train_labels), (val_imgs, val_labels), (test_image_paths, test_label_paths)