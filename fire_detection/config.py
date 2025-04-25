CONFIG = {
    'IMAGE_SIZE': (416, 416),
    'BATCH_SIZE': 16,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'NUM_CLASSES': 1,
    'DATASET_PATH': '/kaggle/input/fire-dataset/fire/D-Fire',
    'MODEL_TYPE': 'hybrid',
    'NUM_PATCHES': 16,
    'PATCH_SIZE': 26,
    'TRANSFORMER_LAYERS': 4,
    'TRANSFORMER_HEADS': 8,
    'TRANSFORMER_DIM': 256,
    'TRANSFORMER_MLP_DIM': 512,
    'ATTENTION_SCALES': [0.5, 1, 2],
}

# Set random seeds for reproducibility
import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)