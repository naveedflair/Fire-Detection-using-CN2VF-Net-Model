## Paper Title: 
# CN2VF-Net: A Hybrid Convolutional Neural Network and Vision Transformer Framework for Multi-Scale Fire Detection in Complex Environments
### This repository implements an advanced fire detection system combining:

- Transformer architectures for global context understanding

- CNN backbones (EfficientNetB0) for local feature extraction

- Multi-scale attention mechanisms for precise fire region localization
### Key features:
✅ Hybrid architecture for improved accuracy
✅ Multi-scale processing for varying fire sizes
✅ Optimized for both performance and efficiency
✅ Comprehensive evaluation metrics (mAP50, mAP50-95, F1-score)
✅ Grad-CAM visualization for model interpretability

### Architecture Components
#### Feature Extraction
    - EfficientNetB0 backbone

    - Multi-scale feature pyramid (P3-P5)

#### Transformer Module
    - Patch embeddings (16×16 patches)
    - 4 transformer layers
    - 8 attention heads

#### Decoder
    - Progressive upsampling
    - Skip connections
    - Attention-guided feature fusion

D-Fire/
├── train/
│   ├── images/
│   └── labels/ (YOLO format)
└── test/
    ├── images/
    └── labels/
