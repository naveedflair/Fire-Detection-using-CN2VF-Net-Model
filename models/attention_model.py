from tensorflow.keras import layers, models
from .base_model import create_cnn_encoder
from .layers import attention_module, decoder_module

def build_dynamic_multi_scale_attention_model(input_shape=(416, 416, 3),
                                            attention_scales=[0.5, 1, 2]):
    inputs = layers.Input(shape=input_shape)
    
    cnn_model, cnn_features = create_cnn_encoder(inputs)
    p5 = cnn_features[3]
    
    p4 = layers.Conv2D(256, 1, padding='same')(cnn_features[2])
    p4 = layers.BatchNormalization()(p4)
    p4 = layers.ReLU()(p4)
    
    p3 = layers.Conv2D(128, 1, padding='same')(cnn_features[1])
    p3 = layers.BatchNormalization()(p3)
    p3 = layers.ReLU()(p3)
    
    p5_up = layers.UpSampling2D(size=(2, 2))(p5)
    p4 = layers.Add()([p4, p5_up])
    
    p4_up = layers.UpSampling2D(size=(2, 2))(p4)
    p3 = layers.Add()([p3, p4_up])
    
    p3_attended, _ = attention_module(p3, attention_scales)
    
    features = layers.Conv2D(128, 3, padding='same')(p3_attended)
    features = layers.BatchNormalization()(features)
    features = layers.ReLU()(features)
    
    output = decoder_module(features, cnn_features)
    
    model = models.Model(inputs, output, name="DynamicMultiScaleAttention")
    return model