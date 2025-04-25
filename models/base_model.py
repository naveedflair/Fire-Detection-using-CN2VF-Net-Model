from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def create_cnn_encoder(inputs):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    c1 = base_model.get_layer('block2a_expand_activation').output
    c2 = base_model.get_layer('block3a_expand_activation').output
    c3 = base_model.get_layer('block5c_add').output
    c4 = base_model.get_layer('top_activation').output
    return base_model, [c1, c2, c3, c4]