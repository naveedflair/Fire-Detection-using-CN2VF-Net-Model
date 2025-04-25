from tensorflow.keras import layers, models
from .base_model import create_cnn_encoder
from .layers import (PatchEmbedding, TransformerEncoder, 
                    FeatureFusionModule, DecoderModule)

def build_hybrid_transformer_cnn_model(input_shape=(416, 416, 3), 
                                     num_patches=16, 
                                     patch_size=26, 
                                     transformer_layers=4, 
                                     transformer_heads=8,
                                     transformer_dim=256,
                                     transformer_mlp_dim=512):
    inputs = layers.Input(shape=input_shape)
    
    cnn_model, cnn_features = create_cnn_encoder(inputs)
    patch_embedding_layer = PatchEmbedding(patch_size, transformer_dim)
    patch_embeddings = patch_embedding_layer(inputs)
    
    transformer_encoder = TransformerEncoder(
        num_layers=transformer_layers,
        num_heads=transformer_heads,
        dim=transformer_dim,
        mlp_dim=transformer_mlp_dim
    )
    transformer_features = transformer_encoder(patch_embeddings)
    
    feature_fusion = FeatureFusionModule(patch_size=patch_size, dim=transformer_dim)
    fused_features = feature_fusion([transformer_features, cnn_features])
    
    decoder = DecoderModule()
    output = decoder([fused_features, cnn_features])
    
    model = models.Model(inputs, output, name="HybridTransformerCNN")
    return model