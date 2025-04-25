import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

def mlp(x, hidden_units, dropout_rate=0.1):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, num_heads, dim, mlp_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.position_embedding = layers.Embedding(input_dim=1000, output_dim=dim)
        
        self.layer_norm1 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.mha = [layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout_rate
        ) for _ in range(num_layers)]
        self.layer_norm2 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.mlp_layers = []
        for _ in range(num_layers):
            mlp_block = []
            for units in [mlp_dim, dim]:
                mlp_block.append(layers.Dense(units, activation=tf.nn.gelu))
                mlp_block.append(layers.Dropout(dropout_rate))
            self.mlp_layers.append(mlp_block)
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, encoded_patches):
        positions = tf.range(start=0, limit=tf.shape(encoded_patches)[1], delta=1)
        position_embeddings = self.position_embedding(positions)
        x = encoded_patches + position_embeddings
        
        for i in range(self.num_layers):
            x1 = self.layer_norm1[i](x)
            attention_output = self.mha[i](x1, x1)
            x2 = layers.Add()([attention_output, x])
            x3 = self.layer_norm2[i](x2)
            mlp_output = x3
            for layer in self.mlp_layers[i]:
                mlp_output = layer(mlp_output)
            x = layers.Add()([mlp_output, x2])
        
        encoded_patches = self.final_layer_norm(x)
        return encoded_patches

@register_keras_serializable()
class DecoderModule(layers.Layer):
    def __init__(self):
        super(DecoderModule, self).__init__()
        self.upsample1 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')
        self.c2_conv = layers.Conv2D(128, 1, padding='same')
        self.conv1 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        
        self.upsample2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')
        self.c1_conv = layers.Conv2D(64, 1, padding='same')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        
        self.upsample3 = layers.Conv2DTranspose(32, 4, strides=2, padding='same')
        self.conv3 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        
        self.output_conv = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
    
    def call(self, inputs):
        features, cnn_features = inputs
        x = features
        
        x = self.upsample1(x)
        c2_conv = self.c2_conv(cnn_features[1])
        x = layers.Concatenate()([x, c2_conv])
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.upsample2(x)
        c1_conv = self.c1_conv(cnn_features[0])
        x = layers.Concatenate()([x, c1_conv])
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        output = self.output_conv(x)
        return output

@register_keras_serializable()
class FeatureFusionModule(layers.Layer):
    def __init__(self, patch_size=26, dim=256):
        super(FeatureFusionModule, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.transpose1 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')
        self.transpose2 = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')
        self.c3_conv = layers.Conv2D(256, 1, padding='same')
        self.c4_conv = layers.Conv2D(256, 1, padding='same')
        self.fusion_conv = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.fusion_bn = layers.BatchNormalization()
        self.transformer_conv = layers.Conv2D(256, 1, padding='same')

    def call(self, inputs, training=None):
        transformer_features, cnn_features = inputs
        batch_size = tf.shape(transformer_features)[0]
        
        reshaped_transformer = tf.reshape(transformer_features, [batch_size, 16, 16, self.dim])
        transformer_resized = self.transpose1(reshaped_transformer)
        transformer_resized = self.transpose2(transformer_resized)
        transformer_resized = tf.image.resize(transformer_resized, (52, 52))
        
        c3_conv = self.c3_conv(cnn_features[2])
        c3_up = tf.image.resize(c3_conv, (52, 52))
        
        c4_conv = self.c4_conv(cnn_features[3])
        c4_up = tf.image.resize(c4_conv, (52, 52))
        
        transformer_resized = self.transformer_conv(transformer_resized)
        
        fused = layers.Concatenate(axis=-1)([transformer_resized, c3_up, c4_up])
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused, training=training)
        
        return fused

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, 52, 52, 256)

@register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.projection = layers.Dense(dim)

    def build(self, input_shape):
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_channels = input_shape[3]
        patch_dim = self.patch_size * self.patch_size * self.input_channels
        self.projection.build((None, patch_dim))
        self.built = True

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        patch_embeddings = self.projection(patches)
        return patch_embeddings

    def compute_output_shape(self, input_shape):
        h_patches = self.input_height // self.patch_size
        w_patches = self.input_width // self.patch_size
        num_patches = h_patches * w_patches
        return (input_shape[0], num_patches, self.dim)

def attention_module(feature_maps, scales):
    attention_maps = []
    
    for scale in scales:
        if scale != 1.0:
            if scale < 1.0:
                scale_factor = int(1 / scale)
                x = layers.AveragePooling2D(pool_size=(scale_factor, scale_factor))(feature_maps)
            else:
                scale_factor = int(scale)
                x = layers.UpSampling2D(size=(scale_factor, scale_factor))(feature_maps)
        else:
            x = feature_maps
        
        attn = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        attn = layers.BatchNormalization()(attn)
        attn = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(attn)
        
        if scale != 1.0:
            attn = tf.image.resize(attn, tf.shape(feature_maps)[1:3])
        
        attention_maps.append(attn)
    
    combined_attention = layers.Add()(attention_maps)
    combined_attention = layers.Lambda(lambda x: x / len(scales))(combined_attention)
    
    attended_features = layers.Multiply()([feature_maps, combined_attention])
    output = layers.Add()([feature_maps, attended_features])
    
    return output, combined_attention

def decoder_module(features, cnn_features):
    # Start with the fused features
    x = features
    
    # Upsample and fuse with CNN features (c2, mid level)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    c2_conv = layers.Conv2D(128, 1, padding='same')(cnn_features[1])
    x = layers.Concatenate()([x, c2_conv])
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Upsample and fuse with CNN features (c1, low level)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    c1_conv = layers.Conv2D(64, 1, padding='same')(cnn_features[0])
    x = layers.Concatenate()([x, c1_conv])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Final upsampling to original image size
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Output segmentation mask
    output = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    return output