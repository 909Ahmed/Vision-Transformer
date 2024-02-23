import tensorflow as tf
from tensorflow.keras.layers import Dense
from model.PatchLayer import PatchEmbedding
from model.Transformer import Transformer

def VIT(
        emd_dim=128, 
        num_patches=256, 
        patch_size=16, 
        num_heads=8, 
        mat_dim=64, 
        num_classes=10
    ):
    
    i = tf.keras.layers.Input(shape=(256, 256, 3))
    x = PatchEmbedding(emd_dim, num_patches, patch_size) (i)
    
    x = Transformer(num_heads, emd_dim, mat_dim)(x)
    x = Transformer(num_heads, emd_dim, mat_dim)(x)
    x = Transformer(num_heads, emd_dim, mat_dim)(x)
    
    x = Dense(num_classes, activation = 'softmax')(x[:,0])

    return tf.keras.Model(i, x)