import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from einops import repeat
from einops.layers.tensorflow import Rearrange


class PatchEmbedding(Layer):
    
    def __init__(self, emd_dim, num_patches, patch_size):
        super(PatchEmbedding, self).__init__(name='PATCH_LAYER')
        
        self.patch_embedding = tf.keras.Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            Dense(units=emd_dim),
        ])

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, emd_dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, emd_dim]))

    def call(self, input_tensor):
        
        x = self.patch_embedding(input_tensor)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=64)
        x = tf.keras.layers.Concatenate(axis = 1)([cls_tokens, x])
        x += self.pos_embedding
                
        return x