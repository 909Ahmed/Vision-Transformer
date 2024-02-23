import tensorflow as tf
from tensorflow.keras.layers import Layer
from model.MultiHeadAttention import MultiHead
from model.MLP import MLP

class Transformer(Layer):
    
    def __init__(self, num_heads, model_dim, mat_dim):
        super(Transformer, self).__init__()
        
        self.attention = MultiHead(num_heads, model_dim, mat_dim)
        self.ff = MLP(model_dim)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        
    def call(self, i):
        
        x = self.norm1(i)
        x = self.attention(x)
        
        res1 = self.add1([x, i])
        
        x = self.norm2(res1)
        x = self.ff(x)
        
        res2 = self.add2([x, res1])
        
        return res2