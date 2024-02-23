import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class MLP(Layer):
    
    def __init__(self, emd_dim):
        super(MLP, self).__init__()
        self.emd_dim = emd_dim
        self.ff = tf.keras.Sequential([
            Dense(self.emd_dim * 2, activation=tf.keras.activations.gelu),
            Dense(self.emd_dim)
        ])
        
    def call(self, input_tensor):
                
        return self.ff(input_tensor)