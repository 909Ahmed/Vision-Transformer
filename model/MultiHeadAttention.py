import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class MultiHead(Layer):
    def __init__(self, num_heads, model_dim, mat_dim):
        super(MultiHead, self).__init__(name="MSA")
        
        self.WQ = Dense(mat_dim)
        self.WK = Dense(mat_dim)
        self.WV = Dense(mat_dim)
        
        self.WO = Dense(model_dim)        
        
        self.model_dim = model_dim        
        self.heads = num_heads
        self.mat_dim = mat_dim
    
    def change_shape(self, x):
        
        shape = tf.shape(x)
        x = tf.reshape(x, shape=(shape[0], shape[1], self.heads, -1))
        return tf.transpose(x, perm=(0, 2, 1, 3))
        
    def rechange_shape(self, x):
        
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        shape = tf.shape(x)
        return tf.reshape(x, shape=(shape[0], shape[1], self.mat_dim))        
        
    def attention(self, queries, keys, values):
        
        scores = tf.linalg.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(self.mat_dim, 'float32')) 
        weights = tf.nn.softmax(scores)
        return tf.linalg.matmul(weights, values)
        
    def call(self, x):

        Qp = self.change_shape(self.WQ(x))
        Kp = self.change_shape(self.WK(x))
        Vp = self.change_shape(self.WV(x))
        
        output = self.attention(Qp, Kp, Vp)
        output = self.rechange_shape(output)
        
        return self.WO(output)