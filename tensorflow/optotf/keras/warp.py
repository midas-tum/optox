import tensorflow as tf
import optotf.warp

class Warp(tf.keras.layers.Layer):
    def __init__(self, channel_last=True, mode='zeros'):
        super().__init__()
        self.channel_last = channel_last
        self.op = optotf.warp.warp_2d
        self.mode = mode
    
    def call(self, x, u):
        if self.channel_last:
            x = tf.transpose(x, [0, 3, 1, 2])

        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            out = tf.complex(self.op(tf.math.real(x), u, tf.math.real(x), mode=self.mode)[0], self.op(tf.math.imag(x), u, tf.math.real(x), mode=self.mode)[0])
        else:
            out, _ = self.op(x, u, x, mode=self.mode)
        
        if self.channel_last:
            out = tf.transpose(out, [0, 2, 3, 1])

        return out

class WarpTranspose(tf.keras.layers.Layer):
    def __init__(self, channel_last=True, mode='zeros'):
        super().__init__()
        self.channel_last = channel_last
        self.op = optotf.warp.warp_2d_transpose
        self.mode = mode
    
    def call(self, grad_out, u, x):
        if self.channel_last:
            grad_out = tf.transpose(grad_out, [0, 3, 1, 2])
            x = tf.transpose(x, [0, 3, 1, 2])

        if grad_out.dtype == tf.complex64 or grad_out.dtype == tf.complex128:
            grad_x = tf.complex(self.op(tf.math.real(grad_out), u, tf.math.real(x), mode=self.mode)[0], self.op(tf.math.imag(grad_out), u, tf.math.imag(x), mode=self.mode)[0])
        else:
            grad_x, _ = self.op(grad_out, u, x, mode=self.mode)
        
        if self.channel_last:
            grad_x = tf.transpose(grad_x, [0, 2, 3, 1])
            
        return grad_x
