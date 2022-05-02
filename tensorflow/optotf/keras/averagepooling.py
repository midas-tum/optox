import tensorflow as tf
import optotf.averagepooling
import numpy as np


class Averagepooling1d(tf.keras.layers.Layer):
    """ Averagepooling1d layer. """
    def __init__(self, pool_size=(2,), strides=(2,), alpha=1, beta=1, channel_first=False, mode='VALID',
                 dilations_rate=(1, 1), name="averagepooling1d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.averagepooling.averagepooling1d
        self.channel_first = channel_first
        self.layer_name = name
        self.mode = mode
        self.dilations_rate = dilations_rate
        self.argmax = argmax

    def call(self, inputs):
        out = self.op(inputs, pool_size=self.pool_size,
                      strides=self.strides,
                      alpha=self.alpha, beta=self.beta, name=self.layer_name,
                      dilations_rate=self.dilations_rate, channel_first=self.channel_first,
                      mode=self.mode)
        if inputs is not list:
            return out
        else:
            return tf.math.real(out), tf.math.imag(out)


class Averagepooling2d(tf.keras.layers.Layer):
    """ Averagepooling2d layer. """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), alpha=1, beta=1, channel_first=False, mode='VALID',
                 dilations_rate=(1, 1), name="averagepooling2d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.averagepooling.averagepooling2d
        self.channel_first = channel_first
        self.layer_name = name
        self.mode = mode
        self.dilations_rate = dilations_rate
        self.argmax = argmax

    def call(self, inputs):
        out = self.op(inputs, pool_size=self.pool_size,
                      strides=self.strides,
                      alpha=self.alpha, beta=self.beta, name=self.layer_name,
                      dilations_rate=self.dilations_rate, channel_first=self.channel_first,
                      mode=self.mode)
        if inputs is not list:
            return out
        else:
            return tf.math.real(out), tf.math.imag(out)


class Averagepooling3d(tf.keras.layers.Layer):
    """ Averagepooling3d layer. """
    def __init__(self, pool_size=(2, 2, 2), strides=(2, 2, 2), alpha=1, beta=1, dilations_rate=(1, 1, 1),
                 channel_first=False, mode='VALID', name="averagepooling3d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.averagepooling.averagepooling3d
        self.layer_name = name
        self.dilations_rate = dilations_rate
        self.channel_first = channel_first
        self.mode = mode
        self.argmax = argmax

    def call(self, inputs):
        out = self.op(inputs, pool_size=self.pool_size,
                      strides=self.strides,
                      alpha=self.alpha, beta=self.beta, name=self.layer_name,
                      dilations_rate=self.dilations_rate, channel_first=self.channel_first,
                      mode=self.mode)
        if inputs is not list:
            return out
        else:
            return tf.math.real(out), tf.math.imag(out)


class Averagepooling4d(tf.keras.layers.Layer):
    """ Averagepooling4d layer. """
    def __init__(self, pool_size=(2, 2, 2, 2), strides=(2, 2, 2, 2), alpha=1, beta=1, dilations_rate=(1, 1, 1, 1),
                 channel_first=False, mode='VALID', name="averagepooling4d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.averagepooling.averagepooling4d
        self.layer_name = name
        self.dilations_rate = dilations_rate
        self.channel_first = channel_first
        self.mode = mode
        self.argmax = argmax

    def call(self, inputs):
        out = self.op(inputs, pool_size=self.pool_size,
                      strides=self.strides,
                      alpha=self.alpha, beta=self.beta, name=self.layer_name,
                      dilations_rate=self.dilations_rate, channel_first=self.channel_first,
                      mode=self.mode)
        if inputs is not list:
            return out
        else:
            return tf.math.real(out), tf.math.imag(out)





