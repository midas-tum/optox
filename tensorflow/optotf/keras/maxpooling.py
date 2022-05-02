import tensorflow as tf
import optotf.maxpooling
import numpy as np


class Maxpooling1d(tf.keras.layers.Layer):
    """ Maxpooling1d layer. """
    def __init__(self, pool_size=(2,), strides=(2,), alpha=1, beta=1, channel_first=False, mode='VALID',
                 dilations_rate=(1, 1), name="maxpooling1d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.maxpooling.maxpooling1d
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


class Maxpooling2d(tf.keras.layers.Layer):
    """ Maxpooling2d layer. """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), alpha=1, beta=1, channel_first=False, mode='VALID',
                 dilations_rate=(1, 1), name="maxpooling2d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.maxpooling.maxpooling2d
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


class Maxpooling3d(tf.keras.layers.Layer):
    """ Maxpooling3d layer. """
    def __init__(self, pool_size=(2, 2, 2), strides=(2, 2, 2), alpha=1, beta=1, dilations_rate=(1, 1, 1),
                 channel_first=False, mode='VALID', name="maxpooling3d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.maxpooling.maxpooling3d
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


class Maxpooling4d(tf.keras.layers.Layer):
    """ Maxpooling4d layer. """
    def __init__(self, pool_size=(2, 2, 2, 2), strides=(2, 2, 2, 2), alpha=1, beta=1, dilations_rate=(1, 1, 1, 1),
                 channel_first=False, mode='VALID', name="maxpooling3d", argmax=False):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.alpha = alpha
        self.beta = beta
        self.op = optotf.maxpooling.maxpooling4d
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








