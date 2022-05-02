import tensorflow as tf
import optotf.pad

class _PadNd(tf.keras.layers.Layer):
    def __init__(self, ndim, padding, mode, channel_last=True):
        super().__init__()
        self.ndim = ndim
        self.padding = padding
        self.mode = mode
        self.channel_last = channel_last

    def build(self, input_shape):
        shape = tf.unstack(input_shape)

        if self.channel_last:
            shape = [shape[0], shape[-1], *shape[1:-1]]

        new_shape = [-1, *shape[2:]]
        new_shape = tf.stack(new_shape)

        padded_shape = shape
        for i in range(self.ndim):
            padded_shape[-1-i] += self.padding[i*2] + self.padding[i*2+1]
        padded_shape = tf.stack(padded_shape)

        self.pre_pad_shape = new_shape
        self.post_pad_shape = padded_shape        

    def call(self, x):
        # first reshape the input
        if self.channel_last:
            axes = [0, self.ndim+1, *range(1, self.ndim+1)]
            x = tf.transpose(x, axes)

        x_r = tf.reshape(x, self.pre_pad_shape)

        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            x_r = tf.complex(self.op(tf.math.real(x_r), self.mode, *self.padding), 
                             self.op(tf.math.imag(x_r), self.mode, *self.padding))
        else:
            x_r = self.op(x_r, self.mode, *self.padding)

        if self.channel_last:
            axes = [0, *range(2, self.ndim+2), 1]
            return tf.transpose(tf.reshape(x_r, self.post_pad_shape), axes)
        else:
            return tf.reshape(x_r, self.post_pad_shape)

class _PadNdTranspose(tf.keras.layers.Layer):
    def __init__(self, ndim, padding, mode, channel_last=True):
        super().__init__()
        self.ndim = ndim
        self.padding = padding
        self.mode = mode
        self.channel_last = channel_last

    def build(self, input_shape):
        shape = tf.unstack(input_shape)

        if self.channel_last:
            shape = [shape[0], shape[-1], *shape[1:-1]]

        new_shape = [-1, *shape[2:]]
        new_shape = tf.stack(new_shape)

        padded_shape = shape
        for i in range(self.ndim):
            padded_shape[-1-i] -= self.padding[i*2] + self.padding[i*2+1]
        padded_shape = tf.stack(padded_shape)

        self.pre_pad_shape = new_shape
        self.post_pad_shape = padded_shape        

    def call(self, x):
        # first reshape the input
        if self.channel_last:
            axes = [0, self.ndim+1, *range(1, self.ndim+1)]
            x = tf.transpose(x, axes)

        x_r = tf.reshape(x, self.pre_pad_shape)

        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            x_r = tf.complex(self.op(tf.math.real(x_r), self.mode, *self.padding), 
                             self.op(tf.math.imag(x_r), self.mode, *self.padding))
        else:
            x_r = self.op(x_r, self.mode, *self.padding)

        if self.channel_last:
            axes = [0, *range(2, self.ndim+2), 1]
            return tf.transpose(tf.reshape(x_r, self.post_pad_shape), axes)
        else:
            return tf.reshape(x_r, self.post_pad_shape)

class Pad1d(_PadNd):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(1, padding, mode, channel_last)
        self.op = optotf.pad._ext.pad1d

class Pad2d(_PadNd):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(2, padding, mode, channel_last)
        self.op = optotf.pad._ext.pad2d

class Pad3d(_PadNd):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(3, padding, mode, channel_last)
        self.op = optotf.pad._ext.pad3d

class Pad1dTranspose(_PadNdTranspose):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(1, padding, mode, channel_last)
        self.op = optotf.pad._ext.pad1d_transpose

class Pad2dTranspose(_PadNdTranspose):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(2, padding, mode, channel_last)
        self.op = optotf.pad._ext.pad2d_transpose

class Pad3dTranspose(_PadNdTranspose):
    def __init__(self, padding, mode, channel_last=True):
        super().__init__(3, padding, mode, channel_last)
        self.op = optotf.pad._ext.pad3d_transpose
