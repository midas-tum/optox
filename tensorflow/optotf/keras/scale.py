import tensorflow as tf
import optotf.scale

class Scale(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.op = lambda x: optotf.scale.scale(x)

    def call(self, x):
        return self.op(x)