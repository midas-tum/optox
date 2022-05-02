import tensorflow as tf
import optotf.nabla

class Nabla2d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_2d(x, hx=hx, hy=hy)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class Nabla3d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_3d(x, hx=hx, hy=hy, hz=hz)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class Nabla4d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1, ht=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_4d(x, hx=hx, hy=hy, hz=hz, ht=ht)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)


class NablaT2d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_2d_adjoint(x, hx=hx, hy=hy)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class NablaT3d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_3d_adjoint(x, hx=hx, hy=hy, hz=hz)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)

class NablaT4d(tf.keras.layers.Layer):
    def __init__(self, hx=1, hy=1, hz=1, ht=1):
        super().__init__()
        self.op = lambda x: optotf.nabla.nabla_4d_adjoint(x, hx=hx, hy=hy, hz=hz, ht=ht)

    def call(self, x):
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            return tf.complex(self.op(tf.math.real(x)), 
                              self.op(tf.math.imag(x)))
        else:
            return self.op(x)
