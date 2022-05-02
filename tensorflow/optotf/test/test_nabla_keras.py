import unittest
import tensorflow as tf
import optotf.keras.nabla

class TestNabla(unittest.TestCase):
    def test2d(self):
        x = tf.random.normal((10, 10))
        op = optotf.keras.nabla.Nabla2d()
        Kx = op(x)
        self.assertTrue((2, *x.shape) == Kx.shape)

    def test2d_complex(self):
        x = tf.complex(tf.random.normal((10, 10)),
                       tf.random.normal((10, 10)))
        op = optotf.keras.nabla.Nabla2d()
        Kx = op(x)
        self.assertTrue((2, *x.shape) == Kx.shape)

    def test2d_adjoint(self):
        x = tf.random.normal((2, 10, 10))
        op = optotf.keras.nabla.NablaT2d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test2d_adjoint_complex(self):
        x = tf.complex(tf.random.normal((2, 10, 10)),
                       tf.random.normal((2, 10, 10)))
        op = optotf.keras.nabla.NablaT2d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test3d(self):
        x = tf.random.normal((10, 10, 10))
        op = optotf.keras.nabla.Nabla3d()
        Kx = op(x)
        self.assertTrue((3, *x.shape) == Kx.shape)

    def test3d_complex(self):
        x = tf.complex(tf.random.normal((10, 10, 10)),
                       tf.random.normal((10, 10, 10)))
        op = optotf.keras.nabla.Nabla3d()
        Kx = op(x)
        self.assertTrue((3, *x.shape) == Kx.shape)

    def test3d_adjoint(self):
        x = tf.random.normal((3, 10, 10, 10))
        op = optotf.keras.nabla.NablaT3d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test3d_adjoint_complex(self):
        x = tf.complex(tf.random.normal((3, 10, 10, 10)),
                       tf.random.normal((3, 10, 10, 10)))
        op = optotf.keras.nabla.NablaT3d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test4d(self):
        x = tf.random.normal((10, 10, 10, 10))
        op = optotf.keras.nabla.Nabla4d()
        Kx = op(x)
        self.assertTrue((4, *x.shape) == Kx.shape)

    def test4d_complex(self):
        x = tf.complex(tf.random.normal((10, 10, 10, 10)),
                       tf.random.normal((10, 10, 10, 10)))
        op = optotf.keras.nabla.Nabla4d()
        Kx = op(x)
        self.assertTrue((4, *x.shape) == Kx.shape)

    def test4d_adjoint(self):
        x = tf.random.normal((4, 10, 10, 10, 10))
        op = optotf.keras.nabla.NablaT4d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

    def test4d_adjoint_complex(self):
        x = tf.complex(tf.random.normal((4, 10, 10, 10, 10)),
                       tf.random.normal((4, 10, 10, 10, 10)))
        op = optotf.keras.nabla.NablaT4d()
        Kx = op(x)
        self.assertTrue(x.shape[1:] == Kx.shape)

if __name__ == "__main__":
    unittest.main()
