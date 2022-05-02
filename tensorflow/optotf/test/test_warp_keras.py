import unittest
import tensorflow as tf
import optotf.keras.warp

class TestWarp(unittest.TestCase):
    def test_warp_forward_channelfirst(self):
        x = tf.random.normal((10, 2, 20, 20))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = optotf.keras.warp.Warp(channel_last=False)
        Kx = op(x, u)

    def test_warp_transpose_channelfirst(self):
        grad = tf.random.normal((10, 2, 20, 20))
        x = tf.random.normal((10, 2, 20, 20))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = optotf.keras.warp.WarpTranspose(channel_last=False)
        Kx = op(grad, u, x)

    def test_warp_forward(self):
        x = tf.random.normal((10, 20, 20, 2))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = optotf.keras.warp.Warp(channel_last=True)
        Kx = op(x, u)

    def test_warp_transpose(self):
        grad = tf.random.normal((10, 20, 20, 2))
        x = tf.random.normal((10, 20, 20, 2))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = optotf.keras.warp.WarpTranspose(channel_last=True)
        Kx = op(grad, u, x)

    def test_warp_forward_complex(self):
        x = tf.complex(tf.random.normal((10, 20, 20, 2)), tf.random.normal((10, 20, 20, 2)))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = optotf.keras.warp.Warp(channel_last=True)
        Kx = op(x, u)

    def test_warp_transpose_complex(self):
        grad = tf.complex(tf.random.normal((10, 20, 20, 2)), tf.random.normal((10, 20, 20, 2)))
        x = tf.complex(tf.random.normal((10, 20, 20, 2)), tf.random.normal((10, 20, 20, 2)))
        u = tf.random.normal((10, 20, 20, 2))*10.0
        op = optotf.keras.warp.WarpTranspose(channel_last=True)
        Kx = op(grad, u, x)

if __name__ == "__main__":
    unittest.main()
