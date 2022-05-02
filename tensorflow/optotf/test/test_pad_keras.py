"""
import unittest
import tensorflow as tf
import optotf.keras.pad

class TestPad(unittest.TestCase):
    def test1d(self):
        shape = (5, 2, 10)
        x = tf.random.normal(shape)
        padding = [2, 2,]
        op = optotf.keras.pad.Pad1d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test1d_complex(self):
        shape = (5, 2, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2,]
        op = optotf.keras.pad.Pad1d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test1d_channel_last(self):
        shape = (5, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2,]
        op = optotf.keras.pad.Pad1d(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] += padding[0] + padding[1]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d(self):
        shape = (5, 2, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = optotf.keras.pad.Pad2d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_complex(self):
        shape = (5, 2, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4]
        op = optotf.keras.pad.Pad2d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_channel_last(self):
        shape = (5, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = optotf.keras.pad.Pad2d(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] += padding[0] + padding[1]
        new_shape[-3] += padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 1]
        op = optotf.keras.pad.Pad3d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape[-3] += padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_complex(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4, 1, 1]
        op = optotf.keras.pad.Pad3d(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] += padding[0] + padding[1]
        new_shape[-2] += padding[2] + padding[3]
        new_shape[-3] += padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_channel_last(self):
        shape = (5, 8, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 2]
        op = optotf.keras.pad.Pad3d(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] += padding[0] + padding[1]
        new_shape[-3] += padding[2] + padding[3]
        new_shape[-4] += padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test1d_transpose(self):
        shape = (5, 2, 10)
        x = tf.random.normal(shape)
        padding = [2, 2,]
        op = optotf.keras.pad.Pad1dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test1d_complex_transpose(self):
        shape = (5, 2, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2,]
        op = optotf.keras.pad.Pad1dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test1d_channel_last_transpose(self):
        shape = (5, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2,]
        op = optotf.keras.pad.Pad1dTranspose(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] -= padding[0] + padding[1]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_transpose(self):
        shape = (5, 2, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = optotf.keras.pad.Pad2dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_complex_transpose(self):
        shape = (5, 2, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4]
        op = optotf.keras.pad.Pad2dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test2d_channel_last_transpose(self):
        shape = (5, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4]
        op = optotf.keras.pad.Pad2dTranspose(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] -= padding[0] + padding[1]
        new_shape[-3] -= padding[2] + padding[3]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_transpose(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 1]
        op = optotf.keras.pad.Pad3dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape[-3] -= padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_complex_transpose(self):
        shape = (5, 2, 8, 10, 10)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        padding = [2, 2, 4, 4, 1, 1]
        op = optotf.keras.pad.Pad3dTranspose(padding=padding, mode='symmetric', channel_last=False)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-1] -= padding[0] + padding[1]
        new_shape[-2] -= padding[2] + padding[3]
        new_shape[-3] -= padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

    def test3d_channel_last_transpose(self):
        shape = (5, 8, 10, 10, 2)
        x = tf.random.normal(shape)
        padding = [2, 2, 4, 4, 1, 2]
        op = optotf.keras.pad.Pad3dTranspose(padding=padding, mode='symmetric', channel_last=True)
        Kx = op(x)

        # manually construct new shape
        new_shape = list(x.shape)
        new_shape[-2] -= padding[0] + padding[1]
        new_shape[-3] -= padding[2] + padding[3]
        new_shape[-4] -= padding[4] + padding[5]
        new_shape = tuple(new_shape)

        self.assertTrue(new_shape == Kx.shape)

if __name__ == "__main__":
    unittest.main()
    """
