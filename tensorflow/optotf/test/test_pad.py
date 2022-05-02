import unittest
import numpy as np
import tensorflow as tf
import optotf.pad

# to run execute: python -m unittest [-v] optotf.pad
class TestPadNd(unittest.TestCase):
    def _get_data(self, ndim, channel_last):
        tf_dtype = tf.float64
        N = 5
        W = 40
        H = 30
        D = 20
        C = 3

        pad_x = 4
        pad_y = 3
        pad_z = 2

        if ndim == 1:
            shape =  [W]
            shapeT = [W+2*pad_x]
            padding = [pad_x, pad_x]
        elif ndim == 2:
            shape =  [H, W]
            shapeT = [H+2*pad_y, W+2*pad_x]
            padding = [pad_x, pad_x, pad_y, pad_y]
        elif ndim == 3:
            shape = [D, H, W]
            shapeT = [D+2*pad_z, H+2*pad_y, W+2*pad_x]
            padding = [pad_x, pad_x, pad_y, pad_y, pad_z, pad_z]

        if channel_last:
            shape = [N,] + shape + [C,]
            shapeT = [N,] + shapeT + [C,]
        else:
            shape = [N, C,] + shape
            shapeT = [N, C,] + shapeT

        np_x = np.random.randn(*shape)
        np_y = np.random.randn(*shapeT)

        # transfer to tensorflow
        tf_x = tf.convert_to_tensor(np_x, tf_dtype)
        tf_y = tf.convert_to_tensor(np_y, tf_dtype)

        return tf_x, tf_y, padding

    def _test_adjointness(self, ndim, base_type, channel_last):
        # determine the operator
        A = optotf.pad._pad
        AH = optotf.pad._pad_transpose

        tf_x, tf_y, padding = self._get_data(ndim, channel_last)

        # perform fwd/adj
        tf_Ax = A(ndim,
                  tf_x, 
                  padding,
                  mode=base_type,
                  channel_last=channel_last)
        tf_AHy = AH(ndim,
                  tf_y,
                  padding,
                  mode=base_type,
                  channel_last=channel_last)

        # adjointness check
        lhs = tf.reduce_sum(tf_Ax * tf_y)
        rhs = tf.reduce_sum(tf_AHy * tf_x)
        
        print('adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-5)

    def test_1d_symmetric_adjointness_channelFirst(self):
        self._test_adjointness(1, "symmetric", False)

    def test_1d_reflect_adjointness_channelFirst(self):
        self._test_adjointness(1, "reflect", False)

    def test_1d_replicate_adjointness_channelFirst(self):
        self._test_adjointness(1, "replicate", False)

    def test_2d_symmetric_adjointness_channelFirst(self):
        self._test_adjointness(2, "symmetric", False)

    def test_2d_reflect_adjointness_channelFirst(self):
        self._test_adjointness(2, "reflect", False)

    def test_2d_replicate_adjointness_channelFirst(self):
        self._test_adjointness(2, "replicate", False)

    def test_3d_symmetric_adjointness_channelFirst(self):
        self._test_adjointness(3, "symmetric", False)

    def test_3d_reflect_adjointness_channelFirst(self):
        self._test_adjointness(3, "reflect", False)

    def test_3d_replicate_adjointness_channelFirst(self):
        self._test_adjointness(3, "replicate", False)

    def test_1d_symmetric_adjointness_channelLast(self):
        self._test_adjointness(1, "symmetric", True)

    def test_1d_reflect_adjointness_channelLast(self):
        self._test_adjointness(1, "reflect", True)

    def test_1d_replicate_adjointness_channelLast(self):
        self._test_adjointness(1, "replicate", True)

    def test_2d_symmetric_adjointness_channelLast(self):
        self._test_adjointness(2, "symmetric", True)

    def test_2d_reflect_adjointness_channelLast(self):
        self._test_adjointness(2, "reflect", True)

    def test_2d_replicate_adjointness_channelLast(self):
        self._test_adjointness(2, "replicate", True)

    def test_3d_symmetric_adjointness_channelLast(self):
        self._test_adjointness(3, "symmetric", True)

    def test_3d_reflect_adjointness_channelLast(self):
        self._test_adjointness(3, "reflect", True)

    def test_3d_replicate_adjointness_channelLast(self):
        self._test_adjointness(3, "replicate", True)

class TestComplexPadNd(TestPadNd):
    def _test_adjointness(self, ndim, base_type, channel_last):
        # determine the operator
        A = optotf.pad._pad
        AH = optotf.pad._pad_transpose

        tf_x_re, tf_y_re, padding = self._get_data(ndim, channel_last)
        tf_x_im, tf_y_im, _ = self._get_data(ndim, channel_last)
        tf_x = tf.complex(tf_x_re, tf_x_im)
        tf_y = tf.complex(tf_y_re, tf_y_im)

        # perform fwd/adj
        tf_Ax = A(ndim,
                  tf_x, 
                  padding,
                  mode=base_type,
                  channel_last=channel_last)
        tf_AHy = AH(ndim,
                  tf_y,
                  padding,
                  mode=base_type,
                  channel_last=channel_last)

        # adjointness check
        lhs = tf.reduce_sum(tf_Ax * tf.math.conj(tf_y))
        rhs = tf.reduce_sum(tf.math.conj(tf_AHy) * tf_x)
        
        print('adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-5)

    def test_1d_symmetric_adjointness_channelFirst(self):
        self._test_adjointness(1, "symmetric", False)

    def test_1d_reflect_adjointness_channelFirst(self):
        self._test_adjointness(1, "reflect", False)

    def test_1d_replicate_adjointness_channelFirst(self):
        self._test_adjointness(1, "replicate", False)

    def test_2d_symmetric_adjointness_channelFirst(self):
        self._test_adjointness(2, "symmetric", False)

    def test_2d_reflect_adjointness_channelFirst(self):
        self._test_adjointness(2, "reflect", False)

    def test_2d_replicate_adjointness_channelFirst(self):
        self._test_adjointness(2, "replicate", False)

    def test_3d_symmetric_adjointness_channelFirst(self):
        self._test_adjointness(3, "symmetric", False)

    def test_3d_reflect_adjointness_channelFirst(self):
        self._test_adjointness(3, "reflect", False)

    def test_3d_replicate_adjointness_channelFirst(self):
        self._test_adjointness(3, "replicate", False)

    def test_1d_symmetric_adjointness_channelLast(self):
        self._test_adjointness(1, "symmetric", True)

    def test_1d_reflect_adjointness_channelLast(self):
        self._test_adjointness(1, "reflect", True)

    def test_1d_replicate_adjointness_channelLast(self):
        self._test_adjointness(1, "replicate", True)

    def test_2d_symmetric_adjointness_channelLast(self):
        self._test_adjointness(2, "symmetric", True)

    def test_2d_reflect_adjointness_channelLast(self):
        self._test_adjointness(2, "reflect", True)

    def test_2d_replicate_adjointness_channelLast(self):
        self._test_adjointness(2, "replicate", True)

    def test_3d_symmetric_adjointness_channelLast(self):
        self._test_adjointness(3, "symmetric", True)

    def test_3d_reflect_adjointness_channelLast(self):
        self._test_adjointness(3, "reflect", True)

    def test_3d_replicate_adjointness_channelLast(self):
        self._test_adjointness(3, "replicate", True)

if __name__ == "__main__":
    unittest.main()
