import unittest
import tensorflow as tf
import optotf.warp
import numpy as np

class TestWarp(unittest.TestCase):            
    def _test_adjointness(self, dtype, mode):
        # setup the vaiables
        tf_x = tf.random.normal([10, 5, 20, 20,], dtype=dtype)
        tf_u = tf.random.normal([10, 20, 20, 2,],  dtype=dtype)*2
        tf_p = tf.random.normal([10, 5, 20, 20,], dtype=dtype)

        A = optotf.warp.warp_2d
        AH = optotf.warp.warp_2d_transpose

        tf_warp_x, _ = A(tf_x, tf_u, tf_x, mode=mode)
        tf_warpT_p, _ = AH(tf_p, tf_u, tf_x, mode=mode)

        lhs = tf.reduce_sum(tf_warp_x * tf_p)
        rhs = tf.reduce_sum(tf_x * tf_warpT_p)

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float_gradient(self):
        self._test_adjointness(tf.float32, 'replicate')

    def test_double_gradient(self):
        self._test_adjointness(tf.float64, 'replicate')

if __name__ == "__main__":
    unittest.main()
