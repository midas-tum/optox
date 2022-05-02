import unittest
import tensorflow as tf
import numpy as np
import optotf.keras.maxpooling


def padding_shape(input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
    if padding_mode.lower() == 'valid':
        return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
    elif padding_mode.lower() == 'same':
        return np.ceil(input_spatial_shape / strides)
    else:
        raise Exception('padding_mode can be only valid or same!')


class TestFunction1D(unittest.TestCase):

    def _create_1d_inputs(self, dtype, N=1, H=6, C=2):
        xr = np.zeros((N, H, C), dtype=dtype)
        xi = np.zeros((N, H, C), dtype=dtype)
        for n in range(N):
            for i in range(H):
                for k in range(C):
                    xr[n, i, k], xi[n, i, k] = i, i

        return xr, xi

    def _test_1d_maxpooling_complex(self, pool_size, strides, alpha, beta, padding_mode, dilations_rate, dtype, N=1,
                                    H=10,  C=2):
        xr, xi = self._create_1d_inputs(dtype, N, H, C)
        model = optotf.keras.maxpooling.Maxpooling1d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                 mode=padding_mode, dilations_rate=dilations_rate)
        # test type tf.complex, padding
        inputs = tf.complex(xr, xi)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            out_complex = model(inputs)
            gradients = tape.gradient(tf.math.reduce_sum(out_complex), inputs)

        # Check max pooling 2D shape by complex input
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         C]) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(inputs.shape) - np.array(gradients.shape)).all() < 1e-8)

        return out_complex

    def test_1d(self):
        pool_size = (2, )
        strides = (2, )
        alpha = 1
        beta = 1
        dilations_rate = (1, )

        # complex-valued
        padding_mode = 'VALID'
        self._test_1d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_1d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)

        padding_mode = 'SAME'
        self._test_1d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_1d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)


class TestFunction2D(unittest.TestCase):
    def _create_2d_inputs(self, dtype, N=1, H=10, W=10, C=2):
        xr = np.zeros((N, H, W, C), dtype=dtype)
        xi = np.zeros((N, H, W, C), dtype=dtype)
        for i in range(H):
            for j in range(W):
                xr[N - 1, i, j, 0] = (i * H + j)
                for k in range(C):
                    xr[N - 1, i, j, k] = xr[N - 1, i, j, 0]

        for i in range(H):
            for j in range(W):
                xi[N - 1, i, j, 0] = (i * H + j)
                for k in range(C):
                    xi[N - 1, i, j, k] = xi[N - 1, i, j, 0]

        return xr, xi

    def _test_2d_maxpooling_complex(self, pool_size, strides, alpha, beta, padding_mode, dilations_rate, dtype, N=1,
                                    H=10, W=10, C=2):
        xr, xi = self._create_2d_inputs(dtype, N, H, W, C)
        model = optotf.keras.maxpooling.Maxpooling2d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                 mode=padding_mode, dilations_rate=dilations_rate)
        # test type tf.complex, padding
        inputs = tf.complex(xr, xi)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            out_complex = model(inputs)
            gradients = tape.gradient(tf.math.reduce_sum(out_complex), inputs)

        # Check max pooling 2D shape by complex input
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         C]) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(inputs.shape) - np.array(gradients.shape)).all() < 1e-8)

        return out_complex

    def test_2d(self):
        pool_size = (2, 2)
        strides = (2, 2)
        alpha = 1
        beta = 1
        dilations_rate = (1, 1)

        # complex-valued
        padding_mode = 'VALID'
        self._test_2d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_2d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)

        padding_mode = 'SAME'
        self._test_2d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_2d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)


class TestFunction3D(unittest.TestCase):
    def _create_3d_inputs(self, dtype, N=1, H=6, W=6, D=6, C=2):
        xr = np.zeros((N, H, W, D, C), dtype=dtype)
        xi = np.zeros((N, H, W, D, C), dtype=dtype)
        for i in range(H):
            for j in range(W):
                for k in range(D):
                    xr[N - 1, i, j, k, 0] = (i * H * W + j * W + k)
                    for l in range(C):
                        xr[N - 1, i, j, k, l] = xr[N - 1, i, j, k, 0]

        for i in range(H):
            for j in range(W):
                for k in range(D):
                    xi[N - 1, i, j, k, 0] = (i * H * W + j * W + k)
                    for l in range(C):
                        xi[N - 1, i, j, k, l] = xi[N - 1, i, j, k, 0]
        return xr, xi

    def _test_3d_maxpooling_complex(self, pool_size, strides, alpha, beta, padding_mode, dilations_rate, dtype, N=1,
                                    H=10, W=10, D=6, C=2):
        xr, xi = self._create_3d_inputs(dtype, N, H, W, D, C)
        model = optotf.keras.maxpooling.Maxpooling3d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                 mode=padding_mode, dilations_rate=dilations_rate)
        # test type tf.complex, padding
        inputs = tf.complex(xr, xi)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            out_complex = model(inputs)
            gradients = tape.gradient(tf.math.reduce_sum(out_complex), inputs)

        # Check max pooling 2D shape by complex input
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(D, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         C]) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(inputs.shape) - np.array(gradients.shape)).all() < 1e-8)

        return out_complex

    def test_3d(self):
        pool_size = (2, 2, 2)
        strides = (2, 2, 2)
        alpha = 1
        beta = 1
        dilations_rate = (1, 1, 1)

        # complex-valued
        padding_mode = 'VALID'
        self._test_3d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_3d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)

        padding_mode = 'SAME'
        self._test_3d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_3d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)


class TestFunction4D(unittest.TestCase):
    def _create_4d_inputs(self, dtype, N=1, T=4, H=6, W=6, D=6, C=2):
        xr = np.zeros((N, T, H, W, D, C), dtype=dtype)
        xi = np.zeros((N, T, H, W, D, C), dtype=dtype)
        for n in range(N):
            for t in range(T):
                for i in range(H):
                    for j in range(W):
                        for k in range(D):
                            for l in range(C):
                                xr[n, t, i, j, k, l], xi[n, t, i, j, k, l] = t * T * H * W + i * H * W + j * W + k, \
                                                                             t * T * H * W + i * H * W + j * W + k

        return xr, xi

    def _test_4d_maxpooling_complex(self, pool_size, strides, alpha, beta, padding_mode, dilations_rate, dtype, N=1,
                                    T= 4,  H=6, W=6, D=6, C=2):
        xr, xi = self._create_4d_inputs(dtype, N, T, H, W, D, C)
        model = optotf.keras.maxpooling.Maxpooling4d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                 mode=padding_mode, dilations_rate=dilations_rate)
        # test type tf.complex,
        inputs = tf.complex(xr, xi)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            out_complex = model(inputs)
            gradients = tape.gradient(tf.math.reduce_sum(out_complex), inputs)

        # Check max pooling 4D shape by complex input
        self.assertTrue(np.abs(np.array([N, padding_shape(T, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(H, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(W, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         padding_shape(D, pool_size[3], strides[3], dilations_rate[3], padding_mode),
                                         C]) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(inputs.shape) - np.array(gradients.shape)).all() < 1e-8)

        return out_complex

    def test_4d(self):
        pool_size = (2, 2, 2, 2)
        strides = (2, 2, 2, 2)
        alpha = 1
        beta = 1
        dilations_rate = (1, 1, 1, 1)

        # complex-valued
        padding_mode = 'VALID'
        self._test_4d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_4d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)

        padding_mode = 'SAME'
        self._test_4d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float32)
        self._test_4d_maxpooling_complex(pool_size, strides, alpha, beta, padding_mode, dilations_rate, np.float64)


if __name__ == "__main__":
    unittest.main()
