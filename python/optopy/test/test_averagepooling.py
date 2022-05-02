import unittest
import numpy as np
import optopy.averagepooling


def padding_shape(input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
    if padding_mode.lower() == 'valid':
        return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
    elif padding_mode.lower() == 'same':
        return np.ceil(input_spatial_shape / strides)
    else:
        raise Exception('padding_mode can be only valid or same!')


# to run execute: python -m unittest [-v] optopy.maxpooling
class TestAveragepooling2d(unittest.TestCase):
    def _run_test(self, dtype, padding_mode):
        # test maxpooling2d and maxpooling2d_backward
        pool_size = (2, 2)
        strides = (2, 2)
        alpha = 1
        beta = 1
        dilations_rate = (1, 1)
        pads = (0, 0)
        N = 1
        H = 10
        W = 10
        C = 2
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

        x = xr + 1j * xi

        op = optopy. averagepooling.Averagepooling2d(pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode,
                                              dtype)
        out_complex = op.forward(x)
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         C]) - np.array(
            out_complex.shape)).all() < 1e-8)

        xr = np.ones(out_complex.shape, dtype=dtype) * -1
        xi = np.ones(out_complex.shape, dtype=dtype) * -1
        x_diff = xr + 1j * xi

        back_out = op.backward(x, out_complex, x_diff)
        back_out_indices = op.backward(x, out_complex, x_diff)

        self.assertTrue(np.abs(back_out_indices - back_out).all() < 1e-8)

        # Check max pooling 2D shape by list input
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         C]) - np.array(
            back_out.shape)).all() < 1e-8)

    def test_float32_valid(self):
        self._run_test(np.float32, 'valid')

    def test_float32_same(self):
        self._run_test(np.float32, 'same')

    def test_float64_valid(self):
        self._run_test(np.float64, 'valid')

    def test_float64_same(self):
        self._run_test(np.float64, 'same')


class TestAveragepooling3d(unittest.TestCase):
    def _run_test(self, dtype, padding_mode):
        # test maxpooling3d and maxpooling3d_backward
        pool_size = (2, 2, 2)
        strides = (2, 2, 2)
        alpha = 1
        beta = 1

        dilations_rate = (1, 1, 1)
        pads = (0, 0, 0)

        N = 1
        H = 6
        W = 6
        D = 6
        C = 2

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
        x = xr + 1j * xi

        op = optopy. averagepooling.Averagepooling3d(pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode,
                                              dtype,  ceil_mode=True)
        out_complex= op.forward(x)
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(D, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         C]) - np.array(
            out_complex.shape)).all() < 1e-8)

        xr = np.ones(out_complex.shape, dtype=dtype) * -1
        xi = np.ones(out_complex.shape, dtype=dtype) * -1
        x_diff = xr + 1j * xi
        back_out = op.backward(x, out_complex, x_diff)

        back_out_indices = op.backward(x, out_complex, x_diff)

        self.assertTrue(np.abs(back_out_indices - back_out).all() < 1e-8)

        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(D, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         C]) - np.array(
            back_out.shape)).all() < 1e-8)

    def test_float32_valid(self):
        self._run_test(np.float32, 'valid')

    def test_float32_same(self):
        self._run_test(np.float32, 'same')

    def test_float64_valid(self):
        self._run_test(np.float64, 'valid')

    def test_float64_same(self):
        self._run_test(np.float64, 'same')


if __name__ == "__main__":
    unittest.main()
