import numpy as np
import torch
import unittest
import optoth.maxpooling


def padding_shape(input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
    if padding_mode.lower() == 'valid':
        return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
    elif padding_mode.lower() == 'same':
        return np.ceil(input_spatial_shape / strides)
    else:
        raise Exception('padding_mode can be only valid or same!')


class TestMaxpooling1dFunction(unittest.TestCase):
    def _run_test(self, dtype, padding_mode):
        # test maxpooling1d and maxpooling1d_backward
        pool_size = (2,)
        strides = (2,)
        alpha = 1
        beta = 1
        dilations_rate = (1,)
        pads = (0,)
        N = 2
        H = 10
        C = 2
        xr = np.zeros((N, H, C), dtype=dtype)
        xi = np.zeros((N, H, C), dtype=dtype)
        for n in range(N):
            for i in range(H):
                for k in range(C):
                    xr[n, i, k], xi[n, i, k] = i, i

        t_real = torch.from_numpy(xr)
        t_imag = torch.from_numpy(xi)
        x = torch.complex(t_real, t_imag)
        cuda1 = torch.device('cuda:0')
        x = x.to(device=cuda1)
        x = x.requires_grad_()
        t_dtype = None
        if dtype == np.float32:
            t_dtype = torch.float32
        elif dtype == np.float64:
            t_dtype = torch.float64

        out_complex = optoth.maxpooling.Maxpooling1dFunction.apply(x, pool_size, pads, strides, dilations_rate, alpha,
                                                                     beta, padding_mode,
                                                                     t_dtype, False)

        out_complex.sum().backward()
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         C]) - np.array(
            out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         C]) - np.array(
            x.grad.shape)).all() < 1e-8)

    def test_float32_valid(self):
        self._run_test(np.float32, 'valid')

    def test_float32_same(self):
        self._run_test(np.float32, 'same')

    def test_float64_valid(self):
        self._run_test(np.float64, 'valid')

    def test_float64_same(self):
        self._run_test(np.float64, 'same')


class TestMaxpooling2dFunction(unittest.TestCase):
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
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    for k in range(C):
                        xr[n, i, j, k], xi[n, i, j, k] = i * H + j, i * H + j
        t_real = torch.from_numpy(xr)
        t_imag = torch.from_numpy(xi)
        x = torch.complex(t_real, t_imag)
        cuda1 = torch.device('cuda:0')
        x = x.to(device=cuda1)
        x = x.requires_grad_()
        t_dtype = None
        if dtype == np.float32:
            t_dtype = torch.float32
        elif dtype == np.float64:
            t_dtype = torch.float64

        out_complex = optoth.maxpooling.Maxpooling2dFunction.apply(x, pool_size, pads, strides, dilations_rate, alpha,
                                                                     beta, padding_mode,
                                                                     t_dtype, False)

        out_complex.sum().backward()
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         C]) - np.array(
            out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         C]) - np.array(
            x.grad.shape)).all() < 1e-8)

    def test_float32_valid(self):
        self._run_test(np.float32, 'valid')

    def test_float32_same(self):
        self._run_test(np.float32, 'same')

    def test_float64_valid(self):
        self._run_test(np.float64, 'valid')

    def test_float64_same(self):
        self._run_test(np.float64, 'same')


class TestMaxpooling3dFunction(unittest.TestCase):
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
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    for k in range(D):
                        for l in range(C):
                            xr[n, i, j, k, l], xi[n, i, j, k, l] = i * H * W + j * W + k, i * H * W + j * W + k
        t_real = torch.from_numpy(xr)
        t_imag = torch.from_numpy(xi)
        x = torch.complex(t_real, t_imag)
        cuda1 = torch.device('cuda:0')
        x = x.to(device=cuda1)
        x = x.requires_grad_()
        t_dtype = None
        if dtype == np.float32:
            t_dtype = torch.float32
        elif dtype == np.float64:
            t_dtype = torch.float64

        out_complex = optoth.maxpooling.Maxpooling3dFunction.apply(x, pool_size, pads, strides, dilations_rate, alpha,
                                                                     beta, padding_mode,
                                                                     t_dtype, False)

        out_complex.sum().backward()
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(D, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         C]) - np.array(
            out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array([N, padding_shape(H, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(W, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(D, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         C]) - np.array(
            x.grad.shape)).all() < 1e-8)

    def test_float32_valid(self):
        self._run_test(np.float32, 'valid')

    def test_float32_same(self):
        self._run_test(np.float32, 'same')

    def test_float64_valid(self):
        self._run_test(np.float64, 'valid')

    def test_float64_same(self):
        self._run_test(np.float64, 'same')


class TestMaxpooling4dFunction(unittest.TestCase):
    def _run_test(self, dtype, padding_mode):
        # test maxpooling4d and maxpooling4d_backward
        pool_size = (2, 2, 2, 2)
        strides = (2, 2, 2, 2)
        alpha = 1
        beta = 1

        dilations_rate = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)

        N = 1
        T = 4
        H = 6
        W = 6
        D = 6
        C = 2
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
        t_real = torch.from_numpy(xr)
        t_imag = torch.from_numpy(xi)
        x = torch.complex(t_real, t_imag)
        cuda1 = torch.device('cuda:0')
        x = x.to(device=cuda1)
        x = x.requires_grad_()
        t_dtype = None
        if dtype == np.float32:
            t_dtype = torch.float32
        elif dtype == np.float64:
            t_dtype = torch.float64

        out_complex = optoth.maxpooling.Maxpooling4dFunction.apply(x, pool_size, pads, strides, dilations_rate, alpha,
                                                                     beta, padding_mode,
                                                                     t_dtype, False)

        out_complex.sum().backward()
        self.assertTrue(np.abs(np.array([N, padding_shape(T, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(H, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(W, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         padding_shape(D, pool_size[3], strides[3], dilations_rate[3], padding_mode),
                                         C]) - np.array(
            out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array([N, padding_shape(T, pool_size[0], strides[0], dilations_rate[0], padding_mode),
                                         padding_shape(H, pool_size[1], strides[1], dilations_rate[1], padding_mode),
                                         padding_shape(W, pool_size[2], strides[2], dilations_rate[2], padding_mode),
                                         padding_shape(D, pool_size[3], strides[3], dilations_rate[3], padding_mode),
                                         C]) - np.array(
            x.grad.shape)).all() < 1e-8)

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
