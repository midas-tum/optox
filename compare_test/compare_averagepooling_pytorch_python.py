import torch
import optopy.averagepooling
import optoth.averagepooling
import numpy as np
import unittest

cuda1 = torch.device('cuda:0')
def compare_2d(xr, xi, pool_size, strides, alpha, beta, dilations_rate, pads, mode):
    t_real = torch.from_numpy(xr)
    t_imag = torch.from_numpy(xi)
    x_torch = torch.complex(t_real, t_imag)
    x_torch = x_torch.to(device=cuda1)
    x_python= np.array(xr + 1j * xi)

    out_torch = optoth.averagepooling.averagepooling2d(x_torch, pool_size, pads, strides, dilations_rate, alpha,
                                                        beta, padding_mode=mode)
    out_python = optopy.averagepooling.Averagepooling2d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                                        padding_mode=mode,
                                                        dilations_rate=dilations_rate).forward(x_python)
    out_th = out_torch.cpu().detach().numpy()

    return out_python, out_th, out_torch, x_torch


def compare_2d_backward(x_torch, out_python, out_torch, xr, xi, pool_size, strides, alpha, beta, dilations_rate,
                        pads, mode):
    out_real = np.real(out_python)
    out_imag = np.imag(out_python)
    out_torch = out_torch.to(device=cuda1)

    x_diff_real = np.ones(out_real.shape).astype(np.float32) * -1
    x_diff_imag = np.ones(out_imag.shape).astype(np.float32) * -1
    x_python = xr+1j*xi
    x_diff = x_diff_real+1j*x_diff_imag
    out_python = optopy.averagepooling.Averagepooling2d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                                        padding_mode=mode,
                                                        dilations_rate=dilations_rate).backward(x_python,out_python,x_diff)
    t_real = torch.from_numpy(x_diff_real)
    t_imag = torch.from_numpy(x_diff_imag)
    x_diff = torch.complex(t_real, t_imag)
    x_diff = x_diff.to(device=cuda1)

    back_out_torch = optoth.averagepooling.averagepooling2d_backward(x_torch, out_torch, x_diff, pool_size, pads,
                                                                      strides, dilations_rate, alpha, beta,
                                                                      padding_mode=mode)


    out_th = back_out_torch.cpu().detach().numpy()
    return out_python, out_th, out_torch


def compare_3d(xr, xi, pool_size, strides, alpha, beta, dilations_rate, pads, mode):
    t_real = torch.from_numpy(xr)
    t_imag = torch.from_numpy(xi)
    x_torch = torch.complex(t_real, t_imag)
    x_torch = x_torch.to(device=cuda1)
    x_python = xr+1j*xi
    out_torch = optoth.averagepooling.averagepooling3d(x_torch, pool_size, pads, strides, dilations_rate, alpha,
                                                        beta, padding_mode=mode)
    out_python= optopy.averagepooling.Averagepooling3d(pool_size=pool_size,
                                                        strides=strides, alpha=alpha, beta=beta,
                                                        padding_mode=mode, dilations_rate=dilations_rate).forward(x_python)
    out_th = out_torch.cpu().detach().numpy()
    return out_python, out_th, out_torch, x_torch


def compare_3d_backward(x_torch, out_python, out_torch, xr, xi, pool_size, strides, alpha, beta, dilations_rate,
                        pads, mode):
    out_real = np.real(out_python)
    out_imag = np.imag(out_python)
    out_torch = out_torch.to(device=cuda1)

    x_diff_real = np.ones(out_real.shape).astype(np.float32) * -1
    x_diff_imag = np.ones(out_imag.shape).astype(np.float32) * -1
    x_python = xr+1j*xi
    x_diff = x_diff_real+1j*x_diff_imag
    out_python= optopy.averagepooling.Averagepooling3d(pool_size=pool_size, strides=strides, alpha=alpha, beta=beta,
                                                        padding_mode=mode,
                                                        dilations_rate=dilations_rate).backward(x_python,out_python,x_diff)
    t_real = torch.from_numpy(x_diff_real)
    t_imag = torch.from_numpy(x_diff_imag)
    x_diff = torch.complex(t_real, t_imag)
    x_diff = x_diff.to(device=cuda1)

    back_out_torch = optoth.averagepooling.averagepooling3d_backward(x_torch, out_torch, x_diff, pool_size, pads,
                                                                      strides, dilations_rate, alpha, beta,
                                                                      padding_mode=mode)

    out_th = back_out_torch.cpu().detach().numpy()
    return out_python, out_th, out_torch


class TestFunction2d(unittest.TestCase):
    def test_2d(self):
        self._test_2d_averagepooling('VALID')
        self._test_2d_averagepooling('SAME')

    def _test_2d_averagepooling(self, mode, pool_size=(2, 2), strides=(2, 2), alpha=1, beta=1, dilations_rate=(1, 1),
                               pads=(0, 0)):
        dtype = np.float32
        #pool_size = (2, 2)
        #strides = (2, 2)
        #alpha = 1
        #beta = 1
        #dilations_rate = (1, 1)
        #pads = (0, 0)
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

        out_python, out_th, out_torch, x_torch = compare_2d(xr, xi, pool_size, strides, alpha, beta,
                                                                        dilations_rate, pads, mode)
        self.assertTrue(np.abs(out_python - out_th).all() < 1e-16)

        out_python, out_th,  out_torch = compare_2d_backward(x_torch, out_python, out_torch, xr, xi, pool_size,
                                                                        strides, alpha, beta, dilations_rate, pads,
                                                                        mode)
        self.assertTrue(np.abs(out_python - out_th).all() < 1e-16)


class TestFunction3d(unittest.TestCase):
    def test_3d(self):
        self._test_3d_averagepooling('VALID')
        self._test_3d_averagepooling('SAME')

    def _test_3d_averagepooling(self, mode, pool_size=(2, 2, 2), strides=(2, 2, 2), alpha=1, beta=1,
                                dilations_rate=(1, 1, 1), pads=(0, 0, 0)):
        dtype = np.float32
        #pool_size = (2, 2, 2)
        #strides = (2, 2, 2)
        #pads = (0, 0, 0)
        #alpha = 1
        #beta = 1
        #dilations_rate = (1, 1, 1)
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

        out_python, out_th, out_torch, x_torch = compare_3d(xr, xi, pool_size, strides, alpha, beta,
                                                                        dilations_rate, pads, mode)
        self.assertTrue(np.abs(out_python - out_th).all() < 1e-16)
        out_python, out_th,  out_torch = compare_3d_backward(x_torch, out_python, out_torch, xr, xi, pool_size,
                                                                        strides, alpha, beta, dilations_rate,
                                                                        pads, mode)
        self.assertTrue(np.abs(out_python - out_th).all() < 1e-16)


if __name__ == "__main__":
    # unittest.test()
    unittest.main()
