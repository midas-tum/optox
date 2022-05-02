import torch
import optotf.maxpooling
import optoth.maxpooling
import numpy as np
import tensorflow as tf
import unittest

cuda1 = torch.device('cuda:0')
def compare_2d(xr, xi, pool_size, strides, alpha, beta, dilations_rate, pads, mode):
    t_real = torch.from_numpy(xr)
    t_imag = torch.from_numpy(xi)
    x_torch = torch.complex(t_real, t_imag)
    x_torch = x_torch.to(device=cuda1)

    out_torch = optoth.maxpooling.maxpooling2d(x_torch, pool_size, pads, strides, dilations_rate, alpha,
                                                        beta, padding_mode=mode)
    out_tensorflow = optotf.maxpooling.maxpooling2d(inputs=tf.complex(xr, xi), pool_size=pool_size,
                                                             strides=strides, alpha=alpha, beta=beta,
                                                             mode=mode, dilations_rate=dilations_rate)
    out_tf = out_tensorflow.numpy()
    out_th = out_torch.cpu().detach().numpy()
    return out_tf, out_th, out_tensorflow, out_torch, x_torch


def compare_2d_backward(x_torch, out_tensorflow, out_torch, xr, xi, pool_size, strides, alpha, beta, dilations_rate,
                        pads, mode):
    out_real = tf.math.real(out_tensorflow)
    out_imag = tf.math.imag(out_tensorflow)
    out_torch = out_torch.to(device=cuda1)

    x_diff_real = np.ones(out_real.shape).astype(np.float32) * -1
    x_diff_imag = np.ones(out_imag.shape).astype(np.float32) * -1
    x_diff = [x_diff_real, x_diff_imag]
    back_real_tensorflow, back_imag_tensorflow = optotf.maxpooling.maxpooling2d_grad_backward(x_in=[xr, xi],
                                                                                              x_out=[out_real, out_imag],
                                                                                              x_diff=x_diff,
                                                                                              pool_size=pool_size,
                                                                                              strides=strides,
                                                                                              alpha=alpha,
                                                                                              beta=beta,
                                                                                              mode=mode,
                                                                                              dilations_rate=dilations_rate)
    t_real = torch.from_numpy(x_diff_real)
    t_imag = torch.from_numpy(x_diff_imag)
    x_diff = torch.complex(t_real, t_imag)
    x_diff = x_diff.to(device=cuda1)

    back_out_torch = optoth.maxpooling.maxpooling2d_backward(x_torch, out_torch, x_diff, pool_size, pads,
                                                                      strides, dilations_rate, alpha, beta,
                                                                      padding_mode=mode)
    back_out_tensorflow = tf.complex(back_real_tensorflow, back_imag_tensorflow)
    out_tf = back_out_tensorflow
    out_th = back_out_torch.cpu().detach().numpy()
    return out_tf, out_th, out_tensorflow, out_torch


def compare_3d(xr, xi, pool_size, strides, alpha, beta, dilations_rate, pads, mode):
    t_real = torch.from_numpy(xr)
    t_imag = torch.from_numpy(xi)
    x_torch = torch.complex(t_real, t_imag)
    x_torch = x_torch.to(device=cuda1)

    out_torch = optoth.maxpooling.maxpooling3d(x_torch, pool_size, pads, strides, dilations_rate, alpha,
                                                        beta, padding_mode=mode)
    out_tensorflow = optotf.maxpooling.maxpooling3d(inputs=tf.complex(xr, xi), pool_size=pool_size,
                                                             strides=strides, alpha=alpha, beta=beta,
                                                             mode=mode, dilations_rate=dilations_rate)
    out_tf = out_tensorflow.numpy()
    out_th = out_torch.cpu().detach().numpy()
    return out_tf, out_th, out_tensorflow, out_torch, x_torch


def compare_3d_backward(x_torch, out_tensorflow, out_torch, xr, xi, pool_size, strides, alpha, beta, dilations_rate,
                        pads, mode):
    out_real = tf.math.real(out_tensorflow)
    out_imag = tf.math.imag(out_tensorflow)
    out_torch = out_torch.to(device=cuda1)

    x_diff_real = np.ones(out_real.shape).astype(np.float32) * -1
    x_diff_imag = np.ones(out_imag.shape).astype(np.float32) * -1
    x_diff = [x_diff_real, x_diff_imag]
    back_real_tensorflow, back_imag_tensorflow = optotf.maxpooling.maxpooling3d_grad_backward(x_in=[xr, xi],
                                                                                              x_out=[out_real, out_imag],
                                                                                              x_diff=x_diff,
                                                                                              pool_size=pool_size,
                                                                                              strides=strides,
                                                                                              alpha=alpha,
                                                                                              beta=beta,
                                                                                              mode=mode,
                                                                                              dilations_rate=dilations_rate)
    t_real = torch.from_numpy(x_diff_real)
    t_imag = torch.from_numpy(x_diff_imag)
    x_diff = torch.complex(t_real, t_imag)
    x_diff = x_diff.to(device=cuda1)

    back_out_torch = optoth.maxpooling.maxpooling3d_backward(x_torch, out_torch, x_diff, pool_size, pads,
                                                                      strides, dilations_rate, alpha, beta,
                                                                      padding_mode=mode)
    back_out_tensorflow = tf.complex(back_real_tensorflow, back_imag_tensorflow)
    out_tf = back_out_tensorflow
    out_th = back_out_torch.cpu().detach().numpy()
    return out_tf, out_th, out_tensorflow, out_torch


class TestFunction2d(unittest.TestCase):
    def test_1(self):
        dtype = np.float32
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

        mode = 'VALID'
        out_tf, out_th, out_tensorflow, out_torch, x_torch = compare_2d(xr, xi, pool_size, strides, alpha, beta,
                                                                        dilations_rate, pads, mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)

        out_tf, out_th, out_tensorflow, out_torch = compare_2d_backward(x_torch, out_tf, out_torch, xr, xi, pool_size,
                                                                        strides, alpha, beta, dilations_rate, pads,
                                                                        mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)

        mode = 'SAME'
        out_tf, out_th, out_tensorflow, out_torch, x_torch = compare_2d(xr, xi, pool_size, strides, alpha, beta,
                                                                        dilations_rate, pads, mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)

        out_tf, out_th, out_tensorflow, out_torch = compare_2d_backward(x_torch, out_tf, out_torch, xr, xi, pool_size,
                                                                        strides, alpha, beta, dilations_rate, pads,
                                                                        mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)


class TestFunction3d(unittest.TestCase):
    def test_1(self):
        dtype = np.float32
        pool_size = (2, 2, 2)
        strides = (2, 2, 2)
        pads = (0, 0, 0)
        alpha = 1
        beta = 1
        dilations_rate = (1, 1, 1)

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
        mode = 'VALID'
        out_tf, out_th, out_tensorflow, out_torch, x_torch = compare_3d(xr, xi, pool_size, strides, alpha, beta,
                                                                        dilations_rate, pads, mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)
        out_tf, out_th, out_tensorflow, out_torch = compare_3d_backward(x_torch, out_tf, out_torch, xr, xi, pool_size,
                                                                        strides, alpha, beta, dilations_rate,
                                                                        pads, mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)

        mode = 'SAME'
        out_tf, out_th, out_tensorflow, out_torch, x_torch = compare_3d(xr, xi, pool_size, strides, alpha, beta,
                                                                        dilations_rate, pads, mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)

        out_tf, out_th, out_tensorflow, out_torch = compare_3d_backward(x_torch, out_tf, out_torch, xr, xi, pool_size,
                                                                        strides, alpha, beta, dilations_rate,
                                                                        pads, mode)
        self.assertTrue(np.abs(out_tf - out_th).all() < 1e-16)


if __name__ == "__main__":
    # unittest.test()
    unittest.main()
