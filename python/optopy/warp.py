import numpy as np

import _ext.py_warp_operator

__all__ = ['float_2d', 'double_2d']

float_2d = _ext.py_warp_operator.Warp_2d_float
double_2d = _ext.py_warp_operator.Warp_2d_double

class Warp(object):
    def __init__(self, mode):
        self.u = None
        self.x = None
        self.mode = mode

    def _get_op(self, dtype):
        if dtype == np.float32 or dtype == np.complex64:
            return float_2d(self.mode)
        elif dtype == np.float64 or dtype == np.complex128:
            return double_2d(self.mode)
        else:
            raise RuntimeError('Invalid dtype!')

    def forward(self, x, u):
        # save for backward
        self.u = u
        self.x = x

        op = self._get_op(x.dtype)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.forward(x_re, u, x_re)[0] + 1j * op.forward(x_im, u, x_im)[0]
        else:
            return op.forward(x, u, x)[0]

    def adjoint(self, grad_out, u = None, x = None):
        # check if all inputs are available
        if self.u is None and u is None:
            raise RuntimeError("u is not defined. Forward not called or u not passed in function call.")
        if self.x is None and x is None:
            raise RuntimeError("x is not defined. Forward not called or x not passed in function call.")
                
        op = self._get_op(grad_out.dtype)

        # use passed variables or saved ones from forward path
        u = self.u if u is None else u
        x = self.x if x is None else x

        if np.iscomplexobj(grad_out):
            grad_out_re = np.ascontiguousarray(np.real(grad_out))
            grad_out_im = np.ascontiguousarray(np.imag(grad_out))
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.adjoint(grad_out_re, u, x_re)[0] + 1j * op.adjoint(grad_out_im, u, x_im)[0]
        else:
            return op.adjoint(grad_out, u, x)[0]
