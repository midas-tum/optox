import numpy as np

import _ext.py_pad_operator

__all__ = ['float_1d',
           'double_1d',
           'float_2d',
           'double_2d',
           'float_3d',
           'double_3d']

float_1d = _ext.py_pad_operator.Pad1d_float
double_1d = _ext.py_pad_operator.Pad1d_double

float_2d = _ext.py_pad_operator.Pad2d_float
double_2d = _ext.py_pad_operator.Pad2d_double

float_3d = _ext.py_pad_operator.Pad3d_float
double_3d = _ext.py_pad_operator.Pad3d_double

class _Pad(object):
    def __init__(self, dim, padding, mode):
        assert dim in [1, 2, 3]
        self.padding = padding
        self.mode = mode
        self.dim = dim

    def _get_op(self, dtype):
        assert dtype in [np.float32, np.complex64, np.float64, np.complex128]
        is_double = dtype in [np.float64, np.complex128]

        if self.dim == 1 and not is_double:
            return float_1d(*self.padding, self.mode)
        elif self.dim == 1 and is_double:
            return double_1d(*self.padding, self.mode)
        elif self.dim == 2 and not is_double:
            return float_2d(*self.padding, self.mode)
        elif self.dim == 2 and is_double:
            return double_2d(*self.padding, self.mode)
        elif self.dim == 3 and not is_double:
            return float_3d(*self.padding, self.mode)
        elif self.dim == 3 and is_double:
            return double_3d(*self.padding, self.mode)
        else:
            raise RuntimeError('Invalid dtype!')

    def forward(self, x):
        op = self._get_op(x.dtype)

        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.forward(x_re) + 1j * op.forward(x_im)
        else:
            return op.forward(x)

    def adjoint(self, x):
        op = self._get_op(x.dtype)

        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.adjoint(x_re) + 1j * op.adjoint(x_im)
        else:
            return op.adjoint(x)

class Pad1d(_Pad):
    def __init__(self, padding, mode):
        super().__init__(1, padding, mode)

class Pad2d(_Pad):
    def __init__(self, padding, mode):
        super().__init__(2, padding, mode)

class Pad3d(_Pad):
    def __init__(self, padding, mode):
        super().__init__(3, padding, mode)
