import numpy as np

import _ext.py_nabla_operator

__all__ = ['float_2d',
           'double_2d',
           'float_3d',
           'double_3d',
           'float_4d',
           'double_4d']

float_2d = _ext.py_nabla_operator.Nabla_2d_float
double_2d = _ext.py_nabla_operator.Nabla_2d_double

float_3d = _ext.py_nabla_operator.Nabla_3d_float
double_3d = _ext.py_nabla_operator.Nabla_3d_double

float_4d = _ext.py_nabla_operator.Nabla_4d_float
double_4d = _ext.py_nabla_operator.Nabla_4d_double

class Nabla(object):
    def __init__(self, dim, hx=1, hy=1, hz=1, ht=1):
        self.dim = dim
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ht = ht
    
    def _get_op(self, dtype, dim, hx, hy, hz=1, ht=1):
        if dtype == np.float32 or dtype == np.complex64:
            if dim == 2:
                return float_2d(hx, hy)
            elif dim == 3:
                return float_3d(hx, hy, hz)
            elif dim == 4:
                return float_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        elif dtype == np.float64 or dtype == np.complex128:
            if dim == 2:
                return double_2d(hx, hy)
            elif dim == 3:
                return double_3d(hx, hy, hz)
            elif dim == 4:
                return double_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            raise RuntimeError('Invalid dtype!')
        
    def forward(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.forward(x_re) + 1j * op.forward(x_im)
        else:
            return op.forward(x)

    def adjoint(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return op.adjoint(x_re) + 1j * op.adjoint(x_im)
        else:
            return op.adjoint(x)
