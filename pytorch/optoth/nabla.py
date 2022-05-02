import torch

import _ext.th_nabla_operator

__all__ = ['float_2d',
           'double_2d',
           'float_3d',
           'double_3d',
           'float_4d',
           'double_4d']

float_2d = _ext.th_nabla_operator.Nabla_2d_float
double_2d = _ext.th_nabla_operator.Nabla_2d_double

float_3d = _ext.th_nabla_operator.Nabla_3d_float
double_3d = _ext.th_nabla_operator.Nabla_3d_double

float_4d = _ext.th_nabla_operator.Nabla_4d_float
double_4d = _ext.th_nabla_operator.Nabla_4d_double

class Nabla(torch.nn.Module):
    def __init__(self, dim, hx=1, hy=1, hz=1, ht=1):
        super().__init__()
        self.dim = dim
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ht = ht
    
    def _get_op(self, dtype, dim, hx, hy, hz=1, ht=1):
        assert dtype in [torch.float32, torch.complex64, torch.float64, torch.complex128]
        is_double = dtype in [torch.float64, torch.complex128]

        if not is_double:
            if dim == 2:
                return float_2d(hx, hy)
            elif dim == 3:
                return float_3d(hx, hy, hz)
            elif dim == 4:
                return float_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            if dim == 2:
                return double_2d(hx, hy)
            elif dim == 3:
                return double_3d(hx, hy, hz)
            elif dim == 4:
                return double_4d(hx, hy, hz, ht)
            else:
                raise RuntimeError('Invalid number of dimensions!')

    def forward(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if x.is_complex():
            out_re = op.forward(torch.real(x).contiguous())
            out_im = op.forward(torch.imag(x).contiguous())
            out = torch.complex(out_re, out_im)
        else:
            out = op.forward(x)
        return out

    def adjoint(self, x):
        op = self._get_op(x.dtype, self.dim, self.hx, self.hy, self.hz, self.ht)
        
        if x.is_complex():
            out_re = op.adjoint(torch.real(x).contiguous())
            out_im = op.adjoint(torch.imag(x).contiguous())
            out = torch.complex(out_re, out_im)
        else:
            out = op.adjoint(x)
        return out
