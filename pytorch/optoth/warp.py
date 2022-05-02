import torch

import _ext.th_warp_operator

__all__ = ['WarpFunction', 'WarpTransposeFunction']

def get_operator(dtype, mode):
    assert dtype in [torch.float32, torch.complex64, torch.float64, torch.complex128]
    is_double = dtype in [torch.float64, torch.complex128]
    if not is_double:
        return _ext.th_warp_operator.Warp_float(mode)
    else: #elif dtype == torch.float64:
        return _ext.th_warp_operator.Warp_double(mode)

class WarpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, mode='zeros'):
        ctx.save_for_backward(u, x)
        ctx.op = get_operator(x.dtype, mode)
        if x.is_complex():
            x_re = torch.real(x).contiguous()
            x_im = torch.imag(x).contiguous()
            out_re, _ = ctx.op.forward(x_re, u, x_re)
            out_im, _ = ctx.op.forward(x_im, u, x_im)
            out = torch.complex(out_re, out_im)
        else:
            out, _ = ctx.op.forward(x, u, x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        u = ctx.saved_tensors[0]
        x = ctx.saved_tensors[1]
        if x.is_complex():
            x_re = torch.real(x).contiguous()
            x_im = torch.imag(x).contiguous()
            grad_out_re = torch.real(grad_out).contiguous()
            grad_out_im = torch.imag(grad_out).contiguous()
            grad_x_re, grad_u_re = ctx.op.adjoint(grad_out_re, u, x_re)
            grad_x_im, grad_u_im = ctx.op.adjoint(grad_out_im, u, x_im)
            grad_x = torch.complex(grad_x_re, grad_x_im)
            grad_u = grad_u_re + grad_u_im
        else:
            grad_x, grad_u = ctx.op.adjoint(grad_out, u, x)
        return grad_x, grad_u, None

class WarpTransposeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, mode='zeros'):
        ctx.save_for_backward(u, x)
        ctx.op = get_operator(x.dtype, mode)
        if x.is_complex():
            x_re = torch.real(x).contiguous()
            x_im = torch.imag(x).contiguous()
            out_re, _ = ctx.op.adjoint(x_re, u, x_re)
            out_im, _ = ctx.op.adjoint(x_im, u, x_im)
            out = torch.complex(out_re, out_im)
        else:
            out, _ = ctx.op.adjoint(x, u, x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        u = ctx.saved_tensors[0]
        x = ctx.saved_tensors[1]
        if x.is_complex():
            x_re = torch.real(x).contiguous()
            x_im = torch.imag(x).contiguous()
            grad_out_re = torch.real(grad_out).contiguous()
            grad_out_im = torch.imag(grad_out).contiguous()
            grad_x_re, grad_u_re = ctx.op.forward(grad_out_re, u, x_re)
            grad_x_im, grad_u_im = ctx.op.forward(grad_out_im, u, x_im)
            grad_x = torch.complex(grad_x_re, grad_x_im)
            grad_u = grad_u_re + grad_u_im
        else:
            grad_x, grad_u = ctx.op.forward(grad_out, u, x)
        return grad_x, grad_u, None
