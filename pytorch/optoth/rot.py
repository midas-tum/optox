import torch

import _ext.th_rot_operator

__all__ = ['RotationFunction']

class RotationFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype):
        if dtype == torch.float32:
            return _ext.th_rot_operator.Rot_float()
        elif dtype == torch.float64:
            return _ext.th_rot_operator.Rot_double()
        else:
            raise RuntimeError('Unsupported dtype!')

    @staticmethod
    def forward(ctx, x, angles):
        ctx.save_for_backward(x, angles)
        ctx.op = RotationFunction._get_operator(x.dtype)
        return ctx.op.forward(x, angles)

    @staticmethod
    def backward(ctx, grad_in):
        x, angles = ctx.saved_tensors
        grad_x = ctx.op.adjoint(grad_in, angles)
        return grad_x, None
