import torch

import _ext.th_scale_operator

def get_operator(dtype):
    if dtype == torch.float32:
        return _ext.th_scale_operator.Scale_float()
    elif dtype == torch.float64:
        return _ext.th_scale_operator.Scale_double()       
    elif dtype == torch.complex64:
        return _ext.th_scale_operator.Scale_float2()       
    elif dtype == torch.complex128:
        return _ext.th_scale_operator.Scale_double2()   
    else:
        raise RuntimeError('Invalid dtype!')

class ScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.op = get_operator(x.dtype)
        shape = x.shape
        out = ctx.op.forward(x.flatten())
        return out.view(shape)

    @staticmethod
    def backward(ctx, grad_out):
        shape = grad_out.shape
        out = ctx.op.adjoint(grad_out.flatten())
        return out.view(shape)

class Scale(torch.nn.Module):
    def forward(self, x):
        return ScaleFunction.apply(x)