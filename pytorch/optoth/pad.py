import torch

import _ext.th_pad_operator

__all__ = ['pad1d',
           'pad1d_transpose',
           'pad1d_symmetric',
           'pad1d_symmetric_transpose',
           'pad2d',
           'pad2d_transpose',
           'pad2d_symmetric',
           'pad2d_symmetric_transpose',
           'pad3d',
           'pad3d_tranpose',
           'pad3d_symmetric',
           'pad3d_symmetric_transpose']

def pad1d(x, padding, mode):
    """Padding of a 2d tensor.
    
    This function pads a 1d tensor (rank 3). The tensorformat is [N, C, W]. The tensor is
    padded by values specified in `padding` [W0, W1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunction.apply(1, x, padding, mode)


def pad1d_transpose(x, padding, mode):
    """Transpose padding of a 2d tensor.
    
    This function transpose pads a 1d tensor (rank 3). The tensorformat is [N, C, W]. The tensor is
    padded by values specified in `padding` [W0, W1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunctionTranspose.apply(1, x, padding, mode)


# legacy
def pad1d_symmetric(x, padding):
    return PadFunction.apply(1, x, padding, 'symmetric')


def pad1d_symmetric_transpose(x, padding):
    return PadFunctionTranspose.apply(1, x, padding, 'symmetric')

def get_operator(dim, dtype, pad, mode):
    assert dim in [1, 2, 3]
    assert len(pad) == 2 * dim
    assert type(mode) == str
    assert dtype in [torch.float32, torch.complex64, torch.float64, torch.complex128]
    is_double = dtype in [torch.float64, torch.complex128]

    if dim == 1 and not is_double:
        return _ext.th_pad_operator.Pad1d_float(*pad, mode)
    elif dim == 1 and is_double:
        return _ext.th_pad_operator.Pad1d_double(*pad, mode)
    elif dim == 2 and not is_double:
        return _ext.th_pad_operator.Pad2d_float(*pad, mode)
    elif dim == 2 and is_double:
        return _ext.th_pad_operator.Pad2d_double(*pad, mode)
    elif dim == 3 and not is_double:
        return _ext.th_pad_operator.Pad3d_float(*pad, mode)
    elif dim == 3 and is_double:
        return _ext.th_pad_operator.Pad3d_double(*pad, mode)
    else:
        raise RuntimeError('Invalid dtype!')

class PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, x, pad, mode):
        ctx.op = get_operator(dim, x.dtype, pad, mode)
        ctx.shape = x.shape
        ctx.dim = dim
        pad_shape = list(x.shape)

        for i in range(dim):
            pad_shape[-1-i] += pad[2*i] + pad[2*i+1]

        if x.is_complex():
            out_re = ctx.op.forward(torch.real(x).contiguous().reshape(-1, *x.shape[2:])).view(*pad_shape)
            out_im = ctx.op.forward(torch.imag(x).contiguous().reshape(-1, *x.shape[2:])).view(*pad_shape)
            out = torch.complex(out_re, out_im)
        else:
            out = ctx.op.forward(x.reshape(-1, *x.shape[2:])).view(*pad_shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out.is_complex():
            grad_x_re = ctx.op.adjoint(torch.real(grad_out).contiguous().reshape(-1, *grad_out.shape[2:]))
            grad_x_im = ctx.op.adjoint(torch.imag(grad_out).contiguous().reshape(-1, *grad_out.shape[2:]))
            grad_x = torch.complex(grad_x_re, grad_x_im)
        else:
            grad_x = ctx.op.adjoint(grad_out.reshape(-1, *grad_out.shape[2:]))
        output = None, grad_x.view(ctx.shape), *[None for _ in range(ctx.dim*2)]
        return output


class PadFunctionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, x, pad, mode):
        ctx.op = get_operator(dim, x.dtype, pad, mode)
        ctx.shape = x.shape
        ctx.dim = dim

        pad_shape = list(x.shape)

        for i in range(dim):
            pad_shape[-1-i] -= pad[2*i] + pad[2*i+1]

        if x.is_complex():
            out_re = ctx.op.adjoint(torch.real(x).contiguous().reshape(-1, *x.shape[2:])).view(*pad_shape)
            out_im = ctx.op.adjoint(torch.imag(x).contiguous().reshape(-1, *x.shape[2:])).view(*pad_shape)
            out = torch.complex(out_re, out_im)
        else:
            out = ctx.op.adjoint(x.reshape(-1, *x.shape[2:])).view(*pad_shape)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out.is_complex():
            grad_x_re = ctx.op.forward(torch.real(grad_out).contiguous().reshape(-1, *grad_out.shape[2:]))
            grad_x_im = ctx.op.forward(torch.imag(grad_out).contiguous().reshape(-1, *grad_out.shape[2:]))
            grad_x = torch.complex(grad_x_re, grad_x_im)
        else:
            grad_x = ctx.op.forward(grad_out.reshape(-1, *grad_out.shape[2:]))
        output = None, grad_x.view(ctx.shape), *[None for _ in range(ctx.dim*2)]

        return output

def pad2d(x, padding, mode):
    """Padding of a 2d tensor.
    
    This function pads a 2d tensor (rank 4). The tensorformat is [N, C, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunction.apply(2, x, padding, mode)


def pad2d_transpose(x, padding, mode):
    """Transpose padding of a 2d tensor.
    
    This function transpose pads a 2d tensor (rank 4). The tensorformat is [N, C, D, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunctionTranspose.apply(2, x, padding, mode)


# legacy
def pad2d_symmetric(x, padding):
    return PadFunction.apply(2, x, padding, 'symmetric')


def pad2d_symmetric_transpose(x, padding):
    return PadFunctionTranspose.apply(2, x, padding, 'symmetric')

def pad3d(x, padding, mode):
    """Padding of a 3d tensor.
    
    This function pads a 3d tensor (rank 5). The tensorformat is [N, C, D, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunction.apply(3, x, padding, mode)


def pad3d_transpose(x, padding, mode):
    """Transpose padding of a 3d tensor.
    
    This function transpose pads a 3d tensor (rank 5). The tensorformat is [N, C, D, H, W]. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1, D0, D1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    return PadFunctionTranspose.apply(3, x, padding, mode)


# legacy
def pad3d_symmetric(x, padding):
    return PadFunction.apply(x, padding, 'symmetric')

def pad3d_symmetric_transpose(x, padding):
    return PadFunctionTranspose.apply(3, x, padding, 'symmetric')
