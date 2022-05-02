from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as _ops

__all__ = ['pad1d',
           'pad1d_transpose',
           'pad2d',
           'pad2d_transpose',
           'pad3d',
           'pad3d_transpose']

# load operators from the library
_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_pad_operator.so"))

@_ops.RegisterGradient("Pad1d")
def _pad1d_grad(op, grad):
    grad_x = _ext.pad1d_transpose(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad1dTranspose")
def _pad1d_transpose_grad(op, grad):
    grad_x = _ext.pad1d(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad2d")
def _pad2d_grad(op, grad):
    grad_x = _ext.pad2d_transpose(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad2dTranspose")
def _pad2d_transpose_grad(op, grad):
    grad_x = _ext.pad2d(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad3d")
def _pad3d_grad(op, grad):
    grad_x = _ext.pad3d_transpose(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        front=op.get_attr("front"),
        back=op.get_attr("back"),
        mode=op.get_attr("mode"))
    return [grad_x]  

@_ops.RegisterGradient("Pad3dTranspose")
def _pad3d_transpose_grad(op, grad):
    grad_x = _ext.pad3d(
        grad,
        left=op.get_attr("left"), 
        right=op.get_attr("right"),
        bottom=op.get_attr("bottom"),
        top=op.get_attr("top"),
        front=op.get_attr("front"),
        back=op.get_attr("back"),
        mode=op.get_attr("mode"))
    return [grad_x]  

def _pad(ndim, x, padding, mode, channel_last=True):
    """Padding of a Nd tensor.
    
    This function pads a Nd (ndim) tensor (rank ndim + 2). The tensorformat is either
    [N, C, W] for `channel_last=False` or [N, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        ndim: `int32` Dimensionality of the padding transform
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A padded `Tensor`. Has the same type as `tensor`.
    """
    # first reshape the input
    if channel_last:
        axes = [0, ndim+1, *range(1, ndim+1)]
        x = tf.transpose(x, axes)
        
    shape = tf.unstack(tf.shape(x))
    new_shape = [-1, *shape[2:]]
    new_shape = tf.stack(new_shape)
    x_r = tf.reshape(x, new_shape)

    if ndim == 1:
        op = _ext.pad1d
    elif ndim == 2:
        op = _ext.pad2d
    elif ndim == 3:
        op = _ext.pad3d
    else:
        raise ValueError('ndim must be 1, 2 or 3')

    # compute the output
    if x.dtype == tf.complex64 or x.dtype == tf.complex128:
        x_r = tf.complex(op(tf.math.real(x_r), mode, *padding), 
                         op(tf.math.imag(x_r), mode, *padding))
    else:
        x_r = op(x_r, mode, *padding)

    padded_shape = shape
    for i in range(ndim):
        padded_shape[-1-i] += padding[i*2] + padding[i*2+1]
    padded_shape = tf.stack(padded_shape)

    if channel_last:
        axes = [0, *range(2, ndim+2), 1]
        return tf.transpose(tf.reshape(x_r, padded_shape), axes)
    else:
        return tf.reshape(x_r, padded_shape)

def _pad_transpose(ndim, x, padding, mode, channel_last=True):
    """Transpose Padding of a Nd tensor.
    
    This function pads a Nd (ndim) tensor (rank ndim + 2). The tensorformat is either
    [N, C, W] for `channel_last=False` or [N, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        ndim: `int32` Dimensionality of the padding transform
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transpose padded `Tensor`. Has the same type as `tensor`.
    """
    # first reshape the input
    if channel_last:
        axes = [0, ndim+1, *range(1, ndim+1)]
        x = tf.transpose(x, axes)
        
    shape = tf.unstack(tf.shape(x))
    new_shape = [-1, *shape[2:]]
    new_shape = tf.stack(new_shape)
    x_r = tf.reshape(x, new_shape)

    if ndim == 1:
        op = _ext.pad1d_transpose
    elif ndim == 2:
        op = _ext.pad2d_transpose
    elif ndim == 3:
        op = _ext.pad3d_transpose
    else:
        raise ValueError('ndim must be 1, 2 or 3')

    # compute the output
    if x.dtype == tf.complex64 or x.dtype == tf.complex128:
        x_r = tf.complex(op(tf.math.real(x_r), mode, *padding), 
                         op(tf.math.imag(x_r), mode, *padding))
    else:
        x_r = op(x_r, mode, *padding)

    padded_shape = shape
    for i in range(ndim):
        padded_shape[-1-i] -= padding[i*2] + padding[i*2+1]
    padded_shape = tf.stack(padded_shape)

    if channel_last:
        axes = [0, *range(2, ndim+2), 1]
        return tf.transpose(tf.reshape(x_r, padded_shape), axes)
    else:
        return tf.reshape(x_r, padded_shape)

def pad1d(x, padding, mode, channel_last=True):
    """Padding of a 1d tensor.
    
    This function pads a 1d tensor (rank 3). The tensorformat is either
    [N, C, W] for `channel_last=False` or [N, W, C] for `channel_last=True`. The tensor is
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
    _pad(1, x, padding, mode, channel_last)

def pad1d_transpose(x, padding, mode, channel_last=True):
    """Transpose padding of a 1d tensor.
    
    This function transpose pads a 1d tensor (rank 3). The tensorformat is either
    [N, C, W] for `channel_last=False` or [N, W, C] for `channel_last=True`. The tensor is
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
    _pad_transpose(1, x, padding, mode, channel_last)

def pad2d(x, padding, mode, channel_last=True):
    """Padding of a 2d tensor.
    
    This function pads a 2d tensor (rank 4). The tensorformat is either
    [N, C, H, W] for `channel_last=False` or [N, H, W, C] for `channel_last=True`. The tensor is
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
    _pad(2, x, padding, mode, channel_last)

def pad2d_transpose(x, padding, mode, channel_last=True):
    """Transpose padding of a 2d tensor.
    
    This function transpose pads a 2d tensor (rank 4). The tensorformat is either
    [N, C, H, W] for `channel_last=False` or [N, H, W, C] for `channel_last=True`. The tensor is
    padded by values specified in `padding` [W0, W1, H0, H1] where 0 indicates the padding before and 1
    indicates the padding after. This functions supports the padding modes "reflect", "symmetric" and "replicate".

    Args:
        tensor: A `Tensor`.
        padding: A `Tensor` of type `int32`.
        mode: One of "reflect", "symmetric", or "replicate" (case-insensitive)
        channel_last: 
    Returns:
        A transposed padded `Tensor`. Has the same type as `tensor`.
    """
    _pad_transpose(2, x, padding, mode, channel_last)

def pad3d(x, padding, mode, channel_last=True):
    """Padding of a 3d tensor.
    
    This function pads a 3d tensor (rank 5). The tensorformat is either
    [N, C, D, H, W] for `channel_last=False` or [N, D, H, W, C] for `channel_last=True`. The tensor is
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

    _pad(3, x, padding, mode, channel_last)

def pad3d_transpose(x, padding, mode, channel_last=True):
    """Transpose padding of a 3d tensor.
    
    This function transpose pads a 3d tensor (rank 5). The tensorformat is either
    [N, C, D, H, W] for `channel_last=False` or [N, D, H, W, C] for `channel_last=True`. The tensor is
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
    _pad_transpose(3, x, padding, mode, channel_last)