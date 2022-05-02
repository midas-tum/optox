from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as _ops
import unittest
import numpy as np
import math

__all__ = ['maxpooling1d', 'maxpooling1d_grad_backward', 'maxpooling2d', 'maxpooling2d_grad_backward',
           'maxpooling3d', 'maxpooling3d_grad_backward', 'maxpooling4d', 'maxpooling4d_grad_backward']


def transpose_channel_first_last(inputs=None, change_back=False, dimensions='2d'):
    outputs = []
    if change_back == False:
        for x in inputs:
            if tf.rank(x) == 2:
                # (c,h)->(h, c)
                x = tf.transpose(x, (1, 0))
            elif tf.rank(x) == 3 and dimensions == '1d':
                # (b,c,h)->(b,h,c)
                x = tf.transpose(x, (0, 2, 1))
            elif tf.rank(x) == 3 and dimensions == '2d':
                # (c,h,w)->(h,w,c)
                x = tf.transpose(x, (1, 2, 0))
            elif tf.rank(x) == 4 and dimensions == '2d':
                # (b,c,h,w)->(b,h,w,c)
                x = tf.transpose(x, (0, 2, 3, 1))
            elif tf.rank(x) == 4 and dimensions == '3d':
                # (c, h, w, d)->(h, w, d, c)
                x = tf.transpose(x, (1, 2, 3, 0))
            elif tf.rank(x) == 5 and dimensions == '3d':
                # (b,c,h,w,d)->(b,h,w,d,c)
                x = tf.transpose(x, (0, 2, 3, 4, 1))
            elif tf.rank(x) == 5 and dimensions == '4d':
                # (c,t,h,w,d)->(t,h,w,d,c)
                x = tf.transpose(x, (1, 2, 3, 4, 0))
            elif x.ndim == 6:
                # (b,c,t,h,w,d)->(b,t,h,w,d,c)
                x = tf.transpose(x, (0, 2, 3, 4, 5, 1))

            else:
                raise Exception('input dimension should be 2, 3, 4, 5 or 6!')
            outputs.append(x)
    else:
        for x in inputs:
            if tf.rank(x) == 2:
                # (h,c)->(c,h)
                x = tf.transpose(x, (1, 0))
            elif tf.rank(x) == 3 and dimensions == '1d':
                # (b,h,c)->(b,c,h)
                x = tf.transpose(x, (0, 2, 1))
            elif tf.rank(x) == 3 and dimensions == '2d':
                # (h,w,c)->(c,h,w)
                x = tf.transpose(x, (2, 0, 1))
            elif tf.rank(x) == 4 and dimensions == '2d':
                # (b,h,w,c)->(b,c,h,w)
                x = tf.transpose(x, (0, 3, 1, 2))
            elif tf.rank(x) == 4 and dimensions == '3d':
                # (h,w,d,c)->(c,h,w,d)
                x = tf.transpose(x, (3, 0, 1, 2))
            elif tf.rank(x) == 5 and dimensions == '3d':
                # (b,h,w,d,c)->(b,c,h,w,d)
                x = tf.transpose(x, (0, 4, 1, 2, 3))
            elif x.ndim == 5 and dimensions == '4d':
                # (t,h,w,d,c)->(c,t,h,w,d)
                x = tf.transpose(x, (4, 0, 1, 2, 3))
            elif x.ndim == 6:
                # (b,t,h,w,d,c)->(b,c,t,h,w,d)
                x = tf.transpose(x, (0, 5, 1, 2, 3, 4))
            else:
                raise Exception('input dimension should be 2, 3, 4, 5 or 6!')
            outputs.append(x)
    return outputs


# load operators from the library
_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_maxpooling_operator.so"))


def maxpooling1d(inputs, pool_size=(2,), strides=(2,), alpha=1, beta=1, channel_first=False, name="maxpooling_1d",
                 dilations_rate=(1,), pad=(0,), mode='VALID', ceil_mode=True):
    ceil_mode = 1 if ceil_mode == True else 0
    op = _ext.maxpooling1d

    if type(inputs) is list:
        inputs = tf.complex(inputs[0], inputs[1])
    if inputs.dtype.is_complex is False:
        inputs = tf.complex(inputs, inputs * 0)
    input_real, input_imag = tf.math.real(inputs), tf.math.imag(inputs)
    if channel_first:
        [input_real, input_imag] = transpose_channel_first_last([input_real, input_imag], dimensions='1d')
    out_real, out_imag, out_idx = op(input_real, input_imag,
                                     pool_size[0], strides[0], pad[0],
                                     alpha=alpha, beta=beta, name=name, dilation_rate_h=dilations_rate[0],
                                     mode=mode,ceil_mode=ceil_mode)
    if channel_first:
        [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                            change_back=True, dimensions='1d')

    return tf.complex(out_real, out_imag)


@_ops.RegisterGradient("Maxpooling1d")
def grad(op, grad1, grad2, grad3):
    op_backward = _ext.maxpooling1d_grad_backward

    grad2 = grad1

    kernel_h = op.get_attr("kernel_h")
    stride_h = op.get_attr("stride_h")
    alpha = op.get_attr("alpha")
    beta = op.get_attr("beta")
    dilation_rate_h = op.get_attr("dilation_rate_h")
    pad_h= op.get_attr("pad_h")
    mode = op.get_attr("mode")
    ceil_mode = op.get_attr("ceil_mode")
    top_diff_real, top_diff_imag = tf.math.real(grad1), tf.math.imag(grad2)
    bottom_diff_real, bottom_diff_imag = op_backward(input_real=op.inputs[0],
                                                     input_imag=op.inputs[1],
                                                     output_real=op.outputs[0],
                                                     output_imag=op.outputs[1],
                                                     top_diff_real=top_diff_real,
                                                     top_diff_imag=top_diff_imag,
                                                     indices=op.outputs[2],
                                                     kernel_h=kernel_h,
                                                     stride_h=stride_h,
                                                     pad_h=pad_h,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     dilation_rate_h=dilation_rate_h,
                                                     mode=mode,
                                                     with_indices=False,
                                                     ceil_mode=ceil_mode)
    return bottom_diff_real, bottom_diff_imag


def maxpooling2d(inputs, pool_size=(2, 2), strides=(2, 2), alpha=1, beta=1, channel_first=False, name="maxpooling_1d",
                 dilations_rate=(1, 1), pad=(0,0), mode='VALID', ceil_mode=True):
    ceil_mode =int(ceil_mode)
    op = _ext.maxpooling2d
    if type(inputs) is list:
        inputs = tf.complex(inputs[0], inputs[1])
    if inputs.dtype.is_complex is False:
        inputs = tf.complex(inputs, inputs * 0)
    input_real, input_imag = tf.math.real(inputs), tf.math.imag(inputs)
    if channel_first:
        [input_real, input_imag] = transpose_channel_first_last([input_real, input_imag], dimensions='2d')
    out_real, out_imag, out_idx = op(input_real, input_imag,
                                     pool_size[0], pool_size[1], strides[0], strides[1], pad[0], pad[1],
                                     alpha=alpha, beta=beta, name=name, dilation_rate_h=dilations_rate[0],
                                     dilation_rate_w=dilations_rate[1],
                                     mode=mode, ceil_mode=ceil_mode)
    if channel_first:
        [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                            change_back=True, dimensions='2d')

    return tf.complex(out_real, out_imag)


@_ops.RegisterGradient("Maxpooling2d")
def grad(op, grad1, grad2, grad3):
    grad2 = grad1
    op_backward = _ext.maxpooling2d_grad_backward
    top_diff_real, top_diff_imag = tf.math.real(grad1), tf.math.imag(grad2)
    kernel_h = op.get_attr("kernel_h")
    kernel_w = op.get_attr("kernel_w")
    stride_h = op.get_attr("stride_h")
    stride_w = op.get_attr("stride_w")
    alpha = op.get_attr("alpha")
    beta = op.get_attr("beta")
    dilation_rate_h = op.get_attr("dilation_rate_h")
    dilation_rate_w = op.get_attr("dilation_rate_w")
    pad_h=op.get_attr("pad_h")
    pad_w = op.get_attr("pad_w")
    mode = op.get_attr("mode")
    ceil_mode=op.get_attr("ceil_mode")
    bottom_diff_real, bottom_diff_imag = op_backward(input_real=op.inputs[0],
                                                     input_imag=op.inputs[1],
                                                     output_real=op.outputs[0],
                                                     output_imag=op.outputs[1],
                                                     top_diff_real=top_diff_real,
                                                     top_diff_imag=top_diff_imag,
                                                     indices=op.outputs[2],
                                                     kernel_h=kernel_h,
                                                     kernel_w=kernel_w,
                                                     stride_h=stride_h,
                                                     stride_w=stride_w,
                                                     pad_h=pad_h,
                                                     pad_w=pad_w,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     dilation_rate_h=dilation_rate_h,
                                                     dilation_rate_w=dilation_rate_w,
                                                     mode=mode,
                                                     with_indices=True,
                                                     ceil_mode=ceil_mode)

    return bottom_diff_real, bottom_diff_imag


def maxpooling3d(inputs, pool_size=(2, 2, 2), strides=(2, 2, 2), alpha=1, beta=1, channel_first=False,
                 name="maxpooling_1d",
                 dilations_rate=(1, 1, 1), pad=(0,0,0), mode='VALID', ceil_mode=True):
    ceil_mode = int(ceil_mode)
    op = _ext.maxpooling3d

    if type(inputs) is list:
        inputs = tf.complex(inputs[0], inputs[1])
    if inputs.dtype.is_complex is False:
        inputs = tf.complex(inputs, inputs * 0)
    input_real, input_imag = tf.math.real(inputs), tf.math.imag(inputs)
    if channel_first:
        [input_real, input_imag] = transpose_channel_first_last([input_real, input_imag], dimensions='3d')
    out_real, out_imag, out_idx = op(input_real, input_imag,
                                     pool_size[0], pool_size[1], pool_size[2], strides[0], strides[1], strides[2], pad[0], pad[1],
                                     pad[2],
                                     alpha=alpha, beta=beta, name=name, dilation_rate_h=dilations_rate[0],
                                     dilation_rate_w=dilations_rate[1], dilation_rate_d=dilations_rate[2],
                                     mode=mode, ceil_mode=ceil_mode)
    if channel_first:
        [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                            change_back=True, dimensions='3d')

    return tf.complex(out_real, out_imag)


@_ops.RegisterGradient("Maxpooling3d")
def grad(op, grad1, grad2, grad3):
    grad2 = grad1
    op_backward = _ext.maxpooling3d_grad_backward
    top_diff_real, top_diff_imag = tf.math.real(grad1), tf.math.imag(grad2)

    kernel_h = op.get_attr("kernel_h")
    kernel_w = op.get_attr("kernel_w")
    kernel_d = op.get_attr("kernel_d")
    stride_h = op.get_attr("stride_h")
    stride_w = op.get_attr("stride_w")
    stride_d = op.get_attr("stride_d")
    alpha = op.get_attr("alpha")
    beta = op.get_attr("beta")
    pad_h = op.get_attr("pad_h")
    pad_w = op.get_attr("pad_w")
    pad_d = op.get_attr("pad_d")
    dilation_rate_h = op.get_attr("dilation_rate_h")
    dilation_rate_w = op.get_attr("dilation_rate_w")
    dilation_rate_d = op.get_attr("dilation_rate_d")
    mode = op.get_attr("mode")
    ceil_mode = op.get_attr("ceil_mode")

    bottom_diff_real, bottom_diff_imag = op_backward(input_real=op.inputs[0],
                                                     input_imag=op.inputs[1],
                                                     output_real=op.outputs[0],
                                                     output_imag=op.outputs[1],
                                                     top_diff_real=top_diff_real,
                                                     top_diff_imag=top_diff_imag,
                                                     indices=op.outputs[2],
                                                     kernel_h=kernel_h,
                                                     kernel_w=kernel_w,
                                                     kernel_d=kernel_d,
                                                     stride_h=stride_h,
                                                     stride_w=stride_w,
                                                     stride_d=stride_d,
                                                     pad_h=pad_h,
                                                     pad_w=pad_w,
                                                     pad_d=pad_d,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     dilation_rate_h=dilation_rate_h,
                                                     dilation_rate_w=dilation_rate_w,
                                                     dilation_rate_d=dilation_rate_d,
                                                     mode=mode,
                                                     with_indices=True,
                                                     ceil_mode=ceil_mode)
    return bottom_diff_real, bottom_diff_imag


def maxpooling4d(inputs, pool_size=(2, 2, 2, 2), strides=(2, 2, 2, 2), alpha=1, beta=1, channel_first=False,
                 name='maxpooling_4d', dilations_rate=(1, 1, 1, 1), pad=(0, 0, 0, 0), mode='VALID', ceil_mode=True):
    ceil_mode = int(ceil_mode)
    def calculate_out_size(input_size, pool_size, strides, dilations_rate, mode, pad, ceil_mode):

        assert (len(input_size) == len(pool_size) == len(strides) == len(dilations_rate))
        out_size = [0] * len(pool_size)
        for i in range(len(pool_size)):
            effective_filter_size = (pool_size[i] - 1) * dilations_rate[i] + 1
            if mode.lower() == 'valid':
                if ceil_mode==1:

                    out_ = math.ceil((input_size[i] + 2*pad[i] - effective_filter_size + strides[i]) / strides[i])
                else:
                    out_ = math.floor((input_size[i] + 2*pad[i] - effective_filter_size + strides[i]) / strides[i])
            elif mode.lower() == 'same':
                out_ =math.ceil(input_size[i]/ strides[i])
            else:
                raise ValueError("mode must be either valid or same!")
            out_size[i] = out_
        return out_size

    op = _ext.maxpooling4d

    if type(inputs) is list:
        inputs = tf.complex(inputs[0], inputs[1])
    if inputs.dtype.is_complex is False:
        inputs = tf.complex(inputs, inputs * 0)
    input_real, input_imag = tf.math.real(inputs), tf.math.imag(inputs)
    if channel_first:
        [input_real, input_imag] = transpose_channel_first_last([input_real, input_imag], dimensions='4d')

    input_real_shape = input_real.shape
    if len(input_real_shape) == 5:
        batch_, time_in_, height_in_, width_in_, depth_in_, channels_ = 0, input_real_shape[0], input_real_shape[1], \
                                                                        input_real_shape[2], input_real_shape[3], \
                                                                        input_real_shape[-1]
    elif len(input_real_shape) == 6:
        batch_, time_in_, height_in_, width_in_, depth_in_, channels_ = input_real_shape[0], input_real_shape[1], \
                                                                        input_real_shape[2], input_real_shape[3], \
                                                                        input_real_shape[4], input_real_shape[-1]

    else:
        raise ValueError("Input Dimension must be 5 or 6!")
    input_size = [time_in_, height_in_, width_in_, depth_in_]
    size_in = [int(np.prod(np.array(input_real_shape)))]
    input_real, input_imag = tf.reshape(input_real, size_in), tf.reshape(input_imag, size_in)

    out_size = calculate_out_size(input_size, pool_size, strides, dilations_rate, mode, pad, ceil_mode)
    out_real, out_imag, out_idx = op(input_real, input_imag,
                                     pool_size[0], pool_size[1], pool_size[2], pool_size[3], strides[0],
                                     strides[1],
                                     strides[2], strides[3], pad[0], pad[1], pad[2],
                                     pad[3], alpha, beta, name=name, dilation_rate_t=dilations_rate[0],
                                     dilation_rate_h=dilations_rate[1],
                                     dilation_rate_w=dilations_rate[2], dilation_rate_d=dilations_rate[3],
                                     mode=mode, ceil_mode=ceil_mode,
                                     batch=batch_, time_in=time_in_, height_in=height_in_, width_in=width_in_,
                                     depth_in=depth_in_, channels=channels_)

    time_out, height_out, width_out, depth_out = int(out_size[0]), int(out_size[1]), int(out_size[2]), int(out_size[3])
    if len(input_real_shape) == 5:
        size_out = [time_out, height_out, width_out, depth_out, channels_]
    elif len(input_real_shape) == 6:
        size_out = [batch_, time_out, height_out, width_out, depth_out, channels_]
    else:
        raise ValueError("Input Dimension must be 5 or 6!")
    print('line347', size_out, out_real.shape)
    out_real, out_imag = tf.reshape(out_real, size_out), tf.reshape(out_imag, size_out)

    if channel_first:
        [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                            change_back=True, dimensions='4d')

    return tf.complex(out_real, out_imag)


@_ops.RegisterGradient("Maxpooling4d")
def grad(op, grad1, grad2, grad3):
    op_backward = _ext.maxpooling4d_grad_backward
    top_diff_real, top_diff_imag = tf.math.real(grad1), tf.math.imag(grad2)
    kernel_t = op.get_attr("kernel_t")
    kernel_h = op.get_attr("kernel_h")
    kernel_w = op.get_attr("kernel_w")
    kernel_d = op.get_attr("kernel_d")

    stride_t = op.get_attr("stride_t")
    stride_h = op.get_attr("stride_h")
    stride_w = op.get_attr("stride_w")
    stride_d = op.get_attr("stride_d")

    alpha = op.get_attr("alpha")
    beta = op.get_attr("beta")
    dilation_rate_t = op.get_attr("dilation_rate_t")
    dilation_rate_h = op.get_attr("dilation_rate_h")
    dilation_rate_w = op.get_attr("dilation_rate_w")
    dilation_rate_d = op.get_attr("dilation_rate_d")
    mode = op.get_attr("mode")
    ceil_mode = op.get_attr("ceil_mode")
    pad_t = op.get_attr("pad_t")
    pad_h = op.get_attr("pad_h")
    pad_w = op.get_attr("pad_w")
    pad_d = op.get_attr("pad_d")


    batch = op.get_attr("batch")
    time_in = op.get_attr("time_in")
    height_in = op.get_attr("height_in")
    width_in = op.get_attr("width_in")
    depth_in = op.get_attr("depth_in")
    channels = op.get_attr("channels")

    bottom_diff_real, bottom_diff_imag = op_backward(input_real=op.inputs[0],
                                                     input_imag=op.inputs[1],
                                                     output_real=op.outputs[0],
                                                     output_imag=op.outputs[1],
                                                     top_diff_real=top_diff_real,
                                                     top_diff_imag=top_diff_imag,
                                                     indices=op.outputs[2],
                                                     kernel_t=kernel_t,
                                                     kernel_h=kernel_h,
                                                     kernel_w=kernel_w,
                                                     kernel_d=kernel_d,
                                                     stride_t=stride_t,
                                                     stride_h=stride_h,
                                                     stride_w=stride_w,
                                                     stride_d=stride_d,
                                                     pad_t=pad_t,
                                                     pad_h=pad_h,
                                                     pad_w=pad_w,
                                                     pad_d=pad_d,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     dilation_rate_t=dilation_rate_t,
                                                     dilation_rate_h=dilation_rate_h,
                                                     dilation_rate_w=dilation_rate_w,
                                                     dilation_rate_d=dilation_rate_d,
                                                     batch=batch,
                                                     time_in=time_in,
                                                     height_in=height_in,
                                                     width_in=width_in,
                                                     depth_in=depth_in,
                                                     channels=channels,
                                                     mode=mode,
                                                     with_indices=True,
                                                     ceil_mode=ceil_mode)

    return bottom_diff_real, bottom_diff_imag
