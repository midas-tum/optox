import numpy as np
import torch
import unittest
import _ext.th_averagepooling_operator

__all__ = ['float_2d', 'double_2d', 'float_3d', 'double_3d']

float_1d = _ext.th_averagepooling_operator.AveragePooling1d_float
double_1d = _ext.th_averagepooling_operator.AveragePooling1d_double

float_2d = _ext.th_averagepooling_operator.AveragePooling2d_float
double_2d = _ext.th_averagepooling_operator.AveragePooling2d_double

float_3d = _ext.th_averagepooling_operator.AveragePooling3d_float
double_3d = _ext.th_averagepooling_operator.AveragePooling3d_double

float_4d = _ext.th_averagepooling_operator.AveragePooling4d_float
double_4d = _ext.th_averagepooling_operator.AveragePooling4d_double


def transpose_channel_first_last(inputs=None, change_back=False, dimensions='2d'):
    outputs = []
    if change_back == False:
        for x in inputs:
            if x.ndim == 2:
                # (c,h)->(h, c)
                x = torch.permute(x, (1, 0))
            elif x.ndim == 3 and dimensions == '1d':
                # (b,c,h)->(b,h,c)
                x = torch.permute(x, (0, 2, 1))
            elif x.ndim == 3 and dimensions == '2d':
                # (c,h,w)->(h,w,c)
                x = torch.permute(x, (1, 2, 0))
            elif x.ndim == 4 and dimensions == '2d':
                # (b,c,h,w)->(b,h,w,c)
                x = torch.permute(x, (0, 2, 3, 1))
            elif x.ndim == 4 and dimensions == '3d':
                # (c, h, w, d)->(h, w, d, c)
                x = torch.permute(x, (1, 2, 3, 0))
            elif x.ndim == 5 and dimensions == '3d':
                # (b,c,h,w,d)->(b,h,w,d,c)
                x = torch.permute(x, (0, 2, 3, 4, 1))
            elif x.ndim == 5 and dimensions == '4d':
                # (c,t,h,w,d)->(t,h,w,d,c)
                x = torch.permute(x, (1, 2, 3, 4, 0))
            elif x.ndim == 6:
                # (b,c,t,h,w,d)->(b,t,h,w,d,c)
                x = torch.permute(x, (0, 2, 3, 4, 5, 1))

            else:
                raise Exception('input dimension should be 2, 3, 4, 5 or 6!')
            outputs.append(x)
    else:
        for x in inputs:
            if x.ndim == 2:
                # (h,c)->(c,h)
                x = torch.permute(x, (1, 0))
            elif x.ndim == 3 and dimensions == '1d':
                # (b,h,c)->(b,c,h)
                x = torch.permute(x, (0, 2, 1))
            elif x.ndim == 3 and dimensions == '2d':
                # (h,w,c)->(c,h,w)
                x = torch.permute(x, (2, 0, 1))
            elif x.ndim == 4 and dimensions == '2d':
                # (b,h,w,c)->(b,c,h,w)
                x = torch.permute(x, (0, 3, 1, 2))
            elif x.ndim == 4 and dimensions == '3d':
                # (h,w,d,c)->(c,h,w,d)
                x = torch.permute(x, (3, 0, 1, 2))
            elif x.ndim == 5 and dimensions == '3d':
                # (b,h,w,d,c)->(b,c,h,w,d)
                x = torch.permute(x, (0, 4, 1, 2, 3))
            elif x.ndim == 5 and dimensions == '4d':
                # (t,h,w,d,c)->(c,t,h,w,d)
                x = torch.permute(x, (4, 0, 1, 2, 3))
            elif x.ndim == 6:
                # (b,t,h,w,d,c)->(b,c,t,h,w,d)
                x = torch.permute(x, (0, 5, 1, 2, 3, 4))
            else:
                raise Exception('input dimension should be 2, 3, 4, 5 or 6!')
            outputs.append(x)
    return outputs


def padding_shape(input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
    if padding_mode.lower() == 'valid':
        return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
    elif padding_mode.lower() == 'same':
        return np.ceil(input_spatial_shape / strides)
    else:
        raise Exception('padding_mode can be only valid or same!')


def get_operator_1d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0]
    if dilations_rate is None:
        dilations_rate = [1]
    if len(shape_in) == 2:
        batch = 0
        if dtype == torch.float32:
            return float_1d(shape_in[0], pool_size[0], strides[0],
                            channels, alpha, beta, pads[0], dilations_rate[0], batch,
                            int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_1d(shape_in[0], pool_size[0], strides[0],
                             channels, alpha, beta, pads[0], dilations_rate[0], batch,
                             int(ceil_mode), mode)
    elif len(shape_in) == 3:
        batch = shape_in[0]
        if dtype == torch.float32:
            return float_1d(shape_in[1], pool_size[0], strides[0],
                            channels, alpha, beta, pads[0], dilations_rate[0], batch,
                            int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_1d(shape_in[1], pool_size[0], strides[0],
                             channels, alpha, beta, pads[0], dilations_rate[0], batch,
                             int(ceil_mode),  mode)

    else:
        raise RuntimeError('Invalid dtype!')


def get_operator_2d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0, 0]
    if dilations_rate is None:
        dilations_rate = [1, 1]
    if len(shape_in) == 3:
        batch = 0
        if dtype == torch.float32:
            return float_2d(shape_in[0], shape_in[1], pool_size[0], pool_size[1], strides[0], strides[1],
                            channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                            int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_2d(shape_in[0], shape_in[1], pool_size[0], pool_size[1], strides[0], strides[1],
                             channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                             int(ceil_mode), mode)
    elif len(shape_in) == 4:
        batch = shape_in[0]
        if dtype == torch.float32:
            return float_2d(shape_in[1], shape_in[2], pool_size[0], pool_size[1], strides[0], strides[1],
                            channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                            int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_2d(shape_in[1], shape_in[2], pool_size[0], pool_size[1], strides[0], strides[1],
                             channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                             int(ceil_mode), mode)

    else:
        raise RuntimeError('Invalid dtype!')


def get_operator_3d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0, 0, 0]
    if dilations_rate is None:
        dilations_rate = [1, 1, 1]
    if len(shape_in) == 4:
        batch = 0
        if dtype == torch.float32:
            return float_3d(shape_in[0], shape_in[1], shape_in[2], pool_size[0], pool_size[1], pool_size[2],
                            strides[0], strides[1], strides[2],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                            int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_3d(shape_in[0], shape_in[1], shape_in[2], pool_size[0], pool_size[1], pool_size[2],
                             strides[0], strides[1], strides[2],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                             int(ceil_mode), mode)
    elif len(shape_in) == 5:
        batch = shape_in[0]
        if dtype == torch.float32:
            return float_3d(shape_in[1], shape_in[2], shape_in[3], pool_size[0], pool_size[1], pool_size[2],
                            strides[0], strides[1], strides[2],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                            int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_3d(shape_in[1], shape_in[2], shape_in[3], pool_size[0], pool_size[1], pool_size[2],
                             strides[0], strides[1], strides[2],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                             int(ceil_mode), mode)
    else:
        raise RuntimeError('Invalid dtype!')


def get_operator_4d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0, 0, 0, 0]
    if dilations_rate is None:
        dilations_rate = [1, 1, 1, 1]
    if len(shape_in) == 5:
        batch = 0
        if dtype == torch.float32:
            return float_4d(shape_in[0], shape_in[1], shape_in[2], shape_in[3],
                            pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                            strides[0], strides[1], strides[2], strides[3],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], pads[3],
                            dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                            batch, int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_4d(shape_in[0], shape_in[1], shape_in[2], shape_in[3],
                             pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                             strides[0], strides[1], strides[2], strides[3],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], pads[3],
                             dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                             batch, int(ceil_mode), mode)
    elif len(shape_in) == 6:
        batch = shape_in[0]
        if dtype == torch.float32:
            return float_4d(shape_in[1], shape_in[2], shape_in[3], shape_in[4],
                            pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                            strides[0], strides[1], strides[2], strides[3],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], pads[3],
                            dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                            batch, int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_4d(shape_in[1], shape_in[2], shape_in[3], shape_in[4],
                             pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                             strides[0], strides[1], strides[2], strides[3],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], pads[3],
                             dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                             batch, int(ceil_mode), mode)
    else:
        raise RuntimeError('Invalid dtype!')


class Averagepooling1dFunction(torch.autograd.Function):
    """ Average-pooling of a 1d tensor.

    Forward:
    This function downsamples a 1d tensor (rank 3) according their average values. the downsized value is defined by
    average value= (alpha*sum_real_in_pool_size +beta*sum_imag_in_pool_size)/num_in_pool_size
    The tensorformat is  [N, H, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)
    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1
    Args:
        x_input: A 3D `Tensor`.
        pool_size: tuple of 1 integers, window size over which to take the average.
        pads: tuple of 1 integer.
        strides: tuple/list of 1 integer, specifying the strides of the convolution along the height and width.
             Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 1 integer, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginary value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).
    Returns:
        A  complex `Tensor`which has the same type as inputs.

    Backward:
    Find backward average-pooling gradient of a 1d tensor.
    Only the gradient value with average-pooled indices can be transferred to the bottom.
    Args:
        ctx: A context object storing information from the forward path:
            x_input: A 3D `Tensor`.
            x_output:  A 3D `Tensor` outputs from averagepooling1d.
        grad_out: Backward flowing gradient as a 3D `Tensor`.
    Returns:
        bottom_diff : A complex 3D `Tensor` gradient from bottom (x_input side)
        the type is same as x_input.
    """
    @staticmethod
    def forward(ctx, x_input, pool_size, pads, strides, dilations_rate,
                alpha=1, beta=1, padding_mode='SAME', dtype=torch.float32, channel_first=False, ceil_mode=True):
        ctx.saved_param = pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first

        if channel_first:
            [x_input] = transpose_channel_first_last([x_input], dimensions='1d')
        ctx.op = get_operator_1d(dtype, x_input.shape, pool_size=pool_size, strides=strides,
                                 pads=pads, alpha=1, beta=1, dilations_rate=dilations_rate,
                                 batch=0, mode=padding_mode, ceil_mode=ceil_mode)
        ctx.shape = x_input.shape
        if x_input.dtype == torch.complex64 or x_input.dtype == torch.complex128:
            x_real = x_input.real.contiguous()
            x_imag = x_input.imag.contiguous()
            out_real, out_imag = ctx.op.forward(x_real, x_imag)
            if channel_first:
                [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                                    change_back=True, dimensions='1d')

            return torch.complex(out_real, out_imag)

        else:
            # real inputs
            x_real = x_input.contiguous()
            out_real, _ = ctx.op.forward(x_real, x_real * 0)
            if channel_first:
                [out_real] = transpose_channel_first_last([out_real], change_back=True,
                                                          dimensions='1d')

            return out_real

    @staticmethod
    def backward(ctx, grad_out):
        pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first = ctx.saved_param
        if channel_first:
            [grad_out] = transpose_channel_first_last([grad_out], dimensions='1d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            top_diff_real = grad_out.real.contiguous()
            top_diff_imag = grad_out.imag.contiguous()
        else:
            top_diff_real = grad_out.contiguous()
            top_diff_imag = grad_out.contiguous()

        inputs = torch.zeros(ctx.shape, dtype=dtype).cuda().contiguous()
        bottom_diff_real, bottom_diff_imag = ctx.op.adjoint(inputs, inputs,
                                                            inputs, inputs,
                                                            top_diff_real, top_diff_imag)
        if channel_first:
            [bottom_diff_real, bottom_diff_imag] = transpose_channel_first_last(
                [bottom_diff_real, bottom_diff_imag], change_back=True,
                dimensions='1d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            return torch.complex(bottom_diff_real,
                                 bottom_diff_imag), None, None, None, None, None, None, None, None, None, None
        else:
            return bottom_diff_real, None, None, None, None, None, None, None, None, None, None


class Averagepooling2dFunction(torch.autograd.Function):
    """ Average-pooling of a 2d tensor.

    Forward:
    This function downsamples a 2d tensor (rank 4) according their average values. the downsized value is defined by
    average value= (alpha*sum_real_in_pool_size +beta*sum_imag_in_pool_size)/num_in_pool_size
    The tensorformat is  [N, H, W, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)
    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1
    Args:
        x_input: A 4D `Tensor`.
        pool_size: tuple of 2 integers, window size over which to take the average.
        pads: tuple of 2 integers.
        strides: tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
             Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginary value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).
    Returns:
        A  complex `Tensor`which has the same type as inputs.

    Backward:
    Find backward average-pooling gradient of a 2d tensor.
    Only the gradient value with average-pooled indices can be transferred to the bottom.
    Args:
        ctx: A context object storing information from the forward path:
            x_input: A 4D `Tensor`.
            x_output:  A 4D `Tensor` outputs from averagepooling2d.
        grad_out: Backward flowing gradient as a 4D `Tensor`.
    Returns:
        bottom_diff : A complex 4D `Tensor` gradient from bottom (x_input side)
        the type is same as x_input.
    """
    @staticmethod
    def forward(ctx, x_input, pool_size, pads, strides, dilations_rate,
                alpha=1, beta=1, padding_mode='SAME', dtype=torch.float32, channel_first=False, ceil_mode=True):
        ctx.saved_param = pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first

        if channel_first:
            [x_input] = transpose_channel_first_last([x_input], dimensions='2d')
        
        ctx.op = get_operator_2d(dtype, x_input.shape, pool_size=pool_size, strides=strides,
                                 pads=pads, alpha=1, beta=1, dilations_rate=dilations_rate,
                                 batch=0, mode=padding_mode,ceil_mode=ceil_mode)
        ctx.shape = x_input.shape
        if x_input.dtype == torch.complex64 or x_input.dtype == torch.complex128:
            x_real = x_input.real.contiguous()
            x_imag = x_input.imag.contiguous()
            out_real, out_imag = ctx.op.forward(x_real, x_imag)
            if channel_first:
                [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                                    change_back=True, dimensions='2d')

            return torch.complex(out_real, out_imag)

        else:
            # real inputs
            x_real = x_input.contiguous()
            out_real, _ = ctx.op.forward(x_real, x_real * 0)
            if channel_first:
                [out_real] = transpose_channel_first_last([out_real], change_back=True,
                                                          dimensions='2d')

            return out_real

    @staticmethod
    def backward(ctx, grad_out):
        pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first = ctx.saved_param
        if channel_first:
            [grad_out] = transpose_channel_first_last([grad_out], dimensions='2d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            top_diff_real = grad_out.real.contiguous()
            top_diff_imag = grad_out.imag.contiguous()
        else:
            top_diff_real = grad_out.contiguous()
            top_diff_imag = grad_out.contiguous()

        inputs = torch.zeros(ctx.shape, dtype=dtype).cuda().contiguous()
        bottom_diff_real, bottom_diff_imag = ctx.op.adjoint(inputs, inputs,
                                                            inputs, inputs,
                                                            top_diff_real, top_diff_imag)
        if channel_first:
            [bottom_diff_real, bottom_diff_imag] = transpose_channel_first_last(
                [bottom_diff_real, bottom_diff_imag], change_back=True,
                dimensions='2d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            return torch.complex(bottom_diff_real,
                                 bottom_diff_imag), None, None, None, None, None, None, None, None, None, None
        else:
            return bottom_diff_real, None, None, None, None, None, None, None, None, None, None


class Averagepooling3dFunction(torch.autograd.Function):
    """ Average pooling of a 3d tensor.

    Forward:
    This function downsamples a 3d tensor (rank 5) according their average values. the downsized value is defined by
    average value= (alpha*sum_real_in_pool_size +beta*sum_imag_in_pool_size)/num_in_pool_size
    The tensorformat is  [N, H, W, D, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)
    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1
    Args:
        x_input: A  input 5D `Tensor`.
        pool_size: tuple of 3 integers, window size over which to take the average.
        pads: tuple of 3 integers.
        strides: tuple/list of 3 integers, specifying the strides of the convolution along the height and width.
             Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginary value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).
    Returns:
        A complex `Tensor`which has the same type as inputs.

    Backward:
    Find backward average-pooling gradient of a 3d tensor.
    Only the gradient value with average-pooled indices can be transferred to the bottom.
    Args:
        ctx: A context object storing information from the forward path:
            x_input: A 5D `Tensor`.
            x_output:  A 5D `Tensor` outputs from averagepooling3d.
        grad_out: Backward flowing gradient as a 5D `Tensor`.
    Returns:
        bottom_diff : A complex 5D `Tensor` gradient from bottom (x_input side)
        the type is same as x_input.
    """
    @staticmethod
    def forward(ctx, x_input, pool_size, pads, strides, dilations_rate,
                alpha=1, beta=1, padding_mode='SAME', dtype=torch.float32, channel_first=False, ceil_mode=True):
        ctx.saved_param = pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first

        if channel_first:
            [x_input] = transpose_channel_first_last([x_input], dimensions='3d')
        ctx.op = get_operator_3d(dtype, x_input.shape, pool_size=pool_size, strides=strides,
                                 pads=pads, alpha=1, beta=1, dilations_rate=dilations_rate,
                                 batch=0, mode=padding_mode, ceil_mode=ceil_mode)
        ctx.shape = x_input.shape
        if x_input.dtype == torch.complex64 or x_input.dtype == torch.complex128:
            x_real = x_input.real.contiguous()
            x_imag = x_input.imag.contiguous()
            out_real, out_imag = ctx.op.forward(x_real, x_imag)
            if channel_first:
                [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                                    change_back=True, dimensions='3d')

            return torch.complex(out_real, out_imag)

        else:
            # real inputs
            x_real = x_input.contiguous()
            out_real, _ = ctx.op.forward(x_real, x_real * 0)
            if channel_first:
                [out_real] = transpose_channel_first_last([out_real], change_back=True,
                                                          dimensions='3d')

            return out_real

    @staticmethod
    def backward(ctx, grad_out):
        pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first = ctx.saved_param
        if channel_first:
            [grad_out] = transpose_channel_first_last([grad_out], dimensions='3d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            top_diff_real = grad_out.real.contiguous()
            top_diff_imag = grad_out.imag.contiguous()
        else:
            top_diff_real = grad_out.contiguous()
            top_diff_imag = grad_out.contiguous()

        inputs = torch.zeros(ctx.shape, dtype=dtype).cuda().contiguous()
        bottom_diff_real, bottom_diff_imag = ctx.op.adjoint(inputs, inputs,
                                                            inputs, inputs,
                                                            top_diff_real, top_diff_imag)
        if channel_first:
            [bottom_diff_real, bottom_diff_imag] = transpose_channel_first_last(
                [bottom_diff_real, bottom_diff_imag], change_back=True,
                dimensions='3d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            return torch.complex(bottom_diff_real,
                                 bottom_diff_imag), None, None, None, None, None, None, None, None, None, None
        else:
            return bottom_diff_real, None, None, None, None, None, None, None, None, None, None


class Averagepooling4dFunction(torch.autograd.Function):
    """ Average pooling of a 4d tensor.
    Forward:
    This function downsamples a 4d tensor (rank 6) according their average values. the downsized value is defined by
    average value= (alpha*sum_real_in_pool_size +beta*sum_imag_in_pool_size)/num_in_pool_size
    The tensorformat is  [N, T, H, W, D, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)
    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1
    Args:
        x_input: A  input 6D `Tensor`.
        pool_size: tuple of 4 integers, window size over which to take the average.
        pads: tuple of 4 integers.
        strides: tuple/list of 4 integers, specifying the strides of the convolution along the height and width.
             Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 4 integers, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginary value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).
    Returns:
        A complex `Tensor`which has the same type as inputs.
    Backward:
    Find backward average-pooling gradient of a 4d tensor.
    Only the gradient value with average-pooled indices can be transferred to the bottom.
    Args:
        ctx: A context object storing information from the forward path:
            x_input: A 6D `Tensor`.
            x_output:  A 6D `Tensor` outputs from averagepooling4d.
        grad_out: Backward flowing gradient as a 6D `Tensor`.
    Returns:
        bottom_diff : A complex 6D `Tensor` gradient from bottom (x_input side)
        the type is same as x_input.
    """
    @staticmethod
    def forward(ctx, x_input, pool_size, pads, strides, dilations_rate,
                alpha=1, beta=1, padding_mode='SAME', dtype=torch.float32, channel_first=False, ceil_mode=True):
        ctx.saved_param = pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first
        if channel_first:
            [x_input] = transpose_channel_first_last([x_input], dimensions='4d')
        ctx.op = get_operator_4d(dtype, x_input.shape, pool_size=pool_size, strides=strides,
                                 pads=pads, alpha=1, beta=1, dilations_rate=dilations_rate,
                                 batch=0, mode=padding_mode, ceil_mode=ceil_mode)
        ctx.shape = x_input.shape
        if x_input.dtype == torch.complex64 or x_input.dtype == torch.complex128:
            x_real = x_input.real.contiguous()
            x_imag = x_input.imag.contiguous()

            x_real_shape = x_real.shape
            size_in = (int(np.prod(np.array(x_real_shape))),)
            x_real, x_imag = torch.reshape(x_real, size_in), torch.reshape(x_imag, size_in)

            out_real, out_imag, time_out, height_out, width_out, depth_out = ctx.op.forward(x_real, x_imag)
            if len(x_real_shape) == 5:
                size_out = (time_out, height_out, width_out, depth_out, x_real_shape[-1])
            elif len(x_real_shape) == 6:
                size_out = (x_real_shape[0], time_out, height_out, width_out, depth_out, x_real_shape[-1])
            else:
                raise ValueError("Input Dimension must be 5 or 6!")
            out_real, out_imag = torch.reshape(out_real, size_out), torch.reshape(out_imag, size_out)
            if channel_first:
                [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                                    change_back=True, dimensions='4d')
            return torch.complex(out_real, out_imag)

        else:
            # real inputs
            x_input = x_input.contiguous()
            x_real_shape = x_input.shape
            size_in = (int(np.prod(np.array(x_real_shape))),)
            x_real = torch.reshape(x_input, size_in).contiguous()
            out_real, _, time_out, height_out, width_out, depth_out = ctx.op.forward(x_real, x_real * 0)
            if len(x_real_shape) == 5:
                size_out = (time_out, height_out, width_out, depth_out, x_real_shape[-1])
            elif len(x_real_shape) == 6:
                size_out = (x_real_shape[0], time_out, height_out, width_out, depth_out, x_real_shape[-1])
            else:
                raise ValueError("Input Dimension must be 5 or 6!")
            out_real = torch.reshape(out_real, size_out)

            if channel_first:
                [out_real] = transpose_channel_first_last([out_real], change_back=True,
                                                          dimensions='4d')

            return out_real

    @staticmethod
    def backward(ctx, grad_out):
        pool_size, pads, strides, dilations_rate, alpha, beta, padding_mode, dtype, channel_first = ctx.saved_param
        if channel_first:
            [grad_out] = transpose_channel_first_last([grad_out], dimensions='4d')
            
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            top_diff_real = grad_out.real.contiguous()
            top_diff_imag = grad_out.imag.contiguous()
        else:
            top_diff_real = grad_out.contiguous()
            top_diff_imag = grad_out.contiguous()
        x_input_shape = ctx.shape
        size_in = (int(np.prod(np.array(x_input_shape))),)
        inputs = torch.zeros(size_in, dtype=dtype).cuda().contiguous()
        x_output_shape = grad_out.shape
        size_out = (int(np.prod(np.array(x_output_shape))),)
        top_diff_real, top_diff_imag = torch.reshape(top_diff_real, size_out), torch.reshape(
            top_diff_imag, size_out)

        bottom_diff_real, bottom_diff_imag = ctx.op.adjoint(inputs , inputs ,
                                                            inputs , inputs ,
                                                            top_diff_real, top_diff_imag)

        bottom_diff_real, bottom_diff_imag = torch.reshape(bottom_diff_real, x_input_shape), torch.reshape(
            bottom_diff_imag, x_input_shape)

        if channel_first:
            [bottom_diff_real, bottom_diff_imag] = transpose_channel_first_last(
                [bottom_diff_real, bottom_diff_imag], change_back=True,
                dimensions='4d')
        if grad_out.dtype == torch.complex64 or grad_out.dtype == torch.complex128:
            return torch.complex(bottom_diff_real,
                                 bottom_diff_imag), None, None, None, None, None, None, None, None, None, None
        else:
            return bottom_diff_real, None, None, None, None, None, None, None, None, None, None
