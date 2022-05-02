import numpy as np
import unittest

import _ext.py_averagepooling_operator

__all__ = ['float_2d', 'double_2d', 'float_3d', 'double_3d']

float_2d = _ext.py_averagepooling_operator.AveragePooling2d_float
double_2d = _ext.py_averagepooling_operator.AveragePooling2d_double

float_3d = _ext.py_averagepooling_operator.AveragePooling3d_float
double_3d = _ext.py_averagepooling_operator.AveragePooling3d_double
def transepose_channel_first_last(inputs=None,change_back=False, dimensions='2d'):
    outputs=[]
    if change_back==False:
        for x in inputs:
            if x.ndim == 3 :
                # (c,h,w)->(h,w,c)
                x = np.transpose(x, (1, 2, 0))
            elif x.ndim == 4 and dimensions=='2d':
                # (b,c,h,w)->(b,h,w,c)
                x = np.transpose(x, (0, 2, 3, 1))
            elif x.ndim == 4 and dimensions=='3d':
                # (c, h, w, d)->(h, w, d, c)
                x = np.transpose(x, (1, 2, 3, 0))
            elif x.ndim == 5 :
                # (b,c,h,w,d)->(b,h,w,d,c)
                x = np.transpose(x, (0, 2, 3, 4, 1))
            else:
                raise Exception ('input dimension should be 3, 4, or 5!')
            outputs.append(x)
        else:
            for x in inputs:
                if x.ndim == 3 :
                    # (h,w,c)->(c,h,w)
                    x = np.transpose(x, (2, 0, 1))
                elif x.ndim == 4 and  dimensions=='2d':
                    # (b,h,w,c)->(b,c,h,w)
                    x = np.transpose(x, (0, 3, 1, 2))
                elif x.ndim == 4 and  dimensions=='3d':
                    # (h,w,d,c)->(c,h,w,d)
                    x = np.transpose(x, (3, 0, 1, 2))
                elif x.ndim == 5 :
                    # (b,h,w,d,c)->(b,c,h,w,d)
                    x = np.transpose(x, (0, 4, 1, 2, 3))
                else:
                    raise Exception ('input dimension should be 3, 4, or 5!')
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
                    mode='same', with_indices=False, ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0]
    if dilations_rate is None:
        dilations_rate = [1]
    if len(shape_in) == 2:
        batch = 0
        if dtype == np.float32:
            return float_1d(shape_in[0], pool_size[0], strides[0],
                            channels, alpha, beta, pads[0], dilations_rate[0], batch,
                               int(ceil_mode), mode)
        elif dtype == np.float64:
            return double_1d(shape_in[0], pool_size[0], strides[0],
                             channels, alpha, beta, pads[0], dilations_rate[0], batch,
                               int(ceil_mode), mode)
    elif len(shape_in) == 3:
        batch = shape_in[0]
        if dtype == np.float32:
            return float_1d(shape_in[1], pool_size[0], strides[0],
                            channels, alpha, beta, pads[0], dilations_rate[0], batch,
                              int(ceil_mode), mode)
        elif dtype == np.float64:
            return double_1d(shape_in[1], pool_size[0], strides[0],
                             channels, alpha, beta, pads[0], dilations_rate[0], batch,
                               int(ceil_mode), mode)

    else:
        raise RuntimeError('Invalid dtype!')

def get_operator_2d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', with_indices=False, ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0, 0]
    if dilations_rate is None:
        dilations_rate = [1, 1]
    if len(shape_in) == 3:
        batch = 0
        if dtype == np.float32:
            return float_2d(shape_in[0], shape_in[1], pool_size[0], pool_size[1], strides[0], strides[1],
                            channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                               int(ceil_mode), mode)
        elif dtype == np.float64:
            return double_2d(shape_in[0], shape_in[1], pool_size[0], pool_size[1], strides[0], strides[1],
                             channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                               int(ceil_mode),mode )
    elif len(shape_in) == 4:
        batch = shape_in[0]
        if dtype == np.float32:
            return float_2d(shape_in[1], shape_in[2], pool_size[0], pool_size[1], strides[0], strides[1],
                            channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                              int(ceil_mode), mode )
        elif dtype == np.float64:
            return double_2d(shape_in[1], shape_in[2], pool_size[0], pool_size[1], strides[0], strides[1],
                             channels, alpha, beta, pads[0], pads[1], dilations_rate[0], dilations_rate[1], batch,
                              int(ceil_mode), mode )

    else:
        raise RuntimeError('Invalid dtype!')


def get_operator_3d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', with_indices=False,ceil_mode=True):
    channels = shape_in[-1]
    if pads is None:
        pads = [0, 0, 0]
    if dilations_rate is None:
        dilations_rate = [1, 1, 1]
    if len(shape_in) == 4:
        batch = 0
        if dtype == np.float32:
            return float_3d(shape_in[0], shape_in[1], shape_in[2], pool_size[0], pool_size[1], pool_size[2],
                            strides[0], strides[1], strides[2],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                              int(ceil_mode), mode )
        elif dtype == np.float64:
            return double_3d(shape_in[0], shape_in[1], shape_in[2], pool_size[0], pool_size[1], pool_size[2],
                             strides[0], strides[1], strides[2],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                              int(ceil_mode), mode )
    elif len(shape_in) == 5:
        batch = shape_in[0]
        if dtype == np.float32:
            return float_3d(shape_in[1], shape_in[2], shape_in[3], pool_size[0], pool_size[1], pool_size[2],
                            strides[0], strides[1], strides[2],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                              int(ceil_mode), mode )
        elif dtype == np.float64:
            return double_3d(shape_in[1], shape_in[2], shape_in[3], pool_size[0], pool_size[1], pool_size[2],
                             strides[0], strides[1], strides[2],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], dilations_rate[0], dilations_rate[1], dilations_rate[2], batch,
                               int(ceil_mode), mode )
    else:
        raise RuntimeError('Invalid dtype!')


def get_operator_4d(dtype, shape_in, pool_size, strides, alpha, beta, pads=None, dilations_rate=None, batch=0,
                    mode='same', with_indices=False,ceil_mode=True):
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
                            batch,  int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_4d(shape_in[0], shape_in[1], shape_in[2], shape_in[3],
                             pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                             strides[0], strides[1], strides[2], strides[3],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], pads[3],
                             dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                             batch,   int(ceil_mode), mode)
    elif len(shape_in) == 6:
        batch = shape_in[0]
        if dtype == torch.float32:
            return float_4d(shape_in[0], shape_in[1], shape_in[2], shape_in[3],
                            pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                            strides[0], strides[1], strides[2], strides[3],
                            channels, alpha, beta,
                            pads[0], pads[1], pads[2], pads[3],
                            dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                            batch,   int(ceil_mode), mode)
        elif dtype == torch.float64:
            return double_4d(shape_in[0], shape_in[1], shape_in[2], shape_in[3],
                             pool_size[0], pool_size[1], pool_size[2], pool_size[3],
                             strides[0], strides[1], strides[2], strides[3],
                             channels, alpha, beta,
                             pads[0], pads[1], pads[2], pads[3],
                             dilations_rate[0], dilations_rate[1], dilations_rate[2], dilations_rate[3],
                             batch,    int(ceil_mode), mode)
    else:
        raise RuntimeError('Invalid dtype!')


class Averagepooling2d(object):
    """Max-pooling of a 2d tensor.

    This function downsamples a 2d tensor (rank 4) according their average values. the compared value is defined by
    value=alpha*real_value^2+beta*imag_value^2.
    The tensorformat is  [N, H, W, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)

    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1

    Args:
        inputs: A 4D `Tensor`.
        pool_size: tuple of 2 integers, window size over which to take the maximum.
        pads: tuple of 2 integers
        strides: tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginal value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).

    Returns:
        A complex `Tensor`which has the same type as x_input `tensor`.
    """

    def __init__(self, pool_size=(1, 1), pads=(0, 0), strides=(1, 1), dilations_rate=(1, 1), alpha=1, beta=0,
                 padding_mode='SAME', dtype=np.float32, channel_first=False,ceil_mode=True):

        self.pool_size = pool_size
        self.pads = pads
        self.strides = strides
        self.dilations_rate = dilations_rate
        self.alpha = alpha
        self.beta = beta
        self.padding_mode = padding_mode
        self.dtype = dtype
        self.channel_first = channel_first
        self.ceil_mode=ceil_mode

    def forward(self, x_input):
        if self.channel_first:
            [x_input] = transepose_channel_first_last([x_input], dimensions='2d')

        op = get_operator_2d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                             pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                             batch=0, mode=self.padding_mode, ceil_mode=self.ceil_mode)
        if np.iscomplexobj(x_input):
            x_real = np.ascontiguousarray(np.real(x_input))
            x_imag = np.ascontiguousarray(np.imag(x_input))
            out_real, out_imag = op.forward(x_real, x_imag)
            if self.channel_first:
                [out_real, out_imag] = transepose_channel_first_last([out_real, out_imag], change_back=True,
                                                                     dimensions='2d')

            return out_real + 1j * out_imag
        else:
            out_real, _ = op.forward(x_input, x_input * 0)
            if self.channel_first:
                [out_real] = transepose_channel_first_last([out_real], change_back=True, dimensions='2d')
            return out_real

    def backward(self, x_input, x_output, top_diff):
        if self.channel_first:
            [x_input, x_output, top_diff] = transepose_channel_first_last([x_input, x_output, top_diff],
                                                                          dimensions='2d')

        op = get_operator_2d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                             pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                             batch=0, mode=self.padding_mode)
        if np.iscomplexobj(x_input):
            x_real = np.ascontiguousarray(np.real(x_input))
            x_imag = np.ascontiguousarray(np.real(x_input))
            x_output_real = np.ascontiguousarray(np.real(x_output))
            x_output_imag = np.ascontiguousarray(np.imag(x_output))
            top_diff_real = np.ascontiguousarray(np.real(top_diff))
            top_diff_imag = np.ascontiguousarray(np.imag(top_diff))

            bottom_diff_real, bottom_diff_imag = op.adjoint(x_real, x_imag,
                                                            x_output_real, x_output_imag,
                                                            top_diff_real, top_diff_imag)

            if self.channel_first:
                [bottom_diff_real,bottom_diff_imag] = transepose_channel_first_last([bottom_diff_real,bottom_diff_imag], change_back=True,
                                                                              dimensions='2d')
            return bottom_diff_real + 1j * bottom_diff_imag
        else:
            bottom_diff_real, _ = op.adjoint(x_input, x_input * 0, x_output, x_output * 0,
                                             top_diff, x_output * 0)
            if self.channel_first:
                [bottom_diff_real] = transepose_channel_first_last([bottom_diff_real], change_back=True,
                                                                              dimensions='2d')
            return bottom_diff_real


class Averagepooling3d(object):
    """Max-pooling of a 3d tensor.

    This function downsamples a 3d tensor (rank 5) according their average values. the compared value is defined by
    value=alpha*real_value^2+beta*imag_value^2.
    The tensorformat is  [N, H, W, D, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)

    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1

    Args:
        x_input: A 5D `Tensor`.
        pool_size: tuple of 3 integers, window size over which to take the maximum.
        pads: tuple of 3 integers
        strides: tuple/list of 3 integers, specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginal value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).

    Returns:
        A  complex `Tensor`which has the same type as x_input `tensor`
    """

    def __init__(self, pool_size=(1, 1, 1), pads=(0, 0, 0), strides=(1, 1, 1), dilations_rate=(1, 1, 1), alpha=1,
                 beta=0, padding_mode='SAME', dtype=np.float32, channel_first=False,ceil_mode=True):

        self.pool_size = pool_size
        self.pads = pads
        self.strides = strides
        self.dilations_rate = dilations_rate
        self.alpha = alpha
        self.beta = beta
        self.padding_mode = padding_mode
        self.dtype = dtype
        self.channel_first = channel_first
        self.ceil_mode=ceil_mode

    def forward(self, x_input):
        if self.channel_first:
            [x_input] = transepose_channel_first_last([x_input], dimensions='3d')
        op = get_operator_3d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                             pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                             batch=0, mode=self.padding_mode)
        if np.iscomplexobj(x_input):
            x_real = np.ascontiguousarray(np.real(x_input))
            x_imag = np.ascontiguousarray(np.imag(x_input))
            out_real, out_imag = op.forward(x_real, x_imag)
            if self.channel_first:
                [out_real, out_imag] = transepose_channel_first_last([out_real, out_imag], change_back=True,
                                                                     dimensions='3d')

            return out_real + 1j * out_imag
        else:
            out_real,_=op.forward(x_input, x_input * 0)
            if self.channel_first:
                [out_real] = transepose_channel_first_last([out_real], change_back=True, dimensions='3d')
            return out_real

    def backward(self, x_input, x_output, top_diff):
        if self.channel_first:
            [x_input, x_output, top_diff] = transepose_channel_first_last([x_input, x_output, top_diff],
                                                                          dimensions='3d')
        op = get_operator_3d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                             pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                             batch=0, mode=self.padding_mode, ceil_mode=self.ceil_mode)

        if np.iscomplexobj(x_input):
            x_real = np.ascontiguousarray(np.real(x_input))
            x_imag = np.ascontiguousarray(np.real(x_input))
            x_output_real = np.ascontiguousarray(np.real(x_output))
            x_output_imag = np.ascontiguousarray(np.imag(x_output))
            top_diff_real = np.ascontiguousarray(np.real(top_diff))
            top_diff_imag = np.ascontiguousarray(np.imag(top_diff))

            bottom_diff_real, bottom_diff_imag = op.adjoint(x_real, x_imag,
                                                            x_output_real, x_output_imag,
                                                            top_diff_real, top_diff_imag)

            if self.channel_first:
                [bottom_diff_real, bottom_diff_imag] = transepose_channel_first_last(
                    [bottom_diff_real, bottom_diff_imag], change_back=True,dimensions='3d')

            return bottom_diff_real + 1j * bottom_diff_imag
        else:

            bottom_diff_real, _ = op.adjoint(x_input, x_input * 0,x_output, x_output*0,
                                                                top_diff, x_output*0)
            if self.channel_first:
                [bottom_diff_real] = transepose_channel_first_last([bottom_diff_real], change_back=True,
                                                                   dimensions='3d')
            return bottom_diff_real


class Averagepooling4d(object):
    """Average-pooling of a 4d tensor.

    This function downsamples a 4d tensor (rank 6) according their max values. the compared value is defined by
    value=alpha*real_value^2+beta*imag_value^2.
    The tensorformat is  [N, H, W, D, C].
    This functions supports the padding modes "valid", "same". The resulting output,
    when using the "valid" padding option, has a spatial shape (number of rows or columns) of:
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)

    The resulting output shape when using the "same" padding option is:
    output_shape = math.floor((input_shape - 1) / strides) + 1

    Args:
        x_input: A 5D `Tensor`.
        pool_size: tuple of 4 integers, window size over which to take the maximum.
        pads: tuple of 4 integers
        strides: tuple/list of 4 integers, specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for all spatial dimensions.
        dilations_rate:  tuple/list of 4 integers, specifying the dilation rate to use for dilated convolution.
        alpha: weight of real value.
        beta: weight of imaginal value.
        padding_mode: One of "VALID" or "SAME" (case-insensitive).

    Returns:
        A  complex `Tensor`which has the same type as x_input `tensor`
    """

    def __init__(self, pool_size=(1, 1, 1, 1), pads=(0, 0, 0, 0), strides=(1, 1, 1, 1), dilations_rate=(1, 1, 1, 1),
                 alpha=1,
                 beta=1, padding_mode='SAME', dtype=np.float32, channel_first=False, with_indices=False, ceil_mode=True):

        self.pool_size = pool_size
        self.pads = pads
        self.strides = strides
        self.dilations_rate = dilations_rate
        self.alpha = alpha
        self.beta = beta
        self.padding_mode = padding_mode
        self.dtype = dtype
        self.channel_first = channel_first
        self.with_indices = with_indices
        self.ceil_mode = ceil_mode

    def forward(self, x_input):
        if self.channel_first:
            [x_input] = transpose_channel_first_last([x_input], dimensions='4d')
        op = get_operator_4d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                             pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                             batch=0, mode=self.padding_mode,ceil_mode=self.ceil_mode)
        if np.iscomplexobj(x_input):

            x_real_shape = x_input.shape
            if len(x_real_shape) == 5:
                batch_, channels_ = 0, x_real_shape[-1]
            elif len(x_real_shape) == 6:
                batch_, channels_ = x_real_shape[0], x_real_shape[-1]
            else:
                raise ValueError("Input Dimension must be 5 or 6!")
            size_in = (int(np.prod(np.array(x_real_shape))),)
            x_input = np.reshape(x_input, size_in)
            x_real = np.ascontiguousarray(np.real(x_input))
            x_imag = np.ascontiguousarray(np.imag(x_input))

            out_real, out_imag, time_out, height_out, width_out, depth_out = op.forward(x_real, x_imag)

            if len(x_real_shape) == 5:
                size_out = (time_out, height_out, width_out, depth_out, channels_)
            elif len(x_real_shape) == 6:
                size_out = (batch_, time_out, height_out, width_out, depth_out, channels_)
            else:
                raise ValueError("Input Dimension must be 5 or 6!")

            out_real, out_imag = np.reshape(out_real, size_out), np.reshape(out_imag, size_out)

            if self.channel_first:
                [out_real, out_imag] = transpose_channel_first_last([out_real, out_imag],
                                                                             change_back=True, dimensions='4d')

            return out_real + 1j * out_imag
        else:
            x_real_shape = x_input.shape
            if len(x_real_shape) == 5:
                batch_, channels_ = 0, x_real_shape[-1]
            elif len(x_real_shape) == 6:
                batch_, channels_ = x_real_shape[0], x_real_shape[-1]
            else:
                raise ValueError("Input Dimension must be 5 or 6!")
            size_in = (int(np.prod(np.array(x_real_shape))),)
            x_input = np.reshape(x_input, size_in)
            x_input = np.ascontiguousarray(x_input)

            out_real, _, time_out, height_out, width_out, depth_out = op.forward(x_input, x_input * 0)

            if len(x_real_shape) == 5:
                size_out = (time_out, height_out, width_out, depth_out, channels_)
            elif len(x_real_shape) == 6:
                size_out = (batch_, time_out, height_out, width_out, depth_out, channels_)
            else:
                raise ValueError("Input Dimension must be 5 or 6!")

            out_real = np.reshape(out_real, size_out)

            if self.channel_first:
                [out_real] = transpose_channel_first_last([out_real], change_back=True,
                                                                   dimensions='4d')

            return out_real

    def backward(self, x_input, x_output, top_diff, indices=None, with_indices=False):
        self.with_indices = with_indices
        if self.with_indices == False:
            if self.channel_first:
                [x_input, x_output, top_diff] = transpose_channel_first_last([x_input, x_output, top_diff],
                                                                             dimensions='4d')
            op = get_operator_4d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                                 pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                                 batch=0, mode=self.padding_mode,ceil_mode=self.ceil_mode)
            if np.iscomplexobj(x_input):
                x_real = np.ascontiguousarray(np.real(x_input))
                x_imag = np.ascontiguousarray(np.real(x_input))
                x_output_real = np.ascontiguousarray(np.real(x_output))
                x_output_imag = np.ascontiguousarray(np.imag(x_output))
                top_diff_real = np.ascontiguousarray(np.real(top_diff))
                top_diff_imag = np.ascontiguousarray(np.imag(top_diff))

                x_real_shape = x_real.shape
                size_in = (int(np.prod(np.array(x_real_shape))),)
                x_real, x_imag = np.reshape(x_real, size_in), np.reshape(x_imag, size_in)

                top_diff_real_shape = top_diff_real.shape
                size_out = (int(np.prod(np.array(top_diff_real_shape))),)
                x_output_real, x_output_imag = np.reshape(x_output_real, size_out), np.reshape(x_output_imag, size_out)
                top_diff_real, top_diff_imag = np.reshape(top_diff_real, size_out), np.reshape(top_diff_imag,
                                                                                               size_out)

                bottom_diff_real, bottom_diff_imag = op.adjoint(x_real, x_imag,
                                                                x_output_real, x_output_imag,
                                                                top_diff_real, top_diff_imag, x_input * 0)

                bottom_diff_real, bottom_diff_imag = np.reshape(bottom_diff_real, x_real_shape), np.reshape(
                    bottom_diff_imag, x_real_shape)

                if self.channel_first:
                    [bottom_diff_real, bottom_diff_imag] = transpose_channel_first_last(
                        [bottom_diff_real, bottom_diff_imag], change_back=True, dimensions='4d')
                return bottom_diff_real + 1j * bottom_diff_imag
            else:

                x_real_shape = x_input.shape
                size_in = (int(np.prod(np.array(x_real_shape))),)
                x_input = np.reshape(x_input, size_in)

                top_diff_real_shape = top_diff.shape
                size_out = (int(np.prod(np.array(top_diff_real_shape))),)
                x_output = np.reshape(top_diff, size_out)
                top_diff = np.reshape(top_diff, size_out)

                bottom_diff_real, _ = op.adjoint(x_input, x_input * 0, x_output, x_output * 0,
                                                 top_diff, top_diff * 0, x_input * 0)

                bottom_diff_real = np.reshape(bottom_diff_real, x_real_shape)
                if self.channel_first:
                    [bottom_diff_real] = transpose_channel_first_last([bottom_diff_real], change_back=True,
                                                                      dimensions='4d')
                return bottom_diff_real
        else:
            indices_input = indices
            if self.channel_first:
                [x_input, indices_input, top_diff] = transpose_channel_first_last([x_input, indices_input, top_diff],
                                                                                  dimensions='4d')
            op = get_operator_4d(self.dtype, x_input.shape, pool_size=self.pool_size, strides=self.strides,
                                 pads=self.pads, alpha=self.alpha, beta=self.beta, dilations_rate=self.dilations_rate,
                                 batch=0, mode=self.padding_mode, with_indices=True,ceil_mode=self.ceil_mode)
            if np.iscomplexobj(top_diff):
                assert (np.iscomplexobj(x_input) == True)
                x_input = np.ascontiguousarray(np.real(x_input))
                indices_input = np.ascontiguousarray(np.imag(indices_input))
                top_diff_real = np.ascontiguousarray(np.real(top_diff))
                top_diff_imag = np.ascontiguousarray(np.imag(top_diff))

                x_input_shape = x_input.shape
                size_in = (int(np.prod(np.array(x_input_shape))),)
                x_input = np.reshape(x_input, size_in)

                top_diff_real_shape = top_diff_real.shape
                size_out = (int(np.prod(np.array(top_diff_real_shape))),)
                top_diff_real, top_diff_imag, indices_input = np.reshape(top_diff_real, size_out), np.reshape(
                    top_diff_imag, size_out), np.reshape(indices_input, size_out)

                bottom_diff_real, bottom_diff_imag = op.adjoint(x_input * 0, x_input * 0,
                                                                x_input * 0, x_input * 0,
                                                                top_diff_real, top_diff_imag, indices_input)

                bottom_diff_real, bottom_diff_imag = np.reshape(bottom_diff_real, x_input_shape), np.reshape(
                    bottom_diff_imag, x_input_shape)

                if self.channel_first:
                    [bottom_diff_real, bottom_diff_imag] = transpose_channel_first_last(
                        [bottom_diff_real, bottom_diff_imag], change_back=True, dimensions='4d')
                return bottom_diff_real + 1j * bottom_diff_imag
            else:
                x_input = np.ascontiguousarray(np.imag(x_input))
                indices_input = np.ascontiguousarray(np.real(indices_input))
                top_diff_real = np.ascontiguousarray(top_diff)

                x_input_shape = x_input.shape
                size_in = (int(np.prod(np.array(x_input_shape))),)
                x_input = np.reshape(x_input, size_in)
                top_diff_real_shape = top_diff_real.shape
                size_out = (int(np.prod(np.array(top_diff_real_shape))),)
                top_diff_real, indices_input = np.reshape(top_diff_real, size_out) ,np.reshape(indices_input, size_out)

                bottom_diff_real, _ = op.adjoint(x_input * 0, x_input * 0, x_input * 0, x_input * 0,
                                                 top_diff_real, top_diff_real * 0, indices_input)
                bottom_diff_real = np.reshape(bottom_diff_real, x_input_shape)
                if self.channel_first:
                    [bottom_diff_real] = transpose_channel_first_last([bottom_diff_real], change_back=True,
                                                                      dimensions='4d')
                return bottom_diff_real

