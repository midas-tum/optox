from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as _ops

_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_warp_operator.so"))

__all__ = ['warp_2d', 'warp_2d_transpose']

warp_2d = _ext.warp
warp_2d_transpose = _ext.warp_transpose

@_ops.RegisterGradient("Warp")
def _warp_forward_grad(op, grad):
    grad_in, grad_u = warp_2d_transpose(grad,
                                op.inputs[1],
                                op.inputs[0],
                                mode=op.get_attr('mode'))
    return [grad_in, grad_u]

@_ops.RegisterGradient("WarpTranspose")
def _warp_transpose_grad(op, grad):
    grad_in, grad_u = warp_2d(grad,
                      op.inputs[1],
                      op.inputs[0],
                      mode=op.get_attr('mode'))
    return [grad_in, grad_u]
