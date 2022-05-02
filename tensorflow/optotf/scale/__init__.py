from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as _ops

_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_scale_operator.so"))

__all__ = ['scale']

scale = _ext.scale

@_ops.RegisterGradient("Scale")
def _scale_forward_grad(op, grad):
    grad_in = _ext.scale_grad(grad)
    return [grad_in]
