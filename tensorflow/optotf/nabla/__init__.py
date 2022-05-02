from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as _ops

_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_nabla_operator.so"))

__all__ = ['nabla_2d',
           'nabla_2d_adjoint',
           'nabla_3d',
           'nabla_3d_adjoint',
           'nabla_4d',
           'nabla_4d_adjoint']

nabla_2d = _ext.nabla2d_operator
nabla_2d_adjoint = _ext.nabla2d_operator_adjoint
nabla_3d = _ext.nabla3d_operator
nabla_3d_adjoint = _ext.nabla3d_operator_adjoint
nabla_4d = _ext.nabla4d_operator
nabla_4d_adjoint = _ext.nabla4d_operator_adjoint
