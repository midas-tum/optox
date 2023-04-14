from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as _ops

__all__ = ['forward',
           'adjoint',
           'forward_singlecoil',
           'adjoint_singlecoil']

# load operators from the library
_ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_gpunufft_operator.so"))

forward = _ext.gpu_nufft_operator_forward
adjoint = _ext.gpu_nufft_operator_adjoint
forward_singlecoil = _ext.gpu_nufft_single_coil_operator_forward
adjoint_singlecoil = _ext.gpu_nufft_single_coil_operator_adjoint
forward_multires = _ext.gpu_nufft_multires_operator_forward
adjoint_multires = _ext.gpu_nufft_multires_operator_adjoint

@_ops.RegisterGradient("GpuNufftOperatorForward")
def _gpunufft_forward_grad(op, grad):
    grad_in = adjoint(grad,
                      op.inputs[1],
                      op.inputs[2],
                      op.inputs[3],
                      osf=op.get_attr('osf'),
                      sector_width=op.get_attr('sector_width'),
                      kernel_width=op.get_attr('kernel_width'),
                      img_dim=op.get_attr('img_dim'))
    return [grad_in, None, None, None]

@_ops.RegisterGradient("GpuNufftOperatorAdjoint")
def _gpunufft_adjoint_grad(op, grad):
    grad_in = forward(grad,
                      op.inputs[1],
                      op.inputs[2],
                      op.inputs[3],
                      osf=op.get_attr('osf'),
                      sector_width=op.get_attr('sector_width'),
                      kernel_width=op.get_attr('kernel_width'),
                      img_dim=op.get_attr('img_dim'))
    return [grad_in, None, None, None]

@_ops.RegisterGradient("GpuNufftSingleCoilOperatorForward")
def _gpunufft_singlecoil_forward_grad(op, grad):
    grad_in = adjoint_singlecoil(grad,
                                op.inputs[1],
                                op.inputs[2],
                                osf=op.get_attr('osf'),
                                sector_width=op.get_attr('sector_width'),
                                kernel_width=op.get_attr('kernel_width'),
                                img_dim=op.get_attr('img_dim'))
    return [grad_in,  None, None]

@_ops.RegisterGradient("GpuNufftSingleCoilOperatorAdjoint")
def _gpunufft_singlecoil_adjoint_grad(op, grad):
    grad_in = forward_singlecoil(grad,
                                op.inputs[1],
                                op.inputs[2],
                                osf=op.get_attr('osf'),
                                sector_width=op.get_attr('sector_width'),
                                kernel_width=op.get_attr('kernel_width'),
                                img_dim=op.get_attr('img_dim'))
    return [grad_in,  None, None]

@_ops.RegisterGradient("GpuNufftMultiresOperatorForward")
def _gpunufft_multires_forward_grad(op, grad1, grad2):
    grad_in1, grad_in2 = adjoint_multires(grad1, # img
                      op.inputs[1], # traj
                      op.inputs[2], # dcf
                      grad2,      # img2
                      op.inputs[4], # traj2
                      op.inputs[5], # dcf2
                      op.inputs[6], # csm
                      osf=op.get_attr('osf'),
                      sector_width=op.get_attr('sector_width'),
                      kernel_width=op.get_attr('kernel_width'),
                      img_dim=op.get_attr('img_dim'))
    return [grad_in1, None, None, grad_in2, None, None, None]

@_ops.RegisterGradient("GpuNufftMultiresOperatorAdjoint")
def _gpunufft_multires_adjoint_grad(op, grad1, grad2):
    grad_in1, grad_in2 = forward_multires(grad1,
                      op.inputs[1],
                      op.inputs[2],
                      grad2,
                      op.inputs[4],
                      op.inputs[5], # dcf2
                      op.inputs[6], # csm
                      osf=op.get_attr('osf'),
                      sector_width=op.get_attr('sector_width'),
                      kernel_width=op.get_attr('kernel_width'),
                      img_dim=op.get_attr('img_dim'))
    return [grad_in1, None, None, grad_in2, None, None, None]
