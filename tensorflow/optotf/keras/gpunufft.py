import tensorflow as tf
import optotf.gpunufft
import unittest

class GpuNUFFT(tf.keras.layers.Layer):
    def __init__(self, osf, sector_width, kernel_width, img_dim):
        super().__init__()
        self.osf = osf
        self.sector_width = sector_width
        self.kernel_width = kernel_width
        self.img_dim = img_dim
        self.op = optotf.gpunufft.forward

    def call(self, x, csm, traj, dcf):
        return self.op(x,
                       csm,
                       traj,
                       dcf,
                       osf=self.osf,
                       sector_width=self.sector_width,
                       kernel_width=self.kernel_width,
                       img_dim=self.img_dim)

class GpuNUFFTAdjoint(GpuNUFFT):
    def __init__(self, osf, sector_width, kernel_width, img_dim):
        super().__init__(osf, sector_width, kernel_width, img_dim)
        self.op = optotf.gpunufft.adjoint

class GpuNUFFTMultires(tf.keras.layers.Layer):
    def __init__(self, osf, sector_width, kernel_width, img_dim):
        super().__init__()
        self.osf = osf
        self.sector_width = sector_width
        self.kernel_width = kernel_width
        self.img_dim = img_dim
        self.op = optotf.gpunufft.forward_multires

    def call(self, x1, traj1, dcf1, x2, traj2, dcf2, csm):
        return self.op(x1,
                       traj1,
                       dcf1,
                       x2,
                       traj2,
                       dcf2,
                       csm,
                       osf=self.osf,
                       sector_width=self.sector_width,
                       kernel_width=self.kernel_width,
                       img_dim=self.img_dim)

class GpuNUFFTMultiresAdjoint(GpuNUFFTMultires):
    def __init__(self, osf, sector_width, kernel_width, img_dim):
        super().__init__(osf, sector_width, kernel_width, img_dim)
        self.op = optotf.gpunufft.adjoint_multires

class GpuNUFFTSingleCoil(tf.keras.layers.Layer):
    def __init__(self, osf, sector_width, kernel_width, img_dim):
        super().__init__()
        self.osf = osf
        self.sector_width = sector_width
        self.kernel_width = kernel_width
        self.img_dim = img_dim
        self.op = optotf.gpunufft.forward_singlecoil

    def call(self, x, traj, dcf):
        return self.op(x,
                       traj,
                       dcf,
                       osf=self.osf,
                       sector_width=self.sector_width,
                       kernel_width=self.kernel_width,
                       img_dim=self.img_dim)

class GpuNUFFTSingleCoilAdjoint(GpuNUFFTSingleCoil):
    def __init__(self, osf, sector_width, kernel_width, img_dim):
        super().__init__(osf, sector_width, kernel_width, img_dim)
        self.op = optotf.gpunufft.adjoint_singlecoil
