import numpy as np

import _ext.py_gpunufft_operator
import _ext.py_gpunufft_singlecoil_operator

__all__ = ['GpuNufft', 'GpuNufftSingleCoil']

float_2d = _ext.py_gpunufft_operator.GPUNufft_float
singlecoil_float_2d = _ext.py_gpunufft_singlecoil_operator.GPUNufft_singlecoil_float

def getEquivalentDtype(dtype):
    if dtype == np.float32:
        return np.complex64
    elif dtype == np.float64:
        return np.complex128
    elif dtype == np.complex64:
        return np.float32
    elif dtype == np.complex128:
        return np.float64
    else:
        raise ValueError(f'Dtype {dtype} has no complex equivalent')

def complex2float(x):
    assert np.iscomplexobj(x)
    return x.view(getEquivalentDtype(x.dtype)).reshape(x.shape + (2,))

def float2complex(x):
    return x.reshape(x.shape[:-2] + (-1,)).view(getEquivalentDtype(x.dtype))

class GpuNufft(object):
    def __init__(self, img_dim, osf, kernel_width, sector_width):
        self.img_dim = img_dim
        self.osf = osf
        self.kernel_width = kernel_width
        self.sector_width = sector_width
        self.csm = None
        self.traj = None
        self.dcf = None
        self.op = _ext.py_gpunufft_operator.GPUNufft_float(img_dim, osf, kernel_width, sector_width)

    def setCsm(self, csm):
        self.csm = complex2float(csm)

    def setTraj(self, traj):
        self.traj = traj

    def setDcf(self, dcf):
        self.dcf = dcf

    def forward(self, x):
        return float2complex(self.op.forward(complex2float(x), self.csm, self.traj, self.dcf))

    def adjoint(self, y):
        return float2complex(self.op.adjoint(complex2float(y), self.csm, self.traj, self.dcf))

class GpuNufftSingleCoil(object):
    def __init__(self, img_dim, osf, kernel_width, sector_width):
        self.img_dim = img_dim
        self.osf = osf
        self.kernel_width = kernel_width
        self.sector_width = sector_width
        self.traj = None
        self.dcf = None
        self.op = _ext.py_gpunufft_singlecoil_operator.GPUNufft_singlecoil_float(img_dim, osf, kernel_width, sector_width)

    def setTraj(self, traj):
        self.traj = traj

    def setDcf(self, dcf):
        self.dcf = dcf

    def forward(self, x):
        return float2complex(self.op.forward(complex2float(x), self.traj, self.dcf))

    def adjoint(self, y):
        return float2complex(self.op.adjoint(complex2float(y), self.traj, self.dcf))
