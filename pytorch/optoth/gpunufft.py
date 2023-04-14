import torch
import torch.nn as nn

import _ext.th_gpunufft_operator
import _ext.th_gpunufft_singlecoil_operator

__all__ = ['GPUNufftFunction', 'GPUNufftSingleCoilFunction']

class GPUNufftSingleCoilFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, img_dim, osf, kernel_width, sector_width):
        if dtype == torch.float32:
            return _ext.th_gpunufft_singlecoil_operator.GPUNufft_singlecoil_float(img_dim, osf, kernel_width, sector_width)
        else:
            raise RuntimeError('Unsupported dtype!')

    @staticmethod
    def forward(ctx, x, trajectory, dcf, img_dim, osf, kernel_width, sector_width):
        ctx.save_for_backward(trajectory, dcf)
        ctx.op = GPUNufftSingleCoilFunction._get_operator(dcf.dtype, img_dim, osf, kernel_width, sector_width)
        return torch.view_as_complex(ctx.op.forward(torch.view_as_real(x), trajectory, dcf))

    @staticmethod
    def backward(ctx, grad_in):
        trajectory, dcf = ctx.saved_tensors
        grad_x = torch.view_as_complex(ctx.op.adjoint(torch.view_as_real(grad_in), trajectory, dcf))
        return grad_x, None, None, None, None, None, None

class GPUNufftSingleCoilFunctionAdjoint(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, img_dim, osf, kernel_width, sector_width):
        if dtype == torch.float32:
            return _ext.th_gpunufft_singlecoil_operator.GPUNufft_singlecoil_float(img_dim, osf, kernel_width, sector_width)
        else:
            raise RuntimeError('Unsupported dtype!')

    @staticmethod
    def forward(ctx, y, trajectory, dcf, img_dim, osf, kernel_width, sector_width):
        ctx.save_for_backward(trajectory, dcf)
        ctx.op = GPUNufftSingleCoilFunction._get_operator(dcf.dtype, img_dim, osf, kernel_width, sector_width)
        return torch.view_as_complex(ctx.op.adjoint(torch.view_as_real(y), trajectory, dcf))

    @staticmethod
    def backward(ctx, grad_in):
        trajectory, dcf = ctx.saved_tensors
        grad_y = torch.view_as_complex(ctx.op.forward(torch.view_as_real(grad_in), trajectory, dcf))
        return grad_y, None, None, None, None, None, None

class GPUNufftFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, img_dim, osf, kernel_width, sector_width):
        if dtype == torch.float32:
            return _ext.th_gpunufft_operator.GPUNufft_float(img_dim, osf, kernel_width, sector_width)
        else:
            raise RuntimeError('Unsupported dtype!')

    @staticmethod
    def forward(ctx, x, csm, trajectory, dcf, img_dim, osf, kernel_width, sector_width):
        ctx.save_for_backward(csm, trajectory, dcf)
        ctx.op = GPUNufftFunction._get_operator(dcf.dtype, img_dim, osf, kernel_width, sector_width)
        return torch.view_as_complex(ctx.op.forward(torch.view_as_real(x), torch.view_as_real(csm), trajectory, dcf))

    @staticmethod
    def backward(ctx, grad_in):
        csm, trajectory, dcf = ctx.saved_tensors
        grad_x = torch.view_as_complex(ctx.op.adjoint(torch.view_as_real(grad_in), torch.view_as_real(csm), trajectory, dcf))
        return grad_x, None, None, None, None, None, None, None

class GPUNufftFunctionAdjoint(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, img_dim, osf, kernel_width, sector_width):
        if dtype == torch.float32:
            return _ext.th_gpunufft_operator.GPUNufft_float(img_dim, osf, kernel_width, sector_width)
        else:
            raise RuntimeError('Unsupported dtype!')

    @staticmethod
    def forward(ctx, y, csm, trajectory, dcf, img_dim, osf, kernel_width, sector_width):
        ctx.save_for_backward(csm, trajectory, dcf)
        ctx.op = GPUNufftFunction._get_operator(dcf.dtype, img_dim, osf, kernel_width, sector_width)
        return torch.view_as_complex(ctx.op.adjoint(torch.view_as_real(y), torch.view_as_real(csm), trajectory, dcf))

    @staticmethod
    def backward(ctx, grad_in):
        csm, trajectory, dcf = ctx.saved_tensors
        grad_y = torch.view_as_complex(ctx.op.forward(torch.view_as_real(grad_in), torch.view_as_real(csm), trajectory, dcf))
        return grad_y, None, None, None, None, None, None, None

class GpuNufft(nn.Module):
    def __init__(self, img_dim, osf, kernel_width=3, sector_width=8):
        super(GpuNufft, self).__init__()

        self.img_dim = img_dim
        self.osf     = osf
        self.kernel_width = kernel_width
        self.sector_width = sector_width

        self.op = GPUNufftFunction
        self.op_backward = GPUNufftFunctionAdjoint

    def forward(self, x, csm, trajectory, dcf):
        # compute the output
        y = self.op.apply(x, csm, trajectory, dcf, self.img_dim, self.osf, self.kernel_width, self.sector_width)
        return y

    def backward(self, y, csm, trajectory, dcf):
        x = self.op_backward.apply(y, csm, trajectory, dcf, self.img_dim, self.osf, self.kernel_width, self.sector_width)
        return x

    def extra_repr(self):
        s = "img_dim={img_dim} osf={osf} kernel_width={kernel_width} sector_width={sector_width}"
        return s.format(**self.__dict__)


class GpuNufftSingleCoil(nn.Module):
    def __init__(self, img_dim, osf, kernel_width=3, sector_width=8):
        super(GpuNufftSingleCoil, self).__init__()

        self.img_dim = img_dim
        self.osf     = osf
        self.kernel_width = kernel_width
        self.sector_width = sector_width

        self.op = GPUNufftSingleCoilFunction
        self.op_backward = GPUNufftSingleCoilFunctionAdjoint

    def forward(self, x, trajectory, dcf):
        # compute the output
        y = self.op.apply(x, trajectory, dcf, self.img_dim, self.osf, self.kernel_width, self.sector_width)
        return y

    def backward(self, y, trajectory, dcf):
        x = self.op_backward.apply(y, trajectory, dcf, self.img_dim, self.osf, self.kernel_width, self.sector_width)
        return x

    def extra_repr(self):
        s = "img_dim={img_dim} osf={osf} kernel_width={kernel_width} sector_width={sector_width}"
        return s.format(**self.__dict__)
