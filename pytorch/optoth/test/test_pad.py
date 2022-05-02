import numpy as np
import unittest
import torch
import optoth.pad

# to run execute: python -m unittest [-v] optoth.pad2d
class Testpad1dFunction(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # setup the hyper parameters for each test
        S, C, N = 4, 3, 32

        pad = [3,3]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        Ax = optoth.pad.pad1d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]


        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = optoth.pad.pad1d_transpose(p, pad, mode)
        ATx = torch.autograd.grad(Ap, p, x)[0]

        lhs = (Ap * x).sum().item()
        rhs = (p * ATx).sum().item()

        print('adjoint: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(torch.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(torch.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(torch.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(torch.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(torch.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(torch.float64, 'reflect')

    def test_cfloat_symmetric(self):
        self._test_adjointness(torch.complex64, 'symmetric')

    def test_cfloat_replicate(self):
        self._test_adjointness(torch.complex64, 'replicate')
    
    def test_cfloat_reflect(self):
        self._test_adjointness(torch.complex64, 'reflect')
    
    def test_cdouble_symmetric(self):
        self._test_adjointness(torch.complex128, 'symmetric')

    def test_cdouble_replicate(self):
        self._test_adjointness(torch.complex128, 'replicate')
    
    def test_cdouble_reflect(self):
        self._test_adjointness(torch.complex128, 'reflect')
        
class Testpad2dFunction(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # setup the hyper parameters for each test
        S, C, M, N = 4, 3, 32, 32

        pad = [3,3,2,2]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, M, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, M+pad[2]+pad[3], N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        Ax = optoth.pad.pad2d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]


        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = optoth.pad.pad2d_transpose(p, pad, mode)
        ATx = torch.autograd.grad(Ap, p, x)[0]

        lhs = (Ap * x).sum().item()
        rhs = (p * ATx).sum().item()

        print('adjoint: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(torch.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(torch.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(torch.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(torch.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(torch.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(torch.float64, 'reflect')

    def test_cfloat_symmetric(self):
        self._test_adjointness(torch.complex64, 'symmetric')

    def test_cfloat_replicate(self):
        self._test_adjointness(torch.complex64, 'replicate')
    
    def test_cfloat_reflect(self):
        self._test_adjointness(torch.complex64, 'reflect')
    
    def test_cdouble_symmetric(self):
        self._test_adjointness(torch.complex128, 'symmetric')

    def test_cdouble_replicate(self):
        self._test_adjointness(torch.complex128, 'replicate')
    
    def test_cdouble_reflect(self):
        self._test_adjointness(torch.complex128, 'reflect')
        
class Testpad3dFunction(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):                   
        # setup the hyper parameters for each test
        S, C, D, M, N =4, 3, 16, 32, 32

        pad = [3,3,2,2,1,1]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, D, M, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, D+pad[4]+pad[5], M+pad[2]+pad[3], N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        Ax = optoth.pad.pad3d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]

        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = optoth.pad.pad3d_transpose(p, pad, mode)
        ATx = torch.autograd.grad(Ap, p, x)[0]

        lhs = (Ap * x).sum().item()
        rhs = (p * ATx).sum().item()

        print('adjoint: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(torch.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(torch.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(torch.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(torch.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(torch.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(torch.float64, 'reflect')

    def test_cfloat_symmetric(self):
        self._test_adjointness(torch.complex64, 'symmetric')

    def test_cfloat_replicate(self):
        self._test_adjointness(torch.complex64, 'replicate')
    
    def test_cfloat_reflect(self):
        self._test_adjointness(torch.complex64, 'reflect')
    
    def test_cdouble_symmetric(self):
        self._test_adjointness(torch.complex128, 'symmetric')

    def test_cdouble_replicate(self):
        self._test_adjointness(torch.complex128, 'replicate')
    
    def test_cdouble_reflect(self):
        self._test_adjointness(torch.complex128, 'reflect')
        
if __name__ == "__main__":
    unittest.main()
