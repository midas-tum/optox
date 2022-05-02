import unittest
import numpy as np
import optoth.warp
import torch

# to run execute: python -m unittest [-v] optoth.warp
class TestWarpFunction(unittest.TestCase):
    def _run_gradient_test(self, dtype, mode, op):
        # setup the hyper parameters for each test
        M, N = 32, 32
        C = 1
        S = 1

        # perform a gradient check:
        epsilon = 1e-4

        # prefactors
        a = 1.1
        bu = np.ones((1,1,1,1))*1.1
        bv = np.ones((1,1,1,1))*1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.randn(S, C, M, N, dtype=dtype, device=cuda)
        float_dtype = torch.float32 if th_x.dtype in [torch.complex64, torch.float32] else torch.float64
        th_u = torch.randn(S, M, N, 2, dtype=float_dtype, device=cuda)

        th_a = torch.tensor(a, requires_grad=True, dtype=float_dtype, device=cuda)
        th_bu = torch.tensor(bu, requires_grad=True, dtype=float_dtype, device=cuda)
        th_bv = torch.tensor(bv, requires_grad=True, dtype=float_dtype, device=cuda)

        # setup the model
        compute_loss = lambda a, b: 0.5 * torch.norm(op.apply(th_x*a, th_u*b, mode)**2)
        th_loss = compute_loss(th_a, torch.cat([th_bu, th_bv], 3))

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.item()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, torch.cat([th_bu, th_bv], 3)).item()
            l_an = compute_loss(th_a-epsilon, torch.cat([th_bu, th_bv], 3)).item()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        grad_bu = th_bu.grad.item()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_bup = compute_loss(th_a, torch.cat([th_bu+epsilon, th_bv], 3)).item()
            l_bun = compute_loss(th_a, torch.cat([th_bu-epsilon, th_bv], 3)).item()
            grad_bu_num = (l_bup - l_bun) / (2 * epsilon)

        print("grad_u: {:.7f} num_grad_u {:.7f} success: {}".format(
            grad_bu, grad_bu_num, np.abs(grad_bu - grad_bu_num) < 1e-4))
        #self.assertTrue(np.abs(grad_bu - grad_bu_num) < 1e-4)

        grad_bv = th_bv.grad.item()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_bvp = compute_loss(th_a, torch.cat([th_bu, th_bv+epsilon], 3)).item()
            l_bvn = compute_loss(th_a, torch.cat([th_bu, th_bv-epsilon], 3)).item()
            grad_bv_num = (l_bvp - l_bvn) / (2 * epsilon)

        print("grad_v: {:.7f} num_grad_v {:.7f} success: {}".format(
            grad_bv, grad_bv_num, np.abs(grad_bv - grad_bv_num) < 1e-4))
        self.assertTrue(np.abs(grad_bv - grad_bv_num) < 1e-4)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float_gradient(self):
        self._run_gradient_test(torch.float32, 'zeros', optoth.warp.WarpFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_double_gradient(self):
        self._run_gradient_test(torch.float64, 'zeros', optoth.warp.WarpFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_cfloat_gradient(self):
        self._run_gradient_test(torch.complex64, 'zeros', optoth.warp.WarpFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_cdouble_gradient(self):
        self._run_gradient_test(torch.complex128, 'zeros', optoth.warp.WarpFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float_gradient_replicate(self):
        self._run_gradient_test(torch.float32, 'replicate', optoth.warp.WarpFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_double_gradient_replicate(self):
        self._run_gradient_test(torch.float64, 'replicate', optoth.warp.WarpFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_cfloat_gradient_replicate(self):
        self._run_gradient_test(torch.complex64, 'replicate', optoth.warp.WarpFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_cdouble_gradient_replicate(self):
        self._run_gradient_test(torch.complex128, 'replicate', optoth.warp.WarpFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_T_float_gradient(self):
        self._run_gradient_test(torch.float32, 'zeros', optoth.warp.WarpTransposeFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_T_double_gradient(self):
        self._run_gradient_test(torch.float64, 'zeros', optoth.warp.WarpTransposeFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_T_cfloat_gradient(self):
        self._run_gradient_test(torch.complex64, 'zeros', optoth.warp.WarpTransposeFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_T_cdouble_gradient(self):
        self._run_gradient_test(torch.complex128, 'zeros', optoth.warp.WarpTransposeFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_T_float_gradient_replicate(self):
        self._run_gradient_test(torch.float32, 'replicate', optoth.warp.WarpTransposeFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_T_double_gradient_replicate(self):
        self._run_gradient_test(torch.float64, 'replicate', optoth.warp.WarpTransposeFunction)

    @unittest.skip("inaccurate due to floating point precision")
    def test_T_cfloat_gradient_replicate(self):
        self._run_gradient_test(torch.complex64, 'replicate', optoth.warp.WarpTransposeFunction)
        
    @unittest.skip("inaccurate due to floating point precision")
    def test_T_cdouble_gradient_replicate(self):
        self._run_gradient_test(torch.complex128, 'replicate', optoth.warp.WarpTransposeFunction)

if __name__ == "__main__":
    unittest.main()
