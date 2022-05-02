import numpy as np
import torch
import unittest
import optoth.nabla2

# to run execute: python -m unittest [-v] optoth.nabla2
class TestNabla2Function(unittest.TestCase):
    def _test_adjointness(self, dtype, dim):
        # get the corresponding operator
        op = optoth.nabla2.Nabla2(dim)
        # setup the vaiables
        cuda = torch.device('cuda')
        shape = [dim,] + [30 for i in range(dim)]
        th_x = torch.randn(*shape, dtype=dtype, device=cuda)
        shape[0] = dim**2
        th_p = torch.randn(*shape, dtype=dtype, device=cuda)

        th_nabla_x = op.forward(th_x)
        th_nablaT_p = op.adjoint(th_p)

        lhs = (th_nabla_x * th_p).sum().cpu().numpy()
        rhs = (th_x * th_nablaT_p).sum().cpu().numpy()

        print('dtype: {} dim: {} diff: {}'.format(dtype, dim, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(torch.float32, 2)

    def test_float3_gradient(self):
        self._test_adjointness(torch.float32, 3)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float4_gradient(self):
        self._test_adjointness(torch.float32, 4)

    def test_double2_gradient(self):
        self._test_adjointness(torch.float64, 2)

    def test_double3_gradient(self):
        self._test_adjointness(torch.float64, 3)

    def test_double4_gradient(self):
        self._test_adjointness(torch.float64, 4)

    def test_cfloat2_gradient(self):
        self._test_adjointness(torch.complex64, 2)

    def test_cfloat3_gradient(self):
        self._test_adjointness(torch.complex64, 3)

    @unittest.skip("inaccurate due to floating point precision")
    def test_cfloat4_gradient(self):
        self._test_adjointness(torch.complex64, 4)

    def test_cdouble2_gradient(self):
        self._test_adjointness(torch.complex128, 2)

    def test_cdouble3_gradient(self):
        self._test_adjointness(torch.complex128, 3)

    def test_cdouble4_gradient(self):
        self._test_adjointness(torch.complex128, 4)


if __name__ == "__main__":
    unittest.main()
