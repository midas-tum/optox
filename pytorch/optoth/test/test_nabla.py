import numpy as np
import torch
import unittest
import optoth.nabla

# to run execute: python -m unittest [-v] optoth.nabla
class TestNablaFunction(unittest.TestCase):
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        assert dim in [2, 3, 4]
        # get the corresponding operator
        op = optoth.nabla.Nabla(dim, hx, hy, hz, ht)
        # setup the vaiables
        cuda = torch.device('cuda')
        shape = [30 for i in range(dim)]
        th_x = torch.randn(*shape, dtype=dtype, device=cuda)
        shape.insert(0, dim)
        th_p = torch.randn(*shape, dtype=dtype, device=cuda)

        th_nabla_x = op.forward(th_x)
        th_nablaT_p = op.adjoint(th_p)

        lhs = (th_nabla_x * th_p).sum().cpu().numpy()
        rhs = (th_x * th_nablaT_p).sum().cpu().numpy()

        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    # float32
    def test_float2_gradient(self):
        self._test_adjointness(torch.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 1)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float4_gradient(self):
        self._test_adjointness(torch.float32, 4, 1, 1, 1, 1)

    # complex64
    def test_cfloat2_gradient(self):
        self._test_adjointness(torch.complex64, 2, 1, 1)

    def test_cfloat3_gradient(self):
        self._test_adjointness(torch.complex64, 3, 1, 1, 1)

    @unittest.skip("inaccurate due to floating point precision")
    def test_cfloat4_gradient(self):
        self._test_adjointness(torch.complex64, 4, 1, 1, 1, 1)

    # float64
    def test_double2_gradient(self):
        self._test_adjointness(torch.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(torch.float64, 4, 1, 1, 1, 1)

    # complex128
    def test_cdouble2_gradient(self):
        self._test_adjointness(torch.complex128, 2, 1, 1)

    def test_cdouble3_gradient(self):
        self._test_adjointness(torch.complex128, 3, 1, 1, 1)

    def test_cdouble4_gradient(self):
        self._test_adjointness(torch.complex128, 4, 1, 1, 1, 1)

    # anisotropic
    # float32
    def test_float3_aniso_gradient(self):
        self._test_adjointness(torch.float32, 3, 1, 1, 2)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float4_aniso_gradient(self):
        self._test_adjointness(torch.float32, 4, 1, 1, 2, 4)

    # complex128
    def test_cfloat3_aniso_gradient(self):
        self._test_adjointness(torch.complex64, 3, 1, 1, 2)

    def test_cfloat4_aniso_gradient(self):
        self._test_adjointness(torch.complex64, 4, 1, 1, 2, 4)

    # float64
    def test_double3_aniso_gradient(self):
        self._test_adjointness(torch.float64, 3, 1, 1, 2)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(torch.float64, 4, 1, 1, 2, 4)

    # complex128
    def test_cdouble3_aniso_gradient(self):
        self._test_adjointness(torch.complex128, 3, 1, 1, 2)

    def test_cdouble4_aniso_gradient(self):
        self._test_adjointness(torch.complex128, 4, 1, 1, 2, 4)

if __name__ == "__main__":
    unittest.main()
