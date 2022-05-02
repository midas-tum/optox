import numpy as np
import unittest
import optopy.nabla

# to run execute: python -m unittest [-v] optopy.nabla
class TestNablaFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        assert dim in [2, 3, 4]
        # get the corresponding operator
        op = optopy.nabla.Nabla(dim, hx, hy, hz)
        # setup the vaiables
        shape = [10 for i in range(dim)]
        np_x = np.random.randn(*shape).astype(dtype)
        shape.insert(0, dim)
        np_p = np.random.randn(*shape).astype(dtype)

        np_nabla_x = op.forward(np_x)
        np_nablaT_p = op.adjoint(np_p)

        lhs = (np_nabla_x * np_p).sum()
        rhs = (np_x * np_nablaT_p).sum()
        
        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(np.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 1)

    def test_float4_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 1, 1)

    def test_double2_gradient(self):
        self._test_adjointness(np.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 1, 1)

    # anisotropic
    def test_float3_aniso_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 2)

    def test_double3_aniso_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 2)

    def test_float4_aniso_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 2, 4)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 2, 4)

class TestComplexNablaFunction(unittest.TestCase):           
    def _test_adjointness(self, dtype, dim, hx, hy, hz=1, ht=1):
        # get the corresponding operator
        op = optopy.nabla.Nabla(dim, hx, hy, hz, ht)
        # setup the vaiables
        shape = [10 for i in range(dim)]
        shape[0] *= 2
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        shape.insert(0, dim)
        np_p = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)

        np_nabla_x = op.forward(np_x)
        np_nablaT_p = op.adjoint(np_p)

        lhs = (np_nabla_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_nablaT_p)).sum()

        if dim == 2:
            print('dtype: {} dim: {} hx: {} hy: {} diff: {}'.format(dtype, dim, hx, hy, np.abs(lhs - rhs)))
        elif dim == 3:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} diff: {}'.format(dtype, dim, hx, hy, hz, np.abs(lhs - rhs)))
        else: # dim == 4:
            print('dtype: {} dim: {} hx: {} hy: {} hz: {} ht: {} diff: {}'.format(dtype, dim, hx, hy, hz, ht, np.abs(lhs - rhs)))

        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(np.float32, 2, 1, 1)

    def test_float3_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 1)

    def test_float4_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 1, 1)

    def test_double2_gradient(self):
        self._test_adjointness(np.float64, 2, 1, 1)

    def test_double3_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 1)

    def test_double4_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 1, 1)

    # anisotropic
    def test_float3_aniso_gradient(self):
        self._test_adjointness(np.float32, 3, 1, 1, 2)

    def test_double3_aniso_gradient(self):
        self._test_adjointness(np.float64, 3, 1, 1, 2)

    def test_float4_aniso_gradient(self):
        self._test_adjointness(np.float32, 4, 1, 1, 2, 4)

    def test_double4_aniso_gradient(self):
        self._test_adjointness(np.float64, 4, 1, 1, 2, 4)

if __name__ == "__main__":
    unittest.main()
