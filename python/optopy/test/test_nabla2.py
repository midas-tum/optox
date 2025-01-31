import unittest
import numpy as np
import optopy.nabla2

# to run execute: python -m unittest [-v] optopy.nabla2
class TestNabla2Function(unittest.TestCase):
    def _get_nabla_op(self, dtype, dim):
        if dtype == np.float32:
            if dim == 2:
                return optopy.nabla2.float_2d()
            elif dim == 3:
                return optopy.nabla2.float_3d()
            elif dim == 4:
                return optopy.nabla2.float_4d()
            else:
                raise RuntimeError('Invalid number of dimensions!')
        elif dtype == np.float64:
            if dim == 2:
                return optopy.nabla2.double_2d()
            elif dim == 3:
                return optopy.nabla2.double_3d()
            elif dim == 4:
                return optopy.nabla2.double_4d()
            else:
                raise RuntimeError('Invalid number of dimensions!')
        else:
            raise RuntimeError('Invalid dtype!')
            
    def _test_adjointness(self, dtype, dim):
        # get the corresponding operator
        op = self._get_nabla_op(dtype, dim)
        # setup the vaiables
        shape = [dim,] + [5 for i in range(dim)]
        np_x = np.random.randn(*shape).astype(dtype)
        shape[0] = dim**2
        np_p = np.random.randn(*shape).astype(dtype)

        np_nabla_x = op.forward(np_x)
        np_nablaT_p = op.adjoint(np_p)

        lhs = (np_nabla_x * np_p).sum()
        rhs = (np_x * np_nablaT_p).sum()

        print('dtype: {} dim: {} diff: {}'.format(dtype, dim, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_gradient(self):
        self._test_adjointness(np.float32, 2)

    def test_float3_gradient(self):
        self._test_adjointness(np.float32, 3)

    def test_float4_gradient(self):
        self._test_adjointness(np.float32, 4)

    def test_double2_gradient(self):
        self._test_adjointness(np.float64, 2)

    def test_double3_gradient(self):
        self._test_adjointness(np.float64, 3)

    def test_double4_gradient(self):
        self._test_adjointness(np.float64, 4)

if __name__ == "__main__":
    unittest.main()
