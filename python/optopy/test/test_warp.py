import unittest
import numpy as np
import optopy.warp

class TestWarp(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        op = optopy.warp.Warp(mode)

        # setup the vaiables
        shape_x = (2, 1, 20, 20)
        shape_u = (2, 20, 20, 2)

        np_x = np.random.randn(*shape_x).astype(dtype)
        np_u = np.random.randn(*shape_u).astype(dtype) * 10.0

        np_p = np.random.randn(*shape_x).astype(dtype)
        np_warp_x = op.forward(np_x, np_u)
        np_warpT_p = op.adjoint(np_p, np_u, np_x)

        lhs = (np_warp_x * np_p).sum()
        rhs = (np_x * np_warpT_p).sum()

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_adjointness_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_double2_adjointness_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_float2_adjointness(self):
        self._test_adjointness(np.float32, 'zeros')

    def test_double2_adjointness(self):
        self._test_adjointness(np.float64, 'zeros')

class TestComplexWarp(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        op = optopy.warp.Warp(mode)

        # setup the vaiables
        shape_x = (10, 5, 20, 20)
        shape_u = (10, 20, 20, 2)

        np_x = np.random.randn(*shape_x).astype(dtype) + 1j * np.random.randn(*shape_x).astype(dtype) 
        np_u = np.random.randn(*shape_u).astype(dtype) * 10.0

        np_p = np.random.randn(*shape_x).astype(dtype) + 1j * np.random.randn(*shape_x).astype(dtype) 
        np_warp_x = op.forward(np_x, np_u)
        np_warpT_p = op.adjoint(np_p, np_u, np_x)

        lhs = (np_warp_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_warpT_p)).sum()

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float2_adjointness_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_double2_adjointness_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_float2_adjointness(self):
        self._test_adjointness(np.float32, 'zeros')

    def test_double2_adjointness(self):
        self._test_adjointness(np.float64, 'zeros')

if __name__ == "__main__":
    unittest.main()
