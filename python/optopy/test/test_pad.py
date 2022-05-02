import unittest
import optopy.pad
import numpy as np

# to run execute: python -m unittest [-v] optopy.pad
class TestPad1d(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3, 3]
        op = optopy.pad.Pad1d(padding, mode)
        # setup the vaiables
        shape = [4, 32]
        np_x = np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np_p).sum()
        rhs = (np_x * np_KT_p).sum()

        print('forward: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')

    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')


class TestComplexPad1d(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3, 3]
        op = optopy.pad.Pad1d(padding, mode)

        # setup the vaiables
        shape = [4, 32]
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype) + 1j * np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('adjoint: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')

    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')


class TestPad2d(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3, 3, 2, 2]
        op = optopy.pad.Pad2d(padding, mode)
        # setup the vaiables
        shape = [4, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[2] + padding[3]
        padded_shape[2] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np_p).sum()
        rhs = (np_x * np_KT_p).sum()

        print('forward: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')

    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')


class TestComplexPad2d(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3, 3, 2, 2]
        op = optopy.pad.Pad2d(padding, mode)

        # setup the vaiables
        shape = [4, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[2] + padding[3]
        padded_shape[2] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype) + 1j * np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('adjoint: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')

    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')


class TestPad3d(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3, 3, 2, 2, 1, 1]
        op = optopy.pad.Pad3d(padding, mode)
        # setup the vaiables
        shape = [4, 32, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[4] + padding[5]
        padded_shape[2] += padding[2] + padding[3]
        padded_shape[3] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np_p).sum()
        rhs = (np_x * np_KT_p).sum()

        print('forward: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')

    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')


class TestComplexPad3d(unittest.TestCase):
    def _test_adjointness(self, dtype, mode):
        # get the corresponding operator
        padding = [3, 3, 2, 2, 1, 1]
        op = optopy.pad.Pad3d(padding, mode)

        # setup the vaiables
        shape = [4, 32, 32, 32]
        np_x = np.random.randn(*shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
        padded_shape = shape
        padded_shape[1] += padding[4] + padding[5]
        padded_shape[2] += padding[2] + padding[3]
        padded_shape[3] += padding[0] + padding[1]
        np_p = np.random.randn(*padded_shape).astype(dtype) + 1j * np.random.randn(*padded_shape).astype(dtype)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('adjoint: dtype={} mode={} diff: {}'.format(dtype, mode, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(np.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(np.float32, 'replicate')

    def test_float_reflect(self):
        self._test_adjointness(np.float32, 'reflect')

    def test_double_symmetric(self):
        self._test_adjointness(np.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(np.float64, 'replicate')

    def test_double_reflect(self):
        self._test_adjointness(np.float64, 'reflect')

if __name__ == "__main__":
    unittest.main()