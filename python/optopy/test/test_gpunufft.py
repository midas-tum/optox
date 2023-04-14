import numpy as np
import unittest
import optopy.gpunufft

class TestGpunufftFunction(unittest.TestCase):           
    def test_adjointness(self):
        nCh = 6
        nFE = 128
        nFrames = 10
        nSpokes = 64
        dtype = np.float32

        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # get the corresponding operator
        op = optopy.gpunufft.GpuNufft(img_dim, osf, kernel_width, sector_width)

        x_shape = [nFrames, img_dim, img_dim]
        y_shape = [nFrames, nCh, nSpokes * nFE]

        # setup the vaiables
        np_x = np.random.randn(*x_shape).astype(dtype) + 1j * np.random.randn(*x_shape).astype(dtype)
        np_p = np.random.randn(*y_shape).astype(dtype) + 1j * np.random.randn(*y_shape).astype(dtype)

        dcf = np.abs(np.random.randn(*[nFrames, 1, nSpokes * nFE])).astype(dtype)
        traj = np.abs(np.random.randn(*[nFrames, 2, nSpokes * nFE])).astype(dtype)
        traj_abs = np.maximum(np.abs(np.max(traj)), np.abs(np.min(traj)))
        traj /= 2 * traj_abs
        csm = np.random.randn(*[nCh, img_dim, img_dim]).astype(dtype) + 1j * np.random.randn(*[nCh, img_dim, img_dim]).astype(dtype)

        op.setCsm(csm)
        op.setTraj(traj)
        op.setDcf(dcf)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_adjointness_singlecoil(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64
        dtype = np.float32

        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # get the corresponding operator
        op = optopy.gpunufft.GpuNufftSingleCoil(img_dim, osf, kernel_width, sector_width)

        x_shape = [nFrames, img_dim, img_dim]
        y_shape = [nFrames, 1, nSpokes * nFE]

        # setup the vaiables
        np_x = np.random.randn(*x_shape).astype(dtype) + 1j * np.random.randn(*x_shape).astype(dtype)
        np_p = np.random.randn(*y_shape).astype(dtype) + 1j * np.random.randn(*y_shape).astype(dtype)

        dcf = np.abs(np.random.randn(*[nFrames, 1, nSpokes * nFE])).astype(dtype)
        traj = np.abs(np.random.randn(*[nFrames, 2, nSpokes * nFE])).astype(dtype)
        traj_abs = np.maximum(np.abs(np.max(traj)), np.abs(np.min(traj)))
        traj /= 2 * traj_abs

        op.setTraj(traj)
        op.setDcf(dcf)

        np_K_x = op.forward(np_x)
        np_KT_p = op.adjoint(np_p)

        lhs = (np_K_x * np.conj(np_p)).sum()
        rhs = (np_x * np.conj(np_KT_p)).sum()

        print('dtype: {} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)


if __name__ == "__main__":
    unittest.main()