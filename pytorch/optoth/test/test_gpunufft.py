import unittest
import numpy as np
import torch
import optoth.gpunufft

# to run execute: python -m unittest [-v] optoth.gpunufft
class TestGpuNufft(unittest.TestCase):
    def test_adjointness(self):
        nCh = 6
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # setup the variables
        cuda = torch.device('cuda')
        th_x = torch.randn(nFrames, img_dim, img_dim, dtype=torch.complex64, device=cuda)
        # make dcf positive
        th_dcf = torch.abs(torch.randn(nFrames, 1, nSpokes * nFE, dtype=torch.float32, device=cuda))
        # scale trajectory between [-0.5, 0.5]
        th_traj = torch.randn(nFrames, 2, nSpokes * nFE, dtype=torch.float32, device=cuda)
        th_traj_abs = torch.max(torch.abs(th_traj.min()), torch.abs(th_traj.max()))
        th_traj /= 2 * th_traj_abs
        th_csm = torch.randn(nCh, img_dim, img_dim, dtype=torch.complex64, device=cuda)
        th_y = torch.randn(nFrames, nCh, nSpokes * nFE, dtype=torch.complex64, device=cuda)

        # create operator and perform fwd/adj
        op = optoth.gpunufft.GpuNufft(img_dim=img_dim, osf=osf, kernel_width=kernel_width, sector_width=sector_width)
        th_A_x = op.forward(th_x, th_csm, th_traj, th_dcf)
        th_AH_y = op.backward(th_y, th_csm, th_traj, th_dcf)

        # adjointness check
        lhs = torch.sum(torch.conj(th_A_x) * th_y).cpu().numpy()
        rhs = torch.sum(torch.conj(th_x) * th_AH_y).cpu().numpy()
        
        print('gpuNufft adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_adjointness_singlecoil(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # setup the variables
        cuda = torch.device('cuda')
        th_x = torch.randn(nFrames, img_dim, img_dim, dtype=torch.complex64, device=cuda)
        # make dcf positive
        th_dcf = torch.abs(torch.randn(nFrames, 1, nSpokes * nFE, dtype=torch.float32, device=cuda))
        # scale trajectory between [-0.5, 0.5]
        th_traj = torch.randn(nFrames, 2, nSpokes * nFE, dtype=torch.float32, device=cuda)
        th_traj_abs = torch.max(torch.abs(th_traj.min()), torch.abs(th_traj.max()))
        th_traj /= 2 * th_traj_abs
        th_y = torch.randn(nFrames, 1, nSpokes * nFE, dtype=torch.complex64, device=cuda)

        # create operator and perform fwd/adj
        op = optoth.gpunufft.GpuNufftSingleCoil(img_dim=img_dim, osf=osf, kernel_width=kernel_width, sector_width=sector_width)
        th_A_x = op.forward(th_x, th_traj, th_dcf)
        th_AH_y = op.backward(th_y, th_traj, th_dcf)

        # adjointness check
        lhs = torch.sum(torch.conj(th_A_x) * th_y).cpu().numpy()
        rhs = torch.sum(torch.conj(th_x) * th_AH_y).cpu().numpy()
        
        print('gpuNufft adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

if __name__ == "__main__":
    unittest.main()
