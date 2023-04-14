import tensorflow as tf
import unittest
import numpy as np
import optotf.gpunufft

# to run execute: python -m unittest [-v] optotf.gpunufft
class TestGpunufftFunction(unittest.TestCase):
    def test_gradient(self):
        nCh = 6
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # setup the vaiables
        tf_x = tf.complex(tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs
        tf_csm = tf.complex(tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32))
        tf_y = tf.complex(tf.random.normal([nFrames, nCh, nSpokes * nFE], dtype=tf.float32), tf.random.normal([nFrames, nCh, nSpokes * nFE], dtype=tf.float32))

        with tf.GradientTape() as g:
            g.watch(tf_x)
            tf_A_x = optotf.gpunufft.forward(tf_x,
                         tf_csm,
                         tf_traj,
                         tf_dcf,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)
            tf_AH_A_x = optotf.gpunufft.adjoint(tf_A_x - tf_y, 
                         tf_csm,
                         tf_traj,
                         tf_dcf,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)
        dData_dx = g.gradient(tf_AH_A_x, tf_x)
        
    def test_adjointness(self):
        nCh = 6
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # setup the vaiables
        tf_x = tf.complex(tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs
        tf_csm = tf.complex(tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32))
        tf_y = tf.complex(tf.random.normal([nFrames, nCh, nSpokes * nFE], dtype=tf.float32), tf.random.normal([nFrames, nCh, nSpokes * nFE], dtype=tf.float32))

        # perform fwd/adj
        tf_AH_y = optotf.gpunufft.adjoint(tf_y,
                          tf_csm,
                          tf_traj,
                          tf_dcf,
                          osf=osf,
                          sector_width=sector_width,
                          kernel_width=kernel_width,
                          img_dim=img_dim)

        tf_A_x = optotf.gpunufft.forward(tf_x,
                         tf_csm,
                         tf_traj,
                         tf_dcf,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)

        # adjointness check
        lhs = tf.reduce_sum(tf.math.conj(tf_A_x) * tf_y)
        rhs = tf.reduce_sum(tf.math.conj(tf_x) * tf_AH_y)

        print('gpuNufft adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)


# to run execute: python -m unittest [-v] optotf.gpunufft
class TestGpunufftSingleCoilFunction(unittest.TestCase):
    def test_gradient(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # setup the vaiables
        tf_x = tf.complex(tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs
        tf_y = tf.complex(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32), tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))

        with tf.GradientTape() as g:
            g.watch(tf_x)
            tf_A_x = optotf.gpunufft.forward_singlecoil(tf_x,
                         tf_traj,
                         tf_dcf,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)
            tf_AH_A_x = optotf.gpunufft.adjoint_singlecoil(tf_A_x - tf_y, 
                         tf_traj,
                         tf_dcf,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)
        dData_dx = g.gradient(tf_AH_A_x, tf_x)
        
    def test_adjointness(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        # setup the vaiables
        tf_x = tf.complex(tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs
        tf_y = tf.complex(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32), tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))

        # perform fwd/adj
        tf_AH_y = optotf.gpunufft.adjoint_singlecoil(tf_y,
                          tf_traj,
                          tf_dcf,
                          osf=osf,
                          sector_width=sector_width,
                          kernel_width=kernel_width,
                          img_dim=img_dim)

        tf_A_x = optotf.gpunufft.forward_singlecoil(tf_x,
                         tf_traj,
                         tf_dcf,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)

        # adjointness check
        lhs = tf.reduce_sum(tf.math.conj(tf_A_x) * tf_y)
        rhs = tf.reduce_sum(tf.math.conj(tf_x) * tf_AH_y)

        print('gpuNufft adjointness diff: {}'.format(np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

class TestGpunufftMultiresFunction(unittest.TestCase):
    def test_gradient(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64
        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf
        nCh = 6

        # setup the vaiables
        tf_x1 = tf.complex(tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nFrames, img_dim, img_dim], dtype=tf.float32))
        tf_x2 = tf.complex(tf.random.normal([nFrames*2, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nFrames*2, img_dim, img_dim], dtype=tf.float32))

        # make dcf positive
        tf_dcf1 = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        tf_dcf2 = tf.abs(tf.random.normal([nFrames*2, 1, nSpokes * nFE], dtype=tf.float32))

        # scale trajectory between [-0.5, 0.5]
        tf_traj1 = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj1)), tf.abs(tf.reduce_min(tf_traj1)))
        tf_traj1 /= 2 * tf_traj_abs

        tf_traj2 = tf.random.normal([nFrames*2, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj2)), tf.abs(tf.reduce_min(tf_traj2)))
        tf_traj2 /= 2 * tf_traj_abs

        tf_csm = tf.complex(tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32))

        tf_y1 = tf.complex(tf.random.normal([nFrames, nCh, nSpokes * nFE], dtype=tf.float32), tf.random.normal([nFrames, nCh, nSpokes * nFE], dtype=tf.float32))
        tf_y2 = tf.complex(tf.random.normal([nFrames*2, nCh, nSpokes * nFE], dtype=tf.float32), tf.random.normal([nFrames*2, nCh, nSpokes * nFE], dtype=tf.float32))

        with tf.GradientTape() as g:
            g.watch([tf_x1, tf_x2])
            tf_A_x1, tf_A_x2 = optotf.gpunufft.forward_multires(tf_x1,
                         tf_traj1,
                         tf_dcf1,
                         tf_x2,
                         tf_traj2,
                         tf_dcf2,
                         tf_csm,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)
            tf_AH_A_x1, tf_AH_A_x2 = optotf.gpunufft.adjoint_multires(tf_A_x1 - tf_y1, 
                         tf_traj1,
                         tf_dcf1,
                         tf_A_x2 - tf_y2,
                         tf_traj2,
                         tf_dcf2,
                         tf_csm,
                         osf=osf,
                         sector_width=sector_width,
                         kernel_width=kernel_width,
                         img_dim=img_dim)
        dData_dx1 = g.gradient(tf_AH_A_x1, tf_x1)
        

if __name__ == "__main__":
    unittest.main()
