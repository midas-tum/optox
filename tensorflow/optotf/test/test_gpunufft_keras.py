import unittest
import tensorflow as tf
import optotf.keras.gpunufft


class TestGpunufft(unittest.TestCase):
    def test_gpunufft_forward(self):
        nCh = 6
        nFE = 128
        nFrames = 10
        nSpokes = 64

        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        x_shape = [nFrames, img_dim, img_dim]
        y_shape = [nFrames, nCh, nSpokes * nFE]
        # setup the vaiables
        tf_x = tf.complex(tf.random.normal(x_shape, dtype=tf.float32), tf.random.normal(x_shape, dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs
        tf_csm = tf.complex(tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32))

        op = optotf.keras.gpunufft.GpuNUFFT(osf, sector_width, kernel_width, img_dim)
        Kx = op(tf_x, tf_csm, tf_traj, tf_dcf)
        self.assertTrue(Kx.shape, tuple(y_shape))
    
    def test_gpunufft_adjoint(self):
        nCh = 6
        nFE = 128
        nFrames = 10
        nSpokes = 64

        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        x_shape = [nFrames, img_dim, img_dim]
        y_shape = [nFrames, nCh, nSpokes * nFE]
        # setup the vaiables
        tf_y = tf.complex(tf.random.normal(y_shape, dtype=tf.float32), tf.random.normal(y_shape, dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs
        tf_csm = tf.complex(tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32), tf.random.normal([nCh, img_dim, img_dim], dtype=tf.float32))

        op = optotf.keras.gpunufft.GpuNUFFTAdjoint(osf, sector_width, kernel_width, img_dim)
        KHy = op(tf_y, tf_csm, tf_traj, tf_dcf)
        self.assertTrue(KHy.shape, tuple(x_shape))

class TestGpunufftSinglecoil(unittest.TestCase):
    def test_gpunufft_forward(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64

        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        x_shape = [nFrames, img_dim, img_dim]
        y_shape = [nFrames, 1, nSpokes * nFE]
        # setup the vaiables
        tf_x = tf.complex(tf.random.normal(x_shape, dtype=tf.float32), tf.random.normal(x_shape, dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs

        op = optotf.keras.gpunufft.GpuNUFFTSingleCoil(osf, sector_width, kernel_width, img_dim)
        Kx = op(tf_x, tf_traj, tf_dcf)
        self.assertTrue(Kx.shape, tuple(y_shape))
    
    def test_gpunufft_adjoint(self):
        nFE = 128
        nFrames = 10
        nSpokes = 64

        osf = 2
        kernel_width = 3
        sector_width = 8
        img_dim = nFE // osf

        x_shape = [nFrames, img_dim, img_dim]
        y_shape = [nFrames, 1, nSpokes * nFE]
        # setup the vaiables
        tf_y = tf.complex(tf.random.normal(y_shape, dtype=tf.float32), tf.random.normal(y_shape, dtype=tf.float32))
        # make dcf positive
        tf_dcf = tf.abs(tf.random.normal([nFrames, 1, nSpokes * nFE], dtype=tf.float32))
        # scale trajectory between [-0.5, 0.5]
        tf_traj = tf.random.normal([nFrames, 2, nSpokes * nFE], dtype=tf.float32)
        tf_traj_abs = tf.maximum(tf.abs(tf.reduce_max(tf_traj)), tf.abs(tf.reduce_min(tf_traj)))
        tf_traj /= 2 * tf_traj_abs

        op = optotf.keras.gpunufft.GpuNUFFTSingleCoilAdjoint(osf, sector_width, kernel_width, img_dim)
        KHy = op(tf_y, tf_traj, tf_dcf)
        self.assertTrue(KHy.shape, tuple(x_shape))

if __name__ == "__main__":
    unittest.main()