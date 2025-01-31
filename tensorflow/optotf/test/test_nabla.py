import tensorflow as tf
import unittest
import optotf.nabla
import numpy as np

# to run execute: python -m unittest [-v] optotf.nabla
class TestNablaFunction(unittest.TestCase):
    def _test_adjointness(self, dtype, dim, hx=1, hy=1, hz=1, ht=1):
        # setup the vaiables
        shape = [30 for i in range(dim)]
        if dtype == tf.complex64 or dtype == tf.complex128:
            float_dtype = tf.float32 if dtype == tf.complex64 else tf.float64
            tf_x = tf.complex(tf.random.normal(shape, dtype=float_dtype), tf.random.normal(shape, dtype=float_dtype))
        else:
            tf_x = tf.random.normal(shape, dtype=dtype)
        
        shape.insert(0, dim)
        if dtype == tf.complex64 or dtype == tf.complex128:
            tf_p = tf.complex(tf.random.normal(shape, dtype=float_dtype), tf.random.normal(shape, dtype=float_dtype))
        else:
            tf_p = tf.random.normal(shape, dtype=dtype)
        
        if dim == 2 and (dtype==tf.float32 or dtype==tf.float64):
            A = lambda x: optotf.nabla.nabla_2d(x, hx=hx, hy=hy)
            AH = lambda x: optotf.nabla.nabla_2d_adjoint(x, hx=hx, hy=hy)
        elif dim == 2 and (dtype==tf.complex64 or dtype==tf.complex128):
            A = lambda x: tf.complex(optotf.nabla.nabla_2d(tf.math.real(x), hx=hx, hy=hy),
                                     optotf.nabla.nabla_2d(tf.math.imag(x), hx=hx, hy=hy))
            AH = lambda x: tf.complex(optotf.nabla.nabla_2d_adjoint(tf.math.real(x), hx=hx, hy=hy),
                                      optotf.nabla.nabla_2d_adjoint(tf.math.imag(x), hx=hx, hy=hy))
        elif dim == 3 and (dtype==tf.float32 or dtype==tf.float64):
            A = lambda x: optotf.nabla.nabla_3d(x, hx=hx, hy=hy, hz=hz)
            AH = lambda x: optotf.nabla.nabla_3d_adjoint(x, hx=hx, hy=hy, hz=hz)
        elif dim == 3 and (dtype==tf.complex64 or dtype==tf.complex128):
            A = lambda x: tf.complex(optotf.nabla.nabla_3d(tf.math.real(x), hx=hx, hy=hy, hz=hz), 
                                     optotf.nabla.nabla_3d(tf.math.imag(x), hx=hx, hy=hy, hz=hz))
            AH = lambda x: tf.complex(optotf.nabla.nabla_3d_adjoint(tf.math.real(x), hx=hx, hy=hy, hz=hz),
                                      optotf.nabla.nabla_3d_adjoint(tf.math.imag(x), hx=hx, hy=hy, hz=hz))
        elif dim == 4 and (dtype==tf.float32 or dtype==tf.float64):
            A = lambda x: optotf.nabla.nabla_4d(x, hx=hx, hy=hy, hz=hz, ht=ht)
            AH = lambda x: optotf.nabla.nabla_4d_adjoint(x, hx=hx, hy=hy, hz=hz, ht=ht)
        elif dim == 4 and (dtype==tf.complex64 or dtype==tf.complex128):
            A = lambda x: tf.complex(optotf.nabla.nabla_4d(tf.math.real(x), hx=hx, hy=hy, hz=hz, ht=ht),
                                     optotf.nabla.nabla_4d(tf.math.imag(x), hx=hx, hy=hy, hz=hz, ht=ht))
            AH = lambda x: tf.complex(optotf.nabla.nabla_4d_adjoint(tf.math.real(x), hx=hx, hy=hy, hz=hz, ht=ht),
                                      optotf.nabla.nabla_4d_adjoint(tf.math.imag(x), hx=hx, hy=hy, hz=hz, ht=ht))
        else:
            raise RuntimeError(f'Nabla not defined for dim={dim} and dtype={dtype}')

        tf_nabla_x = A(tf_x)
        tf_nablaT_p = AH(tf_p)

        lhs = tf.reduce_sum(tf_nabla_x * tf.math.conj(tf_p))
        rhs = tf.reduce_sum(tf_x * tf.math.conj(tf_nablaT_p))

        print('dtype: {} dim: {} diff: {}'.format(dtype, dim, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    # isotropic
    def test_float32_2d_gradient(self):
        self._test_adjointness(tf.float32, 2, 1, 1, 1)
      
    def test_float32_3d_gradient(self):
        self._test_adjointness(tf.float32, 3, 1, 1, 1)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float32_4d_gradient(self):
        self._test_adjointness(tf.float32, 4, 1, 1, 1)

    def test_float64_2d_gradient(self):
        self._test_adjointness(tf.float64, 2, 1, 1, 1)

    def test_float64_3d_gradient(self):
        self._test_adjointness(tf.float64, 3, 1, 1, 1)

    def test_float64_4d_gradient(self):
        self._test_adjointness(tf.float64, 4, 1, 1, 1)

    # anisotropic
    def test_float32_3d_aniso_gradient(self):
        self._test_adjointness(tf.float32, 3, 1, 1, 2)

    def test_float64_3d_aniso_gradient(self):
        self._test_adjointness(tf.float64, 3, 1, 1, 2)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float32_4d_aniso_gradient(self):
        self._test_adjointness(tf.float32, 4, 1, 1, 2, 4)

    def test_float64_4d_aniso_gradient(self):
        self._test_adjointness(tf.float64, 4, 1, 1, 2, 4)


    # complex isotropic
    def test_complex64_2d_gradient(self):
        self._test_adjointness(tf.complex64, 2, 1, 1, 1)
      
    def test_complex64_3d_gradient(self):
        self._test_adjointness(tf.complex64, 3, 1, 1, 1)

    @unittest.skip("inaccurate due to floating point precision")
    def test_complex64_4d_gradient(self):
        self._test_adjointness(tf.complex64, 4, 1, 1, 1)

    def test_complex128_2d_gradient(self):
        self._test_adjointness(tf.complex128, 2, 1, 1, 1)

    def test_complex128_3d_gradient(self):
        self._test_adjointness(tf.complex128, 3, 1, 1, 1)

    def test_complex128_4d_gradient(self):
        self._test_adjointness(tf.complex128, 4, 1, 1, 1)

    # complex anisotropic
    def test_complex64_3d_aniso_gradient(self):
        self._test_adjointness(tf.complex64, 3, 1, 1, 2)

    def test_complex128_3d_aniso_gradient(self):
        self._test_adjointness(tf.complex128, 3, 1, 1, 2)

    @unittest.skip("inaccurate due to floating point precision")
    def test_complex64_4d_aniso_gradient(self):
        self._test_adjointness(tf.complex64, 4, 1, 1, 2, 4)

    def test_complex128_4d_aniso_gradient(self):
        self._test_adjointness(tf.complex128, 4, 1, 1, 2, 4)

if __name__ == "__main__":
    unittest.main()
