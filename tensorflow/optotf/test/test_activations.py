import tensorflow as tf
import unittest
import numpy as np
import optotf.activations

# to run execute: python -m unittest [-v] optotf.activations
class TestFunction(unittest.TestCase):
    def _run_gradient_test(self, base_type):
        # setup the hyper parameters for each test
        Nw = 31
        vmin = -1.0
        vmax = 1

        dtype = np.float64
        tf_dtype = tf.float64

        # determine the operator
        if base_type in ["rbf", "linear", "spline"]:
            op = optotf.activations._get_operator(base_type)
        else:
            raise RuntimeError("Unsupported base type '{}'!".format(base_type))

        C = 3

        np_x = np.linspace(vmin, vmax, Nw, dtype=dtype)
        np_w = np.tile(np_x[np.newaxis, :], (C, 1))

        # specify the functions
        np_w[0, :] = np_x
        np_w[1, :] = np_x ** 2
        np_w[2, :] = np.abs(np_x)

        np_x = np.linspace(2 * vmin, 2 * vmax, 1001, dtype=dtype)[np.newaxis, :]
        np_x = np.tile(np_x, (C, 1))

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1
        b = 1.1

        # transfer to tensorflow
        tf_x = tf.convert_to_tensor(np_x, tf_dtype)
        tf_w = tf.convert_to_tensor(np_w, tf_dtype)
        tf_a = tf.Variable(a, trainable=True, dtype=tf_x.dtype)
        tf_b = tf.Variable(b, trainable=True, dtype=tf_w.dtype)
        
        compute_loss = lambda a, b: 0.5 * tf.reduce_sum(op(tf_x*a, tf_w*b, vmin=vmin, vmax=vmax)**2)

        with tf.GradientTape() as g:
            g.watch(tf_x)
        
            # setup the model
            tf_loss = compute_loss(tf_a, tf_b)

        # backpropagate the gradient
        dLoss = g.gradient(tf_loss, [tf_a, tf_b])

        grad_a = dLoss[0].numpy()
        grad_b = dLoss[1].numpy()

        # numerical gradient w.r.t. the input
        l_ap = compute_loss(tf_a+epsilon, tf_b).cpu().numpy()
        l_an = compute_loss(tf_a-epsilon, tf_b).cpu().numpy()
        grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        l_bp = compute_loss(tf_a, tf_b+epsilon).cpu().numpy()
        l_bn = compute_loss(tf_a, tf_b-epsilon).cpu().numpy()
        grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)
        
    def test_rbf_gradient(self):
        self._run_gradient_test("rbf")

    def test_linear_gradient(self):
        self._run_gradient_test("linear")

    def test_spline_gradient(self):
        self._run_gradient_test("spline")

if __name__ == "__main__":
    unittest.main()
