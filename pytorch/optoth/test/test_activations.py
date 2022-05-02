import unittest
import torch
import numpy as np

import optoth.activations.act

# to run execute: python -m unittest [-v] optoth.activations.act
class TestActivationFunction(unittest.TestCase):
    def _run_gradient_test(self, base_type):
        # setup the hyper parameters for each test
        Nw = 31
        vmin = -1.0
        vmax = 1
        dtype = np.float64
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

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.tensor(np_x, device=cuda)
        th_w = torch.tensor(np_w, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_b = torch.tensor(b, requires_grad=True, dtype=th_w.dtype, device=cuda)
        op = optoth.activations.act.ActivationFunction

        # setup the model
        compute_loss = lambda a, b: 0.5 * torch.sum(op.apply(th_x*a, th_w*b, base_type, vmin, vmax)**2)
        th_loss = compute_loss(th_a, th_b)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
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


class TestActivation2Function(unittest.TestCase):
    
    def _run_gradient_test(self, base_type):
        # setup the hyper parameters for each test
        Nw = 31
        vmin = -1.0
        vmax = 1
        dtype = np.float64
        C = 3

        np_x = np.linspace(vmin, vmax, Nw, dtype=dtype)
        np_w = np.tile(np_x[np.newaxis, :], (C, 1))

        # specify the functions
        np_w[0, :] = np_x
        np_w[1, :] = np_x ** 2
        np_w[2, :] = np.sin(np_x*np.pi)

        np_x = np.linspace(2 * vmin, 2 * vmax, 1001, dtype=dtype)[np.newaxis, :]
        np_x = np.tile(np_x, (C, 1))

        # perform a gradient check:
        epsilon = 1e-4

        # prefactors
        a = 1.1
        b = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.tensor(np_x, device=cuda)
        th_w = torch.tensor(np_w, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_b = torch.tensor(b, requires_grad=True, dtype=th_w.dtype, device=cuda)
        op = optoth.activations.act.Activation2Function

        # setup the model
        def compute_loss(a, b):
            f, f_prime = op.apply(th_x*a, th_w*b, base_type, vmin, vmax)
            return 0.5 * torch.sum(f**2 + f_prime**2)
        th_loss = compute_loss(th_a, th_b)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)
        
    def test_rbf_gradient(self):
        self._run_gradient_test("rbf")


if __name__ == "__main__":
    unittest.test()
