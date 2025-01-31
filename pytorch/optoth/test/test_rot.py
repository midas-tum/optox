import unittest
import numpy as np
import torch
import optoth.rot

# to run execute: python -m unittest [-v] optoth.rot
class TestRotFunction(unittest.TestCase):
    def _run_gradient_test(self, dtype):
        # setup the hyper parameters for each test
        ks = 7
        np_angles = np.linspace(0, 2*np.pi, num=4, endpoint=False)

        np_x = np.random.randn(3, 2, ks, ks)

        # perform a gradient check:
        epsilon = 1e-4

        # prefactors
        a = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.tensor(np_x, device=cuda, dtype=dtype)
        th_angles = torch.tensor(np_angles, device=cuda, dtype=dtype)
        print(th_x.shape, th_angles.shape)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = optoth.rot.RotationFunction

        # setup the model
        compute_loss = lambda a: 0.5 * torch.sum(op.apply(th_x*a, th_angles)**2)
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).cpu().numpy()
            l_an = compute_loss(th_a-epsilon).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

    @unittest.skip("inaccurate due to floating point precision")
    def test_float_gradient(self):
        self._run_gradient_test(torch.float32)

    def test_double_gradient(self):
        self._run_gradient_test(torch.float64)


if __name__ == "__main__":
    unittest.main()
