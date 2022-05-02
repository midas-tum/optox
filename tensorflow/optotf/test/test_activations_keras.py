import unittest
import tensorflow as tf
import optotf.keras.activations
import numpy as np

class TestActivations(unittest.TestCase):
    def test(self):
        x = np.random.normal((10, 5))
        op = optotf.keras.activations.TrainableActivation(-0.5, 0.5, 31)
        y = op(x)
        self.assertTrue(x.shape == y.shape)
        
if __name__ == "__main__":
    unittest.main()