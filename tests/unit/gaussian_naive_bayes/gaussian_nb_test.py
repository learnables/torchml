import unittest
import numpy as np
import torch
import torchml as ml
from sklearn.naive_bayes import GaussianNB


BSZ = 128
DIM = 5


class TestGaussianNB(unittest.TestCase):

    def test_fit(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randint(low=0, high=5, size=BSZ)

        ref = GaussianNB()
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.gaussian_naive_bayes.GaussianNB()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))
        model_forward = model(torch.from_numpy(X))

        self.assertTrue(np.allclose(ref_preds, model_preds.numpy()))
        self.assertTrue(np.allclose(ref_preds, model_forward.numpy()))


if __name__ == '__main__':
    unittest.main()
