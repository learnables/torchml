import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.linear_model as linear_model


BSZ = 128
DIM = 5


class TestRidge(unittest.TestCase):

    def test_fit(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randn(BSZ, 1)

        ref = linear_model.Ridge(fit_intercept=False)
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.linear_model.Ridge(fit_intercept=False)
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))
        model_forward = model(torch.from_numpy(X))

        self.assertTrue(np.allclose(ref_preds, model_preds.numpy()))
        self.assertTrue(np.allclose(ref_preds, model_forward.numpy()))

    def test_fit_intercept(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randn(BSZ, 1)

        ref = linear_model.Ridge(fit_intercept=True)
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.linear_model.Ridge(fit_intercept=True)
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))
        model_forward = model(torch.from_numpy(X))

        self.assertTrue(np.allclose(ref_preds, model_preds.numpy()))
        self.assertTrue(np.allclose(ref_preds, model_forward.numpy()))

if __name__ == '__main__':
    unittest.main()
