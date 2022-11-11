import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.linear_model as linear_model
from torch.autograd import gradcheck


BSZ = 128
DIM = 5


class TestLinearRegression(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randn(BSZ, 1)

        ref = linear_model.LinearRegression(fit_intercept=False)
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.linear_model.LinearRegression(fit_intercept=False)
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))
        model_forward = model(torch.from_numpy(X))

        self.assertTrue(np.allclose(ref_preds, model_preds.numpy()))
        self.assertTrue(np.allclose(ref_preds, model_forward.numpy()))

        inputX = torch.from_numpy(X)
        inputX.requires_grad = True
        self.assertTrue(gradcheck(model.predict, inputX, eps=1e-6, atol=1e-3))
        self.assertTrue(gradcheck(model, inputX, eps=1e-6, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
