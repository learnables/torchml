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
        for i in range(2):
            device = torch.device("cuda" if torch.cuda.is_available() and i else "cpu")
            X = np.random.randn(BSZ, DIM)
            y = np.random.randn(BSZ, 1)

            ref = linear_model.LinearRegression(fit_intercept=False)
            ref.fit(X, y)
            ref_preds = ref.predict(X)

            model = ml.linear_model.LinearRegression(fit_intercept=False)
            model.fit(torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))
            model_preds = model.predict(torch.from_numpy(X).to(device))
            model_forward = model(torch.from_numpy(X).to(device))

            self.assertTrue(np.allclose(ref_preds, model_preds.cpu().numpy()))
            self.assertTrue(np.allclose(ref_preds, model_forward.cpu().numpy()))

            inputX = torch.from_numpy(X).to(device)
            inputX.requires_grad = True
            self.assertTrue(gradcheck(model.predict, inputX, eps=1e-6, atol=1e-3))
            self.assertTrue(gradcheck(model, inputX, eps=1e-6, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
