import unittest
import numpy as np
import torch
from sklearn.datasets import make_regression
import sklearn.svm as svm
import time
from torch.autograd import gradcheck

from torchml.svm import LinearSVR

n_samples = 5000
n_features = 4
n_informative = 3


class TestLinearSVR(unittest.TestCase):
    def test_LinearSVR(self):
        for i in range(2):
            device = torch.device("cuda" if torch.cuda.is_available and i else "cpu")
            x, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
            )
            lsvr = LinearSVR(max_iter=1000)
            start = time.time()
            lsvr.fit(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            end = time.time()
            # print(end - start)
            start = time.time()
            reflsvr = svm.LinearSVR(max_iter=100000)
            reflsvr.fit(x, y)

            end = time.time()
            # print(end - start)
            self.assertTrue(
                np.allclose(lsvr.coef_.cpu().numpy(), reflsvr.coef_, atol=1e-2)
            )
            self.assertTrue(
                np.allclose(
                    lsvr.intercept_.cpu().numpy(), reflsvr.intercept_, atol=1e-2
                )
            )
            self.assertTrue(
                np.allclose(
                    lsvr.predict(torch.from_numpy(x).to(device)).cpu().numpy(),
                    reflsvr.predict(x),
                    atol=1e-2,
                )
            )

            inputX = torch.from_numpy(x).to(device)
            inputX.requires_grad = True
            self.assertTrue(gradcheck(lsvr.predict, inputX, eps=1e-6, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
