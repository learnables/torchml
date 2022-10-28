import unittest
import numpy as np
import torch
from sklearn.datasets import make_regression
import sklearn.svm as svm
import time

from torchml.svm import LinearSVR

n_samples = 5000
n_features = 10
n_informative = 7


class TestLinearSVR(unittest.TestCase):
    def test_LinearSVR(self):
        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
        )
        lsvr = LinearSVR(max_iter=1000)
        start = time.time()
        lsvr.fit(torch.from_numpy(x), torch.from_numpy(y))
        end = time.time()
        print(end - start)
        start = time.time()
        reflsvr = svm.LinearSVR(max_iter=100000)
        reflsvr.fit(x, y)

        end = time.time()
        print(end - start)
        self.assertTrue(np.allclose(lsvr.coef_.numpy(), reflsvr.coef_, atol=1e-2))
        self.assertTrue(
            np.allclose(lsvr.intercept_.numpy(), reflsvr.intercept_, atol=1e-2)
        )
        self.assertTrue(
            np.allclose(
                lsvr.predict(torch.from_numpy(x)), reflsvr.predict(x), atol=1e-2
            )
        )


if __name__ == "__main__":
    unittest.main()
