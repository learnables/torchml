import unittest
import numpy as np
import torch
from sklearn.datasets import make_classification
import sklearn.svm as svm
import time

from torchml.svm import LinearSVC

n_samples = 5000
n_features = 10
n_classes = 5
n_informative = 7


class TestLinearSVC(unittest.TestCase):
    def test_LinearSVC(self):
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_features-n_informative
        )
        lsvc = LinearSVC(max_iter=1000)
        start = time.time()
        lsvc.fit(torch.from_numpy(x), torch.from_numpy(y))
        end = time.time()
        # print(end - start)
        start = time.time()
        reflsvc = svm.LinearSVC(max_iter=100000)
        reflsvc.fit(x, y)
        end = time.time()
        # print(end - start)
        self.assertTrue(np.allclose(
            lsvc.coef_.numpy(), reflsvc.coef_, atol=1e-2))
        self.assertTrue(
            np.allclose(lsvc.intercept_.numpy(), reflsvc.intercept_, atol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
