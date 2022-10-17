import unittest
import numpy as np
import torch
from sklearn.datasets import make_classification
import sklearn.svm as svm
import time

import torchml as ml
from torchml.svm import LinearSVC

BSZ = 128
DIM = 5


class TestLinearSVC(unittest.TestCase):
    def test_coef(self):
        x, y = make_classification(n_samples=500, n_features=10,
                                   n_classes=2)
        lsvc = LinearSVC(verbose=0)
        start = time.time()
        lsvc.fit(torch.from_numpy(x), torch.from_numpy(y))
        end = time.time()
        print(end - start)
        print("Here")
        start = time.time()
        reflsvc = svm.LinearSVC()
        reflsvc.fit(x, y)
        end = time.time()
        print(end - start)
        print("Here")
        self.assertTrue(np.allclose(lsvc.coef_.numpy(), reflsvc.coef_, atol=0.03))

if __name__ == "__main__":
    unittest.main()
