import unittest
import numpy as np
import torch
from sklearn.datasets import make_classification
import sklearn.svm as svm
import time
from torch.autograd import gradcheck

from torchml.svm import LinearSVC

n_samples = 5000
n_features = 4
n_classes = 2
n_informative = 4


class TestLinearSVC(unittest.TestCase):
    def test_LinearSVC(self):
        for i in range(2):
            device = torch.device("cuda" if torch.cuda.is_available and i else "cpu")
            x, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=n_informative,
                n_redundant=n_features - n_informative,
            )
            lsvc = LinearSVC(max_iter=1000)
            start = time.time()
            lsvc.fit(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            end = time.time()
            # print(end - start)
            start = time.time()
            reflsvc = svm.LinearSVC(max_iter=100000)
            reflsvc.fit(x, y)

            end = time.time()
            # print(end - start)
            self.assertTrue(
                np.allclose(lsvc.coef_.cpu().numpy(), reflsvc.coef_, atol=1e-2)
            )
            self.assertTrue(
                np.allclose(
                    lsvc.intercept_.cpu().numpy(), reflsvc.intercept_, atol=1e-2
                )
            )
            self.assertTrue(
                np.allclose(
                    lsvc.decision_function(torch.from_numpy(x).to(device))
                    .cpu()
                    .numpy(),
                    reflsvc.decision_function(x),
                    atol=1e-2,
                )
            )

            inputX = torch.from_numpy(x).to(device)
            inputX.requires_grad = True
            self.assertTrue(
                gradcheck(lsvc.decision_function, inputX, eps=1e-6, atol=1e-3)
            )

            self.assertTrue(
                np.allclose(
                    lsvc.predict(torch.from_numpy(x).to(device)).cpu().numpy(),
                    reflsvc.predict(x),
                    atol=1e-2,
                )
            )
            self.assertTrue(gradcheck(lsvc.predict, inputX, eps=1e-6, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
