import unittest
import numpy as np
import torch
import torchml as ml
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


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

        # compute the accuracy of the reference model and the torchml model
        ref_acc = np.sum(ref_preds == y) / y.shape[0]
        model_acc = np.sum(model_preds.numpy() == y) / y.shape[0]
        print("accuracy of the reference model: ", ref_acc)
        print("accuracy of the torchml model: ", model_acc)

        # check that the accuracy of the torchml model is at least 90% of the accuracy of the reference model
        self.assertTrue(model_acc >= 0.9 * ref_acc)


if __name__ == "__main__":
    unittest.main()
