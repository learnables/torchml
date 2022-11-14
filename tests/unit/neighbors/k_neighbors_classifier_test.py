import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors
from torch.autograd import gradcheck


BSZ = 1000
DIM = 50


class TestkneighborsClassifier(unittest.TestCase):
    def test_knn_classifier(self):
        for i in range(2):
            device = torch.device("cuda" if torch.cuda.is_available() and i else "cpu")
            for i in range(1, 5, 1):
                X = np.random.randn(BSZ, DIM)
                y = np.random.randint(low=-100, high=100, size=BSZ)
                p = np.random.randn(5, DIM)

                ref = neighbors.KNeighborsClassifier(
                    weights="distance" if i % 2 else "uniform", p=i
                )
                ref.fit(X, y)
                refr = ref.predict(p)
                refp = ref.predict_proba(p)

                test = ml.neighbors.KNeighborsClassifier(
                    weights="distance" if i % 2 else "uniform", p=i
                )
                test.fit(torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))
                inputP = torch.from_numpy(p).to(device).double()
                inputP.requires_grad = True

                testr = test.predict(torch.from_numpy(p).to(device))
                testp = test.predict_proba(torch.from_numpy(p).to(device))
                self.assertTrue(gradcheck(test.predict, inputP, eps=1e-6, atol=1e-3))
                self.assertTrue(np.allclose(refr, testr.cpu().numpy()))
                self.assertTrue(np.allclose(refp, testp.cpu().numpy()))

                refr2 = ref.kneighbors(p)
                testr2 = test.kneighbors(torch.from_numpy(p).to(device))
                self.assertTrue(gradcheck(test.kneighbors, inputP, eps=1e-6, atol=1e-3))
                self.assertTrue(np.allclose(refr2[0], testr2[0].cpu().numpy()))
                self.assertTrue(np.allclose(refr2[1], testr2[1].cpu().numpy()))


if __name__ == "__main__":
    unittest.main()
