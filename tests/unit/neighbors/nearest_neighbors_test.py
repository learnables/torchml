import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors
from torch.autograd import gradcheck

BSZ = 128
DIM = 5


class Testkneighbors(unittest.TestCase):
    def test_kneighbors_classifier(self):
        for i in range(2):
            device = torch.device('cuda' if torch.cuda.is_available() and i else 'cpu')
            for i in range(1, 5, 1):
                X = np.random.randn(BSZ, DIM)
                y = np.random.randn(5, DIM)
                ref = neighbors.NearestNeighbors(p=i)
                ref.fit(X)
                test = ref.kneighbors(y)

                model = ml.neighbors.NearestNeighbors(p=i)
                model.fit(torch.from_numpy(X).to(device))
                res = model.kneighbors(torch.from_numpy(y).to(device))

                # return distance is true
                self.assertTrue(np.allclose(test[0], res[0].cpu().numpy()))
                self.assertTrue(np.allclose(test[1], res[1].cpu().numpy()))
                inputY = torch.from_numpy(y).to(device)
                inputY.requires_grad = True
                self.assertTrue(gradcheck(model.kneighbors, inputY, eps=1e-6, atol=1e-3))

                ref = neighbors.NearestNeighbors(p=i)
                ref.fit(X)
                test = ref.kneighbors(y, return_distance=False)

                model = ml.neighbors.NearestNeighbors(p=i)
                model.fit(torch.from_numpy(X).to(device))
                res = model.kneighbors(torch.from_numpy(y).to(device), return_distance=False)

                # return distance is false
                self.assertTrue(np.allclose(test, res.cpu().numpy()))
                self.assertTrue(gradcheck(model.kneighbors, inputY, eps=1e-6, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
