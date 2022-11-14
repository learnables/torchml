import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors
from torch.autograd import gradcheck

# define numbers of classes & features
SAMPLES = 10
FEA = 10
CLS = 5


class Testcentroids(unittest.TestCase):
    def test_kneighbors(self):
        for i in range(2):
            device = torch.device('cuda' if torch.cuda.is_available() and i else 'cpu')
            for i in range(100):
                X = np.random.randn(SAMPLES, FEA)
                y = np.random.randint(1, CLS, size=SAMPLES)
                torchX = torch.from_numpy(X).to(device)
                torchy = torch.from_numpy(y).to(device)
                ref = neighbors.NearestCentroid()
                cent = ml.neighbors.NearestCentroid()
                ref.fit(X, y)
                cent.fit(torchX, torchy)
                samp = np.random.randn(SAMPLES, FEA)
                refres = ref.predict(samp)
                centres = cent.predict(torch.from_numpy(samp).to(device)).cpu().numpy()
                self.assertTrue(np.array_equal(refres, centres))
                inputSamp = torch.from_numpy(samp).to(device)
                inputSamp.requires_grad = True
                self.assertTrue(gradcheck(cent.predict, inputSamp, eps=1e-6, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
