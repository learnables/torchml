import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors

BSZ = 128
DIM = 5


class Testkneighbors(unittest.TestCase):

    def test_fit(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randn(1, DIM)

        ref = neighbors.NearestNeighbors()
        ref.fit(X)
        test = ref.kneighbors(y)

        model = ml.neighbors.NearestNeighbors()
        model.fit(torch.from_numpy(X))
        res = model.kneighbors(torch.from_numpy(y))

        self.assertTrue(np.allclose(test[0], res[0].numpy()))
        self.assertTrue(np.allclose(test[1], res[1].numpy()))


if __name__ == '__main__':
    unittest.main()
