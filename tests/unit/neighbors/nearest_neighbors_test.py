import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors

BSZ = 128
DIM = 5


class Testkneighbors(unittest.TestCase):

    def test_kneighbors(self):
        for i in range(1, 200, 1):
            X = np.random.randn(BSZ, DIM)
            y = np.random.randn(1, DIM)
            ref = neighbors.NearestNeighbors(p=i)
            ref.fit(X)
            test = ref.kneighbors(y)

            model = ml.neighbors.NearestNeighbors(p=i)
            model.fit(torch.from_numpy(X))
            res = model.kneighbors(torch.from_numpy(y))

            # return distance is true
            self.assertTrue(np.allclose(test[0], res[0].numpy()))
            self.assertTrue(np.allclose(test[1], res[1].numpy()))

            ref = neighbors.NearestNeighbors(p=i)
            ref.fit(X)
            test = ref.kneighbors(y, return_distance=False)

            model = ml.neighbors.NearestNeighbors(p=i)
            model.fit(torch.from_numpy(X))
            res = model.kneighbors(torch.from_numpy(y), return_distance=False)

            # return distance is false
            self.assertTrue(np.allclose(test, res.numpy()))


if __name__ == '__main__':
    unittest.main()
