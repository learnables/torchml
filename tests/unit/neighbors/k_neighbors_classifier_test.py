import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors

BSZ = 1000
DIM = 50


class TestkneighborsClassifier(unittest.TestCase):

    def test_knn_classifier(self):
        for i in range(1, 200, 1):
            X = np.random.randn(BSZ, DIM)
            y = np.random.randint(low=-100, high=100, size=BSZ)
            p = np.random.randn(5, DIM)

            ref = neighbors.KNeighborsClassifier(weights="distance" if i % 2 else "uniform", p=i)
            ref.fit(X, y)
            refr = ref.predict(p)
            # ref.predict_proba()

            test = ml.neighbors.KNeighborsClassifier(weights="distance" if i % 2 else "uniform", p=i)
            test.fit(torch.from_numpy(X), torch.from_numpy(y))
            testr = test.predict(torch.from_numpy(p))

            self.assertTrue(np.allclose(refr, testr.numpy()))
            if not np.allclose(refr, testr.numpy()):
                i = 0

            refr2 = ref.kneighbors(p)
            testr2 = test.kneighbors(torch.from_numpy(p))
            self.assertTrue(np.allclose(refr2[0], testr2[0].numpy()))
            self.assertTrue(np.allclose(refr2[1], testr2[1].numpy()))


if __name__ == '__main__':
    unittest.main()
