import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors

BSZ = 10
DIM = 2


class TestkneighborsClassifier(unittest.TestCase):

    def test_classifier(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randint(low=-100, high=100, size=BSZ)

        ref = neighbors.KNeighborsClassifier()
        ref.fit(X, y)
        print(ref.predict([[1.1, 2.1]]))

        test = ml.neighbors.KNeighborsClassifier()
        test.fit(torch.from_numpy(X), torch.from_numpy(y))
        test.predict(torch.FloatTensor([[1.1, 2.1]]))


if __name__ == '__main__':
    unittest.main()
