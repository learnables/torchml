from msilib.schema import Feature
import unittest
import numpy as np
#import torch
import torchml as ml
import sklearn.neighbors as neighbors

CLS = 10
FEA = 3


class Testcentroids(unittest.TestCase):

    def test_kneighbors(self):
        X = np.random.randn(CLS, Feature)
        y = np.random.randn(1, CLS)

        for i in range(1, 100):
            ref = neighbors.NearestCentroid()
            cent = ml.neighbors.NearestCentroid()
            ref.fit(X,y)
            cent.fit(X,y)
            samp = np.random.randn(1,FEA)
            self.assertTrue(ref.predict(samp) == cent.predict(samp))

if __name__ == '__main__':
    unittest.main()