import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.neighbors as neighbors

# define numbers of classes & features
SAMPLES = 10
FEA = 10
CLS = 5


class Testcentroids(unittest.TestCase):

    def test_kneighbors(self):
        
        for i in range(100):
            X = np.random.randn(SAMPLES, FEA)
            y = np.random.randint(1,CLS,size=SAMPLES)
            torchX = torch.from_numpy(X)
            torchy = torch.from_numpy(y)
            ref = neighbors.NearestCentroid()
            cent = ml.neighbors.NearestCentroid()
            ref.fit(X,y)
            cent.fit(torchX,torchy)
            samp = np.random.randn(SAMPLES, FEA)
            refres=ref.predict(samp)
            centres = cent.predict(torch.from_numpy(samp)).numpy()
            self.assertTrue(np.array_equal(refres,centres))

if __name__ == '__main__':
    unittest.main()