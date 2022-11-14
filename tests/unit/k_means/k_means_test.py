import unittest
import numpy as np
import torch
import torchml as ml
#from .... import torchml as ml
import sklearn.neighbors as neighbors
from sklearn.cluster import KMeans
#.tests.unit.kmeans.py
# define numbers of classes & features
SAMPLES = 10
FEA = 10
CLS = 5


class Testcentroids(unittest.TestCase):
    def test_k_means (self):

        X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
        #kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        #print(kmeans.labels_)
        #print(kmeans.labels_.dtype)

        torhcX = torch.from_numpy(X)
        kmeans = ml.k_means.KMeans()
        kmeans.fit(X)
        kmeans.print_label

        """
        for i in range(100):
            X = np.random.randn(SAMPLES, FEA)
            y = np.random.randint(1, CLS, size=SAMPLES)
            torchX = torch.from_numpy(X)
            torchy = torch.from_numpy(y)
            ref = neighbors.NearestCentroid()
            cent = ml.neighbors.NearestCentroid()
            ref.fit(X, y)
            cent.fit(torchX, torchy)
            samp = np.random.randn(SAMPLES, FEA)
            refres = ref.predict(samp)
            centres = cent.predict(torch.from_numpy(samp)).numpy()
            self.assertTrue(np.array_equal(refres, centres))
        """

if __name__ == "__main__":
    unittest.main()
