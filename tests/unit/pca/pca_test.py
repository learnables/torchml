import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.decomposition as decomposition


BSZ = 128
DIM = 5


class TestPCA(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM)

        ref = decomposition.PCA(n_components=2)
        ref.fit(X)
        ref_explained_variance = ref.explained_variance_

        model = ml.decomposition.PCA(n_components=2)
        model.fit(torch.from_numpy(X))
        model_explained_variance = model.explained_variance_

        self.assertTrue(
            np.allclose(ref_explained_variance, model_explained_variance.numpy())
        )
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
