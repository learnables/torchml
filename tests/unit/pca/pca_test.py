import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.decomposition as decomposition


BSZ = 128
DIM = 5
N_COMPONENTS = 2


class TestPCA(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM)

        ref = decomposition.PCA(n_components=N_COMPONENTS, svd_solver="full")
        ref_transformed = ref.fit_transform(X)

        model = ml.decomposition.PCA(n_components=N_COMPONENTS, svd_solver="full")
        model_transformed = model.fit_transform(torch.from_numpy(X)).numpy()

        self.assertTrue(np.allclose(ref_transformed, model_transformed))


if __name__ == "__main__":
    unittest.main()
