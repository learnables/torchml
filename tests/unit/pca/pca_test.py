import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.decomposition as decomposition


BSZ = 10
DIM = 10
N_COMPONENTS = 2

torch.manual_seed(0)
np.random.seed(0)


class TestPCA(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM)

        ref = decomposition.PCA(n_components=N_COMPONENTS)
        ref_transformed = ref.fit_transform(X)

        model = ml.decomposition.PCA(n_components=N_COMPONENTS)
        model_transformed = model.fit_transform(torch.from_numpy(X)).numpy()

        try:
            self.assertTrue(np.allclose(ref_transformed, model_transformed))
        except AssertionError:
            import ipdb

            ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
