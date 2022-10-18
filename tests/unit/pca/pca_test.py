import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.decomposition as decomposition


BSZ = 5
DIM = 5
N_COMPONENTS = 2

torch.manual_seed(100)
np.random.seed(100)


class TestPCA(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM)

        # match the default behavior of torch
        ref = decomposition.PCA(
            n_components=N_COMPONENTS,
            svd_solver="randomized",
            iterated_power=2,
            n_oversamples=0,
            power_iteration_normalizer="QR",
        )
        ref_transformed = ref.fit_transform(X)

        model = ml.decomposition.PCA(n_components=N_COMPONENTS)
        model_transformed = model.fit_transform(torch.from_numpy(X)).numpy()

        try:
            # to compensate for the randomness from drawing from a normal distribution as Q in the randomized SVD
            self.assertTrue(np.allclose(model_transformed, ref_transformed, rtol=1))
        except AssertionError:
            import ipdb

            ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
