import unittest
import copy
import numpy as np
import torch
import torchml as ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


BSZ = 128
DIM = 5


class TestLinearDiscriminantAnalysis(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM).astype(np.float32)
        y = np.random.randint(low=0, high=50, size=BSZ)

        ref = LinearDiscriminantAnalysis()
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.discriminant_analysis.LinearDiscriminantAnalysis()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))

        self.assertTrue(
            np.allclose(ref_preds, model_preds.numpy())
        )

    def test_fit_two_classes(self):
        X = np.random.randn(BSZ, DIM).astype(np.float32)
        y = np.random.randint(low=0, high=2, size=BSZ)

        ref = LinearDiscriminantAnalysis()
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.discriminant_analysis.LinearDiscriminantAnalysis()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))

        self.assertTrue(
            np.allclose(ref_preds, model_preds.numpy())
        )

    def test_transform(self):
        X = np.random.randn(BSZ, DIM).astype(np.float32)
        y = np.random.randint(low=0, high=50, size=BSZ)
        
        X_copy = copy.deepcopy(X)

        ref = LinearDiscriminantAnalysis()
        ref.fit(X, y)
        ref_transformed_X = ref.transform(X)

        model = ml.discriminant_analysis.LinearDiscriminantAnalysis()
        model.fit(torch.from_numpy(X_copy), torch.from_numpy(y))
        model_transformed_X = model.transform(torch.from_numpy(X_copy))

        self.assertTrue(
            np.allclose(ref_transformed_X, model_transformed_X.numpy(), atol=1e-4)
        )
    
    def test_transform_two_classes(self):
        X = np.random.randn(BSZ, DIM).astype(np.float32)
        y = np.random.randint(low=0, high=1, size=BSZ)
        
        X_copy = copy.deepcopy(X)

        ref = LinearDiscriminantAnalysis()
        ref.fit(X, y)
        ref_transformed_X = ref.transform(X)

        model = ml.discriminant_analysis.LinearDiscriminantAnalysis()
        model.fit(torch.from_numpy(X_copy), torch.from_numpy(y))
        model_transformed_X = model.transform(torch.from_numpy(X_copy))

        self.assertTrue(
            np.allclose(ref_transformed_X, model_transformed_X.numpy(), atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
