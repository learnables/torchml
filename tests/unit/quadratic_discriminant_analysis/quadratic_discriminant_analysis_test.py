import unittest
import numpy as np
import torch
import torchml as ml
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


BSZ = 128
DIM = 5


class TestQuadraticDiscriminantAnalysis(unittest.TestCase):
    def test_fit(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randint(low=0, high=10, size=BSZ)

        ref = QuadraticDiscriminantAnalysis()
        ref.fit(X, y)
        ref_preds = ref.predict(X)

        model = ml.discriminant_analysis.QuadraticDiscriminantAnalysis()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_preds = model.predict(torch.from_numpy(X))

        self.assertTrue(
            np.allclose(ref_preds, model_preds.numpy())
        )
    
    def test_decision_function(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randint(low=0, high=10, size=BSZ)

        ref = QuadraticDiscriminantAnalysis()
        ref.fit(X, y)
        ref_dec_func = ref.decision_function(X)

        model = ml.discriminant_analysis.QuadraticDiscriminantAnalysis()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_dec_func = model.decision_function(torch.from_numpy(X))
        
        self.assertTrue(
            np.allclose(ref_dec_func, model_dec_func.numpy())
        )
    
    def test_predict_proba(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randint(low=0, high=10, size=BSZ)

        ref = QuadraticDiscriminantAnalysis()
        ref.fit(X, y)
        ref_prob = ref.predict_proba(X)

        model = ml.discriminant_analysis.QuadraticDiscriminantAnalysis()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_prob = model.predict_proba(torch.from_numpy(X))
        
        self.assertTrue(
            np.allclose(ref_prob, model_prob.numpy())
        )
        
    def test_predict_log_proba(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randint(low=0, high=10, size=BSZ)

        ref = QuadraticDiscriminantAnalysis()
        ref.fit(X, y)
        ref_log_prob = ref.predict_log_proba(X)

        model = ml.discriminant_analysis.QuadraticDiscriminantAnalysis()
        model.fit(torch.from_numpy(X), torch.from_numpy(y))
        model_log_prob = model.predict_log_proba(torch.from_numpy(X))
        
        self.assertTrue(
            np.allclose(ref_log_prob, model_log_prob.numpy())
        )


if __name__ == "__main__":
    unittest.main()
