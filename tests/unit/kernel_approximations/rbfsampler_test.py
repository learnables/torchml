import torch
import math
import numpy as np
import scipy.sparse as sp
import unittest

class RBFSampler():
    def __init__(
        self,
        *,
        gamma: float = 1.0,
        n_components: int = 100,
        random_normal_matrix = None,
        random_uniform_matrix = None
    ):
        super(RBFSampler, self).__init__()
        self.gamma = gamma
        self.n_components = n_components
        self.random_normal_matrix = random_normal_matrix
        self.random_uniform_matrix = random_uniform_matrix

    
    def fit(self, X: torch.Tensor, y: torch.Tensor or None = None):
        n_features = X.size(dim=1)
        
        self.random_weights_ = math.sqrt(self.gamma * 2) * self.random_normal_matrix
        self.random_offset_ = self.random_uniform_matrix

        self._n_features_out = self.n_components
        return self
    
    def transform(self, X: torch.Tensor):
        projection = torch.mm(X, self.random_weights_)
        projection += self.random_offset_
        projection = torch.cos(projection)
        projection *= math.sqrt(2.0) / math.sqrt(self.n_components)
        return projection
    
    def fit_transform(self, X: torch.Tensor, y: torch.Tensor or None = None):
        return self.fit(X, y).transform(X)


class RBFSamplerSklearn():
    def __init__(self, *, gamma=1.0, n_components=100, random_normal_matrix=None, random_uniform_matrix=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_normal_matrix = random_normal_matrix
        self.random_uniform_matrix = random_uniform_matrix
        
    def safe_sparse_dot(self, a, b, *, dense_output=False):
        if a.ndim > 2 or b.ndim > 2:
            if sp.issparse(a):
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif sp.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = np.dot(a, b)
        else:
            ret = a @ b

        if (
            sp.issparse(a)
            and sp.issparse(b)
            and dense_output
            and hasattr(ret, "toarray")
        ):
            return ret.toarray()
        return ret

    def fit(self, X, y=None):
        n_features = X.shape[1]
        
        self.random_weights_ = np.sqrt(2 * self.gamma) * self.random_normal_matrix

        self.random_offset_ = self.random_uniform_matrix

        self._n_features_out = self.n_components
        return self

    def transform(self, X):
        projection = self.safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.0) / np.sqrt(self.n_components)
        return projection


BSZ = 128
DIM = 5

class TestRBFSampler(unittest.TestCase):

    def test_fit_transform(self):
        X = np.random.randn(BSZ, DIM)
        y = np.random.randn(BSZ)

        torch.random.seed()
        random_normal_matrix = torch.empty((5, 100)).normal_()
        random_uniform_matrix = torch.empty(100).uniform_(0, 2 * math.pi)

        random_normal_matrix = random_normal_matrix.type(torch.float64)
        random_uniform_matrix = random_uniform_matrix.type(torch.float64)

        rbf_feature = RBFSampler(gamma=1, random_normal_matrix=random_normal_matrix, random_uniform_matrix=random_uniform_matrix)
        model_X_feature = rbf_feature.fit_transform(torch.from_numpy(np.asarray(X)))

        rbf_feature_sklearn = RBFSamplerSklearn(gamma=1, random_normal_matrix=random_normal_matrix.numpy(), random_uniform_matrix=random_uniform_matrix.numpy())
        ref_X_feature = rbf_feature_sklearn.fit(np.asarray(X), np.asarray(y)).transform(np.asarray(X))
        
        self.assertTrue(np.allclose(ref_X_feature, model_X_feature.numpy()))
