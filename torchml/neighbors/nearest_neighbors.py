
import torch

import torchml as ml

import sklearn.neighbors

sklearn.neighbors.NearestNeighbors().radius_neighbors()


class NearestNeighbors(ml.Model):
    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        raise (NotImplementedError())

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        raise (NotImplementedError())

    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        raise (NotImplementedError())
