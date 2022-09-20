import numbers
import torch
import torchml as ml
import sklearn.neighbors as n

class KNeighborsClassifier(ml.Model):
    def __init__(
            self,
            n_neighbors=5,
            *,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
    ):
        super(KNeighborsClassifier, self).__init__()
        self.n_neighbors = n_neighbors,
        self.algorithm = algorithm,
        self.leaf_size = leaf_size,
        self.metric = metric,
        self.p = p,
        self.metric_params = metric_params,
        self.n_jobs = n_jobs,
        self.weights = weights
