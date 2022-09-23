import numbers
import warnings

import torch
import torchml as ml


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
        self._y = None
        self.classes_ = None
        self.outputs_2d_ = None
        self.n_neighbors = n_neighbors,
        self.algorithm = algorithm,
        self.leaf_size = leaf_size,
        self.metric = metric,
        self.p = p,
        self.metric_params = metric_params,
        self.n_jobs = n_jobs,
        self.weights = weights
        self.KNN = ml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p,
                                                 metric=metric, metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.KNN.fit(X)
        self.weights = self._check_weights(weights=self.weights)
        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn(
                    "A column-vector y was passed when a "
                    "1d array was expected. Please change "
                    "the shape of y to (n_samples,), for "
                    "example using ravel().",
                )

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        self.classes_ = []
        self._y = torch.empty(size=y.shape, dtype=torch.long)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = torch.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

    def predict(self, X: torch.Tensor):
        if self.weights == "uniform":
            neigh_ind = self.KNN.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.KNN.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = len(X)
        weights = self._get_weights(neigh_dist, self.weights)

        y_pred = torch.empty((n_queries, n_outputs), dtype=classes_[0].dtype)

        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = torch.mode(_y[neigh_ind, k], dim=1)
            else:
                mode, _ = self._weighted_mode(_y[neigh_ind, k], weights)
            mode = torch.asarray(mode.ravel(), dtype=torch.long)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        return self.KNN.kneighbors(X=X, n_neighbors=n_neighbors, return_distance=return_distance)

    def _check_weights(self, weights: str):
        if weights not in (None, "uniform", "distance") and not callable(weights):
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )
        return weights

    def _get_weights(self, dist: torch.Tensor, weights: str):
        if weights in (None, "uniform"):
            return None
        elif weights == "distance":
            dist = 1.0 / dist
            inf_mask = torch.isinf(dist)
            inf_row = torch.any(inf_mask, dim=1)
            dist[inf_row] = inf_mask[inf_row].double()
            return dist
        else:
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )

    def _weighted_mode(self, a: torch.Tensor, w: torch.Tensor):
        res = torch.empty(0)
        resi = torch.empty(0)
        for i, x in enumerate(a):
            res1 = self._weighted_mode_util(x, w)
            res = torch.cat((res, torch.tensor([res1[0]])))
            resi = torch.cat((resi, torch.tensor([res1[1]])))
        return res, resi

    def _weighted_mode_util(self, a: torch.Tensor, w: torch.Tensor):
        unique_a = torch.unique(a)
        res = torch.empty(0)
        for i, x in enumerate(unique_a):
            cleared = (a == x).float()
            cleared_weights = cleared * w
            sum = torch.sum(cleared_weights)
            res = torch.cat((res, torch.tensor([sum])))
        return unique_a[torch.argmax(res)], torch.max(res)
