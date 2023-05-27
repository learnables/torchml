import numbers
import warnings
from typing import Tuple, Any

import torch
from torch import Tensor

import torchml as ml


class KNeighborsClassifier(ml.Model):
    """
    ## Description

    Unsupervised learner for implementing KNN Classifier.

    ## References

    1. Fix, E. and Hodges, J.L. (1951) Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties. Technical Report 4, USAF School of Aviation Medicine, Randolph Field.
    2. MIT 6.034 Artificial Intelligence, Fall 2010, [10. Introduction to Learning, Nearest Neighbors](https://www.youtube.com/watch?v=09mb78oiPkA)
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) for KNeighborsClassifier.

    ## Arguments

    * `n_neighbors` (int, default=5):
        Number of neighbors to use by default for kneighbors queries.

    * `weights` (str {'uniform', 'distance'} or callable, default='uniform'):
        * 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        * 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
        * [callable] : not implemented yet

    * `algorithm` (str, default='auto'):
        Dummy variable to keep consistency with SKlearn's API, always 'brute' for now.

    * `leaf_size` (int, default=30)
        Dummy variable to keep consistency with SKlearn's API.

    * `metric` (str or callable, default=’minkowski’):
        No metric supprt right now, dummy variable and always minkowski

    * `p` (int, default=2):
        No metric supprt right now, dummy variable and always 2

    * `metric_paramsdict` (default=None):
        Dummy variable to mimic the sklearn API

    * `n_jobs` (int, default=None):
        Dummy variable to mimic the sklearn API

    ## Example

    ~~~python
    import numpy as np
    import torchml as ml
    samples = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    point = np.array([1.1])
    neigh = ml.neighbors.KNeighborsClassifier(n_neighbors=3)
    neigh.fit(torch.from_numpy(samples), torch.from_numpy(y))
    neigh.predict(torch.from_numpy(point))
    neigh.predict_proba(torch.from_numpy(point))
    ~~~
    """

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
        self.n_neighbors = (n_neighbors,)
        self.algorithm = (algorithm,)
        self.leaf_size = (leaf_size,)
        self.metric = (metric,)
        self.p = (p,)
        self.metric_params = (metric_params,)
        self.n_jobs = (n_jobs,)
        self.weights = weights
        self.KNN = ml.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        ## Description

        Initialize the class with training sets

        ## Arguments
        * `X` (torch.Tensor): the training set
        * `y` (torch.Tensor, default=None): dummy variable used to maintain the scikit-learn API consistency

        """
        self.KNN.fit(X)
        self.weights = self._check_weights(weights=self.weights)
        device = X.device
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
        self._y = torch.empty(size=y.shape, dtype=torch.long, device=device)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = torch.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        ## Description

        Predict the class labels for the provided data.

        ## Arguments

        * `X` (torch.Tensor): the target point
        """
        device = X.device
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

        y_pred = torch.empty(
            (n_queries, n_outputs), dtype=classes_[0].dtype, device=device
        )

        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = torch.mode(_y[neigh_ind, k], dim=1)
            else:
                mode, _ = self._weighted_mode(_y[neigh_ind, k], weights)
            mode = torch.tensor(mode.ravel(), dtype=torch.long)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        ## Description

        Return probability estimates for the test data X.

        ## Arguments

        * `X` (torch.Tensor): the target point
        """
        device = X.device
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

        n_queries = len(X)

        weights = self._get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = torch.ones_like(neigh_ind, device=device)

        all_rows = torch.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = torch.zeros((n_queries, len(classes_k)), device=device)

            for i, idx in enumerate(pred_labels.T):
                proba_k[all_rows, idx] += weights[:, i]

            normalizer = proba_k.sum(dim=1)[:, None]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities

    def kneighbors(
        self,
        X: torch.Tensor = None,
        n_neighbors: int = None,
        return_distance: bool = True,
    ) -> any:
        """
        ## Description

        Computes the knearest neighbors and returns those k neighbors

        ## Arguments

        * `X` (torch.Tensor): the target point
        * `n_neighbors` (int, default=None): optional argument to respecify the parameter k in k nearest neighbors
        * `return_distance` (bool, default=True): returns the distances to the neighbors if true
        """
        return self.KNN.kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=return_distance
        )

    def _check_weights(self, weights: str) -> torch.Tensor:
        if weights not in (None, "uniform", "distance") and not callable(weights):
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )
        return weights

    def _get_weights(self, dist: torch.Tensor, weights: str) -> torch.Tensor:
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

    def _weighted_mode(
        self, a: torch.Tensor, w: torch.Tensor
    ) -> tuple[Tensor | Any, Tensor | Any]:
        device = a.device
        res = torch.empty(0, device=device)
        resi = torch.empty(0, device=device)
        for i, x in enumerate(a):
            res1 = self._weighted_mode_util(x, w)
            res = torch.cat((res, torch.tensor([res1[0]], device=device)))
            resi = torch.cat((resi, torch.tensor([res1[1]], device=device)))
        return res, resi

    def _weighted_mode_util(
        self, a: torch.Tensor, w: torch.Tensor
    ) -> tuple[Any, Tensor]:
        device = a.device
        unique_a = torch.unique(a)
        res = torch.empty(0, device=device)
        for i, x in enumerate(unique_a):
            cleared = (a == x).float()
            cleared_weights = cleared * w
            sum = torch.sum(cleared_weights)
            res = torch.cat((res, torch.tensor([sum], device=device)))
        return unique_a[torch.argmax(res)], torch.max(res)
