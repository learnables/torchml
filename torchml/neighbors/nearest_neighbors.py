import numbers
import torch
import torchml as ml


def _distance_matrix(x: torch.Tensor, y: torch.Tensor, p=2,
                     dist_func=None) -> torch.Tensor:  # pairwise distance of vectors

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if dist_func is None:
        return torch.pow(x - y, p).sum(2)
    return dist_func(x, y)


class NearestNeighbors(ml.Model):
    """
    <a href="https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3" class="source-link">[Source]</a>

    ## Description

    neighbor searches

    ## References

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

    ## Arguments

    n_neighbors: int, default=5
        Number of neighbors to use by default for kneighbors queries.

    radius: float, default=1.0
        Range of parameter space to use by default for radius_neighbors queries.

    algorithm: string, default=’auto’
        Dummy variable to mimic the sklearn API

    leaf_size: int, default=30
        Dummy variable to mimic the sklearn API

    metric: str or callable, default=’minkowski’
        No metric supprt right now, dummy variable and always minkowski

    p: int, default=2
        No metric supprt right now, dummy variable and always 2

    metric_paramsdict: default=None
        Dummy variable to mimic the sklearn API

    n_jobs: int, default=None
        Dummy variable to mimic the sklearn API

    ## Example

    ~~~python
    import numpy as np
    import torchml as ml
    samples = np.array([[1, 0, 0], [0, 1, 0], [0, 5, 0], [20, 50, 30]])
    point = np.array([[20, 50, 1]])
    neigh = ml.neighbors.NearestNeighbors(n_neighbors=3)
    neigh.fit(torch.from_numpy(samples))
    neigh.kneighbors(torch.from_numpy(point))
    (tensor([[29.0000, 49.2544, 52.9339]]), tensor([[3, 2, 1]]))
    ~~~
    """

    def __init__(self, *, n_neighbors=5, radius=1.0, algorithm="auto", leaf_size=30, metric="minkowski", p=2,
                 metric_params=None, n_jobs=None):
        super(NearestNeighbors, self).__init__()
        self.train_pts = None
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X: torch.Tensor, y=None):
        self.train_pts = X

    def kneighbors(self, X: torch.Tensor, n_neighbors=None, return_distance=True) -> any:
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, enter integer value"
                % type(n_neighbors)
            )
        elif X is None:
            raise TypeError("X is not specified")
        dist = torch.sqrt(_distance_matrix(X, self.train_pts, self.p, None if isinstance(self.metric, str) else self.metric))
        k = self.n_neighbors if self.n_neighbors <= self.train_pts.size(0) else self.train_pts.size(0)
        knn = torch.topk(dist, k, largest=False)
        if return_distance:
            return knn.values, knn.indices
        return knn.indices

    # TODO: implement radius_neighbors and KNeighborsClassifier
    # def radius_neighbors(
    #         self, X=None, radius=None, return_distance=True, sort_results=False
    # ):
    #     raise (NotImplementedError())
