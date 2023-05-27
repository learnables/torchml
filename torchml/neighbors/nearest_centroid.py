import torch

import torchml as ml


class NearestCentroid(ml.Model):

    """
    ## Description

    Implementation of scikit-learn's Nearest centroid APIs using pytorch.
    Euclidean metric by default.

    ## References

    1. The scikit-learn [documentation page] (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html) for nearest centroids.
    2. M. Thulasidas, "Nearest Centroid: A Bridge between Statistics and Machine Learning," 2020 IEEE International Conference on Teaching, Assessment, and Learning for Engineering (TALE), 2020, pp. 9-16, doi: 10.1109/TALE48869.2020.9368396.

    ## Arguments

    * `metric` (str or callable, default="euclidean"):
        Metric to use for distance computation. Only Euclidiean metric is supported for now.

    * `shrink_threshold` (float, default=None):
        Threshold for shrinking centroids to remove features. Not supported for now.

    ## Example

    ~~~python
    import numpy as np
    import torchml as ml
    X = np.array([[3,2],[1,8],[3,5],[9,7],[7,7]])
    y = np.array([9,1,9,8,3])
    samp = np.array([[6,1],[8,9],[5,3],[8,2],[9,8]])
    torchX = torch.from_numpy(X)
    torchy = torch.from_numpy(y)
    centroid = ml.neighbors.NearestCentroid()
    centroid.fit(torchX,torchy)
    output = centroid.predict(torch.from_numpy(samp)).numpy()
    ~~~
    """

    def __init__(self, metric="euclidean", shrink_threshold=None):
        if metric != "euclidean":
            raise ValueError(
                "The only metric supported is euclidean for now; got" + metric
            )
        if shrink_threshold is not None:
            raise ValueError("shrink_threshold is not supported for now.")
        self.metric = metric
        self.shrink_threshold = shrink_threshold

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        ## Description

        Fit the NearestCentroid model according to the given training data.

        ## Arguments

        * `X` (torch.Tensor): array-like, sparse matrix of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features
        * `y` (torch.Tensor): array-like of shape (n_samples,) Target values
        """

        n_samples, n_features = X.shape

        # y_ind: idx, y_classes: unique tensor
        self.y_type = y[0].dtype
        y_classes, y_ind = torch.unique(y, sorted=False, return_inverse=True)
        self.classes_ = classes = y_classes
        n_classes = classes.size(dim=0)
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )

        # Mask mapping each class to its members.
        self.centroids_ = torch.empty(
            (n_classes, n_features), dtype=X.dtype, device=torch.device("cpu")
        )
        # Number of clusters in each class.

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class

            if self.metric != "euclidean":
                raise ValueError("Only Euclidian is supported.")

            else:
                self.centroids_[cur_class] = torch.mean(X[center_mask], dim=0)

        return self

    def predict(self, X: torch.tensor) -> torch.tensor:
        """
        ## Description

        Computes the classes of the sample data

        ## Arguments

        * `X` (torch.Tensor): the sample data, each with n-features

        ## Return

        * (torch.Tensor): the predicted classes

        """
        if X is None or X.size(dim=0) < 1:
            print("Warning: check input size")

        ret = torch.empty(X.size(dim=0))

        for i in range(X.size(dim=0)):
            ret[i] = self.classes_[
                torch.argmin(torch.nn.PairwiseDistance(p=2)(X[i], self.centroids_))
            ]

        # return ret.to(self.y_type)
        return ret
