import torch

#import torchml as ml

import math

from scipy import sparse as sp

#from ..utils import LabelEncoder

#import numpy as np

#class NearestCentroid(ml.Model):
class NearestCentroid():


    """
    ## Description

    Implementation of scikit-learn's Nearest centroid APIs using pytorch.

    ## References
    1. Scikit-learn library (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html)

    ## Arguments
    * metric : str or callable, default="euclidean"
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. 
        See the documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ 
        Note that "wminkowski", "seuclidean" and "mahalanobis" are not
        supported.

    * shrink_threshold : float, default=None
        Threshold for shrinking centroids to remove features.


    """


    def __init__(self, metric="euclidean", *, shrink_threshold=None):
        self.metric = metric
        self.shrink_threshold = shrink_threshold

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the NearestCentroid model according to the given training data.
        Parameters
        ----------
        * X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        * y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        if self.metric == "precomputed":
            raise ValueError("Precomputed is not supported.")

        """
        TODO: Validate data?
        # If X is sparse and the metric is "manhattan", store it in a csc
        # format is easier to calculate the median.
        if self.metric == "manhattan":
            X, y = self._validate_data(X, y, accept_sparse=["csc"])
        else:
            X, y = self._validate_data(X, y, accept_sparse=["csr", "csc"])
        #TODO: sparse?
        #is_X_sparse = sp.issparse(X)
        #if is_X_sparse and self.shrink_threshold:
            #raise ValueError("threshold shrinking not supported for sparse input")
        """
        
        #TODO: 
        """
        do we need to check for classification? 
        {"binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences"}
        """
        #check_classification_targets(y)

        n_samples, n_features = X.shape
        #label encoder?
        #le = LabelEncoder()
        #y_ind = le.fit_transform(y)
        # y_ind: idx, y_classes: unique tensor
        y_ind, y_classes = torch.unique(y,sorted=False,return_inverse=True)

        #Class: dict[y in Y] = X
        self.classes_ = classes = y_classes
        n_classes = classes.size(dim=0)
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )

        # Mask mapping each class to its members.
        self.centroids_ = torch.empty((n_classes, n_features), dtype=torch.float64)
        # Number of clusters in each class.
        # TODO: nk used for shrink
        nk = torch.zeros(n_classes)

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class
            nk[cur_class] = torch.sum(center_mask)  
            
            """
            TODO: SPARSE
            #if is_X_sparse:
                #center_mask = torch.where(center_mask)[0]
            """

            # XXX: Update other averaging methods according to the metrics.
            if self.metric == "manhattan":
                self.centroids_[cur_class] = torch.median(X[center_mask], axis=0)
                """
                TODO: check if SPARSE
                # NumPy does not calculate median of sparse matrices.
                if not is_X_sparse:
                    self.centroids_[cur_class] = torch.median(X[center_mask], axis=0)
                else:
                    # csc median needs imp
                    self.centroids_[cur_class] = csc_median_axis_0(X[center_mask])
                """
            else:
                """
                TODO: WARING
                if self.metric != "euclidean":
                    warnings.warn(
                        "Averaging for metrics other than "
                        "euclidean and manhattan not supported. "
                        "The average is set to be the mean."
                    )
                """
                #self.centroids_[cur_class] = X[center_mask].mean(axis=0)
                #self.centroids_[cur_class] = X[0:n_samples-1,center_mask].mean(axis=0)
                #for s in range(n_samples):
                print("haha")
                self.centroids_[cur_class] = torch.nanmean(X[:,center_mask])
                #self.centroids_[cur_class] = torch.mean(X[0:n_samples-1,center_mask], axis=0)
                #print(X[0:n_samples-1,center_mask].mean(axis=0))

        """
        TODO: Shrink Threshhold
        if self.shrink_threshold:
            if torch.all(torch.ptp(X, axis=0) == 0):
                raise ValueError("All features have zero variance. Division by zero.")
            dataset_centroid_ = torch.mean(X, axis=0)

            # m parameter for determining deviation
            m = torch.sqrt((1.0 / nk) - (1.0 / n_samples))
            # Calculate deviation using the standard deviation of centroids.
            variance = (X - self.centroids_[y_ind]) ** 2
            variance = variance.sum(axis=0)
            s = torch.sqrt(variance / (n_samples - n_classes))
            s += torch.median(s)  # To deter outliers from affecting the results.
            mm = m.reshape(len(m), 1)  # Reshape to allow broadcasting.
            ms = mm * s
            deviation = (self.centroids_ - dataset_centroid_) / ms
            # Soft thresholding: if the deviation crosses 0 during shrinking,
            # it becomes zero.
            signs = torch.sign(deviation)
            deviation = torch.abs(deviation) - self.shrink_threshold
            torch.clip(deviation, 0, None, out=deviation)
            deviation *= signs
            # Now adjust the centroids using the deviation
            msd = ms * deviation
            self.centroids_ = dataset_centroid_[torch.newaxis, :] + msd
        """
        return self
        
    def predict(self, X: torch.tensor)->torch.tensor:
        print("PREDICTING")
        print(X)
        print(self.centroids_)
        #TODO: check if fitted
        #check_is_fitted(self)

        # TODO: DATA validate
        # X = self._validate_data(X, accept_sparse="csr", reset=False)
        return self.classes_[
            #pairwise_distances(X, self.centroids_, metric=self.metric).argmin(axis=1)
            #torch.argmin(self.pairwise_distances(X, self.centroids_, metric=self.metric))
            
            torch.argmin(torch.nn.PairwiseDistance(p=2)(X,self.centroids_))
        ]
    
"""
    def pairwise_distances (X, Y, metric) -> torch.tensor:
        #Do we use torch.multiprocessing or {from torch.nn.parallel import DistributedDataParallel}?
        #https://discuss.pytorch.org/t/how-do-i-map-joblibs-parallel-function-to-pytorchs-distributeddataparallel/89095

        #TODO: Consider sparse matrix
        if Y is None:
            Y = X
        else:
            ret = torch.empty(Y.size)

            if metric == "euclidean":
                for i in range(Y.size):
                    y = Y[i]
                    XX = torch.einsum("ij,ij->i",X,X)
                    YY = torch.einsum("ij,ij->i",y,y)
                    ret[i] = XX+YY+(-2)*torch.cross(X,y.T)
            

        return ret


def LabelEncoder():
    
    <a class="source-link" href="">https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/preprocessing/_label.py#L20</a>

    Simple label encoder for 1d data 

    ## Arguments

    * `msg` (string): The message to be printed.

    ## Attributes
    * classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    ## Example

    ~~~python
    le = LabelEncoder();
    y_ind = le.fit_tranform()
    le._classes :.. 


    ~~~

    def fit(self, y:torch.tensor):

        self.classes_ = {}
        self.classes_[a]
    """