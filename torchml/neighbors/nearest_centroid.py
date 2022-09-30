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
        X.to(torch.float64)
        y.to(torch.float64)

        if self.metric == "precomputed":
            raise ValueError("Precomputed is not supported.")

        n_samples, n_features = X.shape

        # y_ind: idx, y_classes: unique tensor
        self.y_type = y[0].dtype
        y_classes, y_ind = torch.unique(y,sorted=True,return_inverse=True)
        #print ("y_ind: {} and y_classes: {}".format(y_ind,y_classes))
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
            #nk[cur_class] = torch.sum(center_mask)  
            #print("cur_class: {}".format(cur_class))
            #print("centermask: {}".format(center_mask))
            #print("x:{}".format(X))

            # XXX: Update other averaging methods according to the metrics.
            if self.metric != "euclidean":
                raise ValueError("Only Euclidian is supported.")

            else:

                #print("haha")
                #print("EUCLIDEAN:")
                #print("centroids b4 add: {}".format(self.centroids_))
                #print("X[center_mask]: {} of type {}".format(X[center_mask],X[center_mask].dtype))

                self.centroids_[cur_class] = torch.nanmean(X[center_mask].to(torch.float64))
                #print("After add, centroid[{}] becomes {}".format(cur_class,self.centroids_[cur_class]))
                #self.centroids_[cur_class] = torch.mean(X[0:n_samples-1,center_mask], axis=0)
                #print(X[0:n_samples-1,center_mask].mean(axis=0))

        return self
        
    def predict(self, X: torch.tensor)->torch.tensor:
        #("PREDICTING")
        #print(X)
        #print(self.centroids_)
        #TODO: check if fitted
        #check_is_fitted(self)
        
        if X is None or X.size(dim=0) < 1:
            print("Warning: check input size")

        #print("X size:",X.size())
        ret = torch.empty(X.size(dim=0))
        # TODO: DATA validate
        # X = self._validate_data(X, accept_sparse="csr", reset=False)
        for i in range(X.size(dim=0)):
            ret[i]= self.classes_[torch.argmin(torch.nn.PairwiseDistance(p=2)(X[i],self.centroids_))] 
        
        return ret.to(self.y_type)
       