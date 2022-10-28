""" A PyTorch implementation of Principal Component Analysis (PCA) """
import torch

import torchml as ml


class PCA(ml.Model):

    """
    ## Description

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    The input data is centered but not scaled for each feature before applying the SVD.

    ## References

    1. The scikit-learn [documentation page] (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    2. Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009.

    ## Arguments

    * n_components (int), default=None
    * svd_solver (string), default="full"

    ## Example

    ~~~python
    >>> import torch
    >>> from torchml.decomposition import PCA
    >>> X = torch.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(n_components=2)
    ~~~
    """

    def __init__(
        self,
        *,
        n_components=None,
        svd_solver="full",
    ):
        super(PCA, self).__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver

    def fit(self, X):
        """Fit the model with X.
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        pca = PCA()
        pca.fit(X)
        ~~~
        """
        # Center data
        X -= X.mean(dim=0)

        # Compute SVD
        randomized = True if self.svd_solver == "randomized" else False
        U, S, V = torch.svd(X, some=randomized)

        # flip eigenvectors' sign to enforce deterministic output
        max_abs_cols = U.abs().argmax(dim=0)
        U *= torch.sign(U[max_abs_cols, range(U.shape[1])])

        self.U = U
        self.S = S
        return self

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        pca = PCA()
        X_reduced = pca.fit_transform(X)
        ~~~
        """
        self.fit(X)
        return self.U[:, : self.n_components] * self.S[: self.n_components]
