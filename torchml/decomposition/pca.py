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

    * `n_components` : int, default=None
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)
    * `svd_solver` : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        The algorithm that runs SVD.
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed.

    ## Example

    ~~~python
    import torch
    from torchml.decomposition import PCA
    X = torch.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    ~~~
    """

    def __init__(
        self,
        *,
        n_components=None,
        svd_solver="auto",
    ):
        super(PCA, self).__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver

    def fit(self, X):
        """
        ## Description

        Fit the model with X.

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

        # Set n_components
        if self.n_components is None:
            self.n_components = min(X.shape)

        # Select the solver
        if self.svd_solver == "auto":
            if (
                X.shape[0] > 500
                and X.shape[1] > 500
                and self.n_components < 0.8 * min(X.shape)
            ):
                self.svd_solver = "randomized"
            else:
                self.svd_solver = "full"

        # Compute SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=(self.svd_solver == "full"))
        # U, S, V = torch.svd(X, some=(self.svd_solver == "randomized"))

        # flip eigenvectors' sign to enforce deterministic output
        max_abs_cols = U.abs().argmax(dim=0)
        U *= torch.sign(U[max_abs_cols, range(U.shape[1])])

        self.U = U
        self.S = S
        return self

    def transform(self, X):
        """
        ## Description

        Apply dimensionality reduction to X.

        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        pca = PCA()
        X_reduced = pca.fit(X).transform(X)
        ~~~
        """
        return self.U[:, : self.n_components] * self.S[: self.n_components]

    def fit_transform(self, X):
        """
        ## Description

        Fit the model with X and apply the dimensionality reduction on X.

        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        pca = PCA()
        X_reduced = pca.fit_transform(X)
        ~~~
        """
        self.fit(X)
        return self.transform(X)
