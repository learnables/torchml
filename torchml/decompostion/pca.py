""" A PyTorch implementation of Principal Component Analysis (PCA) 
References: https://github.com/Linardos/PyTorch/blob/master/TorchPCA.py
"""
import torch

import torchml as ml


class PCA(ml.Model):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    One-liner description (eg, "Ordinary least-square model with bias term").

    You can write a more thorough description here, including references and even equations.
    Solves:

    $$ \\min_w \\vert \\vert  Xw - y \\vert \\vert^2 $$

    ## References

    1. Gauss, for OLS?

    ## Arguments

    * `arg1` (int) - The first argument.

    ## Example

    ~~~python
    ~~~
    """

    def __init__(
        self,
        *,
        n_components=None,
    ):
        super(PCA, self).__init__()
        self.n_components = n_components
        self.eigenpair = None
        self.data = None

    def fit(self, X):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        pca = PCA()
        pca.fit(X_train)
        ~~~
        """
        # center data
        n_rows, n_columns = X.size()
        row_means = torch.mean(X, 1)
        # Expand the matrix in order to have the same shape as X and substract, to center
        for_subtraction = row_means.expand(n_rows, n_columns)
        X = X - for_subtraction

        U, S, V = torch.svd(X)
        eigvecs = U.t()[:, : self.n_components]  # the first n_components will be kept
        y = torch.mm(U, eigvecs)

        # Save variables to the class object, the eigenpair and the centered data
        self.eigenpair = (eigvecs, S)
        self.data = X

        # Get variance explained by singular values
        # var_explained = S**2 / torch.sum(S**2)
        # Total sum of eigenvalues (total variance explained)
        tot = sum(S)
        # Variance explained by each principal component
        self.explained_variance_ = [(i / tot) for i in sorted(S, reverse=True)]

        return y  # return the projected data

    def predict(self, X):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        linreg.predict(X_test)
        ~~~
        """
        return
