import torch

import torchml as ml


class LinearRegression(ml.Model):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    Ordinary least-square model with bias term.

    Solves the following optimization problem in closed form:

    $$ \\min_w \\vert \\vert  Xw - y \\vert \\vert^2 $$

    ## References

    1. [Wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares).
    1. Scikit-learn documentation. [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

    ## Arguments

    * `fit_intercept` (bool) - Whether to fit a bias term.
    * `normalized` (str) - Normalizes scheme to use.
    * `copy_X` (bool) - Whether to copy the data X (else, it might be modified in-place).
    * `n_jobs` (int) - Dummy to match the scikit-learn API.
    * `positive` (bool) - Forces the coefficients to be positive when True (not implemented).

    ## Example

    ~~~python
    linreg = LinearRegression(fit_intercept=False)
    ~~~
    """

    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        super(LinearRegression, self).__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

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
        return X @ self.weight

    def fit(self, X, y):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.

        ## Example

        ~~~python
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        ~~~
        """
        if self.fit_intercept:
            raise ValueError("fit_intercept not supported yet.")

        self.weight = torch.pinverse(X.T @ X) @ X.T @ y
        #  self.weight = torch.lstsq(y, X).solution
        self.bias = None
