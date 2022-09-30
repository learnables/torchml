import torch

import torchml as ml


class LinearRegression(ml.Model):

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
