import torch

import torchml as ml


class Ridge(ml.Model):

    """
    <a href="https://github.com/learnables/torchml/blob/master/torchml/linear_model/ridge.py">[Source]</a>

    ## Description

    Linear regression with L2 penalty term.

    $$ w = (X^TX + \\lambda I)^{-1}X^Ty $$

    * `w` - weights of the linear regression with L2 penalty
    * `X` - variates
    * `λ`- constant that multiplies the L2 term
    * `y` - covariates

    The above equation is the closed-form solution for ridge's objective function

    $$ \\min_w \\frac{1}{2} \\vert \\vert  Xw - y \\vert \\vert^2 I + \\frac{1}{2} \\lambda \\vert \\vert w \\vert \\vert^2 $$

    ## References

    1. Arthur E. Hoerl and Robert W. Kennard's introduction to Ridge Regression [paper](https://www.jstor.org/stable/1271436)
    2. Datacamp Lasso and Ridge Regression Tutorial [tutorial](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression#data-splitting-and-scaling)
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=lasso#sklearn.linear_model.ridge)

    ## Arguments

    * `alpha` (float, default=1.0) - Constant that multiplies the L2 term. alpha must be a non-negative float.
    * `fit_intercept` (bool, default=False) - Whether or not to fit intercept in the model.
    * `normalize` (bool, default=False) - If True, the regressors X will be normalized. normalize will be deprecated in the future.
    * `copy_X` (bool, default=True) - If True, X will be copied.
    * `solver` (string, default='auto') - Different solvers or algorithms to use.

    ## Example

    ~~~python
    ridge = Ridge()
    ~~~
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = False,
        normalize: bool = False,
        copy_X: bool = True,
        solver: str = "auto"
    ):
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.solver = solver

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        ## Description

        Compute the weights for the model given variates and covariates.

        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.

        ## Example

        ~~~python
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ~~~
        """
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        device = X.device

        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=device), X], dim=1)

        # L2 penalty term will not apply when alpha is 0
        if self.alpha == 0:
            self.weight = torch.pinverse(X.T @ X) @ X.T @ y
        else:
            ridge = self.alpha * torch.eye(X.shape[1], device=device)
            # intercept term is not penalized when fit_intercept is true
            if self.fit_intercept:
                ridge[0][0] = 0
            self.weight = torch.pinverse((X.T @ X) + ridge) @ X.T @ y

    def predict(self, X: torch.Tensor):
        """
        ## Description

        Predict covariates by the trained model.

        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge.predict(X_test)
        ~~~
        """
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
        return X @ self.weight
