import torch

import torchml as ml


class Ridge(ml.Model):

    """
    <a href="https://github.com/learnables/torchml/blob/master/torchml/linear_model/ridge.py">[Source]</a>

    ## Description

    Ridge: Linear Regression with L2 Penalty Term

    ## Equation

    $$ w = (X^{T}X + λI)^{-1}X^{T}y $$

    ## References

    1. Arthur E. Hoerl and Robert W. Kennard's introduction to Ridge Regression [paper](https://www.jstor.org/stable/1271436)
    2. Datacamp Lasso and Ridge Regression Tutorial [tutorial](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression#data-splitting-and-scaling)
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=lasso#sklearn.linear_model.ridge)

    ## Arguments

    * `alpha` (float, default=1.0) - Constant that multiplies the L2 term. alpha must be a non-negative float.
    * `fit_intercept` (bool, default=True) - Whether or not to fit intercept in the model.
    * `normalize` (bool, default=False) - If True, the regressors X will be normalized. normalize will be deprecated in the future.
    * `copy_X` (bool, default=True) - If True, X will be copied.
    * `solver` (string, default='auto') - Different solvers or algorithms to use

    ## Example

    ~~~python
    ridge = Ridge()
    ~~~
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=False,
        normalize=False,
        copy_X=True,
        solver='auto'
    ):
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.solver = solver

    def fit(self, X: torch.Tensor, y=None):
        """
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

        if self.fit_intercept:
            raise ValueError('fit_intercept not supported yet.')

        # when alpha == 0, L2 penalty term will not apply
        if self.alpha == 0:
            self.weight = torch.pinverse(X.T @ X) @ X.T @ y
        else:
            ridge = self.alpha * torch.eye(X.shape[1])
            self.weight = torch.pinverse((X.T @ X) + ridge) @ X.T @ y

    def predict(self, X: torch.Tensor):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge.predict(X_test)
        ~~~
        """
        return X @ self.weight
