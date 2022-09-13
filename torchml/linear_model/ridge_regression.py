import torch

import torchml as ml

class RidgeRegression(ml.Model):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    Ridge Regression: Linear Regression with L2 Penalty Term
    Equation Used: w = inv(XTX + alpha*I) * XTy

    ## References

    1. Arthur E. Hoerl and Robert W. Kennard's introduction to Ridge Regression [paper](https://www.jstor.org/stable/1271436) for ridge regression.
    2. Datacamp Lasso and Ridge Regression Tutorial [tutorial](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression#data-splitting-and-scaling) for linear and ridge regression.
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=lasso#sklearn.linear_model.ridge) for ridge regression.
    4. The scikit-learn source code [source code](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py) for ridge regression.
    5. Ridge Regression in PyTorch [source code](https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12) for ridge regression.

    ## Arguments

    * `alpha` (float, default=1.0) - Constant that multiplies the L2 term. alpha must be a non-negative float.
    * `fit_intercept` (bool, default=True) - Whether or not to fit intercept in the model.
    * `normalize` (bool, default=False) - If True, the regressors X will be normalized. normalize will be deprecated in the future.
    * `copy_X` (bool, default=True) - If True, X will be copied.
    * `solver` (string, default='auto') - Different solvers or algorithms to use

    ## Example

    ~~~python
    ~~~
    """
    
    def __init__(
        self,
        *,
        alpha = 1.0,
        fit_intercept=False,
        normalize=False,
        copy_X=True,
        solver='auto'
    ):
        super(RidgeRegression, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.solver = solver

    def fit (self, X, y):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.

        ## Example

        ~~~python
        ridgereg = RidgeRegression()
        ridgereg.fit(X_train, y_train)
        ~~~
        """
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)

        # when alpha == 0, L2 penalty term will not apply
        if self.alpha == 0:
            self.weight = torch.pinverse(X.T @ X) @ X.T @ y
        else:
            ridge = self.alpha * torch.eye(X.shape[0])
            self.weight = torch.pinverse((X.T @ X) + ridge) @ X.T @ y
            # self.weight = torch.lstsq(X.T @ y, (X.T @ X) + ridge).solution
    
    def predict (self, X):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        ridgereg = RidgeRegression()
        ridgereg.fit(X_train, y_train)
        ridgereg.predict(X_test)
        ~~~
        """
        return X @ self.weight


