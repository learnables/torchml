import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import torchml as ml


class Lasso(ml.Model):

    """
    <a href="https://github.com/learnables/torchml/blob/master/torchml/linear_model/lasso.py">[Source]</a>

    ## Description

    Linear regression with L1 penalty term.

    $$ \\min_w \\frac{1}{2m} \\vert \\vert  Xw + b - y \\vert \\vert^2 I + \\frac{1}{2} \\lambda \\vert w \\vert $$

    * `m` - number of input samples
    * `X` - variates
    * `w` - weights of the linear regression with L1 penalty
    * `b` - intercept
    * `y` - covariates
    * `λ` - constant that multiplies the L1 term

    Since lasso regression cannot derive into a closed-form equation, we used Cvxpylayers to construct a pytorch layer and directly compute the solution for the objective function above.

    ## References

    1. Robert Tibshirani's introduction to Lasso Regression [paper](https://www.jstor.org/stable/2346178)
    2. Datacamp Lasso and Ridge Regression Tutorial [tutorial](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression#data-splitting-and-scaling)
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

    ## Arguments

    * `alpha` (float, default=1.0) - Constant that multiplies the L1 term. alpha must be a non-negative float.
    * `fit_intercept` (bool, default=False) - Whether or not to fit intercept in the model.
    * `positive` (bool, default=False) - When set to True, forces the weights to be positive.
    * `require_grad` (bool, default=False) - When set to True, tensor's require_grad will set to be true (useful if gradients need to be computed).

    ## Example

    ~~~python
    lasso = Lasso()
    ~~~
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = False,
        positive: bool = False,
        require_grad: bool = False
    ):
        super(Lasso, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.require_grad = require_grad

    def fit(self, X: torch.Tensor, y: torch.Tensor):

        """
        ## Description

        Compute the weights for the model given variates and covariates.

        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.

        ## Example

        ~~~python
        lasso = Lasso()
        lasso.fit(X_train, y_train)
        ~~~
        """

        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        device = X.device

        m, n = X.shape

        w = cp.Variable((n, 1))
        if self.fit_intercept:
            b = cp.Variable()
        X_param = cp.Parameter((m, n))
        y_param = cp.Parameter((m, 1))
        alpha = cp.Parameter(nonneg=True)

        # set up objective
        if self.fit_intercept:
            loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w + b - y_param))
        else:
            loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w - y_param))

        penalty = alpha * cp.norm1(w)
        objective = loss + penalty

        # set up constraints
        constraints = []
        if self.positive:
            constraints = [w >= 0]

        prob = cp.Problem(cp.Minimize(objective), constraints)

        # convert into pytorch layer
        if self.fit_intercept:
            fit_lr = CvxpyLayer(prob, [X_param, y_param, alpha], [w, b])
        else:
            fit_lr = CvxpyLayer(prob, [X_param, y_param, alpha], [w])

        # process input data
        if self.require_grad:
            X.requires_grad_(True)
            y.requires_grad_(True)

        # this object is now callable with pytorch tensors
        if self.fit_intercept:
            self.weight, self.intercept = fit_lr(
                X, y, torch.tensor(self.alpha, dtype=torch.float64, device=device)
            )
        else:
            self.weight = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float64, device=device))
        self.weight = torch.stack(list(self.weight), dim=0)

    def predict(self, X: torch.Tensor):

        """
        ## Description

        Predict covariates by the trained model.

        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        lasso = Lasso()
        lasso.fit(X_train, y_train)
        lasso.predict(X_test)
        ~~~
        """

        if self.fit_intercept:
            return X @ self.weight + self.intercept
        return X @ self.weight
