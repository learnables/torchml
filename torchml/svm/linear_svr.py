import torch

import torchml as ml
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class LinearSVR(ml.Model):
    """
    ## Description

    Support vector regressor with cvxpy

    ## References

    1. Bernhard E. Boser, Isabelle M. Guyon, and Vladimir N. Vapnik. 1992. A training algorithm for optimal margin classifiers. In Proceedings of the fifth annual workshop on Computational learning theory (COLT '92). Association for Computing Machinery, New York, NY, USA, 144–152. https://doi.org/10.1145/130385.130401
    2. MIT 6.034 Artificial Intelligence, Fall 2010, [16. Learning: Support Vector Machines](https://youtu.be/_PwhiWxHK8o)
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) for LinearSVC.

    ## Arguments

    * `loss` (str {‘epsilon_insensitive’, ‘squared_epsilon_insensitive’}, default=’epsilon_insensitive’):
        Specifies the loss function.

    * `epsilon` (float, default=0.0):
        Epsilon parameter in the epsilon-insensitive loss function.

    * `tol` (float, default=1e-4)
        Tolerance for stopping criteria.

    * `C` (float, default=1.0):
        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.

    * `fit_intercept` (bool, default=True):
        Whether to calculate the intercept for this model.

    * `intercept_scaling` (float, default=1):
        Dummy variable to mimic the sklearn API, always 1 for now

    * `dual` (bool, default=True):
        Dummy variable to keep consistency with SKlearn's API, always 'False' for now.

    * `verbose` (int, default=0):
        Dummy variable to mimic the sklearn API, always 0 for now

    * `random_state` (int, RandomState instance or None, default=None):
        Dummy variable to mimic the sklearn API, always None for now

    * `max_iter` (int, default=1000):
        The maximum number of iterations to be run for the underneath convex solver.


    ## Example

    ~~~python
    import numpy as np
    from torchml.svm import LinearSVR
    from sklearn.datasets import make_regression

    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
    )
    svr = LinearSVR(max_iter=1000)
    svr.fit(torch.from_numpy(x), torch.from_numpy(y))
    svr.predict(torch.from_numpy(x))
    ~~~
    """

    def __init__(
        self,
        *,
        epsilon=0.0,
        tol=1e-4,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        super(LinearSVR, self).__init__()
        self.intercept_ = None
        self.coef_ = None
        self.classes_ = None
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.dual = dual
        self.loss = loss

    def fit(self, X: torch.Tensor, y: torch.Tensor, sample_weight=None):
        """
        ## Description

        Initialize the class with training sets

        ## Arguments
        * `X` (torch.Tensor): the training set
        * `y` (torch.Tensor): Target vector relative to X.
        * `sample_weight` (default=None): Dummy variable for feature not supported yet.
        """

        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        m, n = X.shape
        m, n = X.shape

        y = torch.unsqueeze(y, 1)

        w = cp.Variable((n, 1))
        if self.fit_intercept:
            b = cp.Variable()
        X_param = cp.Parameter((m, n))

        loss = cp.multiply((1 / 2.0), cp.norm(w, 2))

        if self.fit_intercept:
            hinge = cp.pos(cp.abs(y - (X_param @ w + b)) - self.epsilon)
        else:
            hinge = cp.pos(cp.abs(y - (X_param @ w + b)) - self.epsilon)

        if self.loss == "epsilon_insensitive":
            loss += self.C * cp.sum(cp.square(hinge))
        elif self.loss == "squared_epsilon_insensitive":
            loss += self.C * cp.sum(hinge)

        objective = loss

        # set up constraints
        constraints = []

        prob = cp.Problem(cp.Minimize(objective), constraints)
        assert prob.is_dpp()
        X_param.value = X.numpy()
        if self.fit_intercept:
            fit_lr = CvxpyLayer(prob, [X_param], [w, b])
        else:
            fit_lr = CvxpyLayer(prob, [X_param], [w])

        if self.fit_intercept:
            self.coef_, self.intercept_ = fit_lr(
                X,
                solver_args={
                    "solve_method": "ECOS",
                    "abstol": self.tol,
                    "max_iters": self.max_iter,
                },
            )
        else:
            (self.coef_,) = fit_lr(
                X,
                solver_args={
                    "solve_method": "ECOS",
                    "abstol": self.tol,
                    "max_iters": self.max_iter,
                },
            )

        self.coef_ = torch.flatten(self.coef_)
        if self.fit_intercept:
            self.intercept_ = torch.flatten(self.intercept_)

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        ## Description

        Predict using the linear model

        ## Arguments

        * `X` (torch.Tensor): Samples.
        """
        return X @ self.coef_ + self.intercept_
