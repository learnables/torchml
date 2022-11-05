import torch

import torchml as ml
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class LinearSVC(ml.Model):
    """
    ## Description

    Support vector classifier with cvxpy

    ## References

    1. Bernhard E. Boser, Isabelle M. Guyon, and Vladimir N. Vapnik. 1992. A training algorithm for optimal margin classifiers. In Proceedings of the fifth annual workshop on Computational learning theory (COLT '92). Association for Computing Machinery, New York, NY, USA, 144–152. https://doi.org/10.1145/130385.130401
    2. MIT 6.034 Artificial Intelligence, Fall 2010, [16. Learning: Support Vector Machines](https://youtu.be/_PwhiWxHK8o)
    3. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) for LinearSVC.

    ## Arguments

    * `penalty` (str {'l1', 'l2'}, default=’l2’):
        Specifies the norm used in the penalization.

    * `loss` (str {‘hinge’, ‘squared_hinge’}, default=’squared_hinge’):
        Specifies the loss function. ‘hinge’ is the standard SVM loss.

    * `dual` (bool, default=True):
        Dummy variable to keep consistency with SKlearn's API, always 'False' for now.

    * `tol` (float, default=1e-4)
        Tolerance for stopping criteria.

    * `C` (float, default=1.0):
        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.

    * `multi_class` (str {‘ovr’, ‘crammer_singer’}, default=’ovr’):
        Dummy variable, always 'ovr' (one class over all the other as a single class)

    * `fit_intercept` (bool, default=True):
        Whether to calculate the intercept for this model.

    * `intercept_scaling` (float, default=1):
        Dummy variable to mimic the sklearn API, always 1 for now

    * `class_weight` (dict or str ‘balanced’, default=None):
        Dummy variable to mimic the sklearn API, always None for now

    * `verbose` (int, default=0):
        Dummy variable to mimic the sklearn API, always 0 for now

    * `random_state` (int, RandomState instance or None, default=None):
        Dummy variable to mimic the sklearn API, always None for now

    * `max_iter` (int, default=1000):
        The maximum number of iterations to be run for the underneath convex solver.


    ## Example

    ~~~python
    import numpy as np
    from torchml.svm import LinearSVC
    from sklearn.datasets import make_classification

    x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
        )
    svc = LinearSVC(max_iter=1000)
    svc.fit(torch.from_numpy(x), torch.from_numpy(y))
    svc.decision_function(torch.from_numpy(x)
    svc.predict(torch.from_numpy(x))
    ~~~
    """

    def __init__(
        self,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual=True,
        tol=1e-4,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        super(LinearSVC, self).__init__()
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.penalty = penalty
        self.loss = loss

    def fit(self, X: torch.Tensor, y: torch.Tensor, sample_weight=None):
        """
        ## Description

        Initialize the class with training sets

        ## Arguments
        * `X` (torch.Tensor): the training set
        * `y` (torch.Tensor, default=None): the class labels for each sample

        """
        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        self.classes_ = torch.unique(y)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        m, n = X.shape
        self.coef_ = torch.empty((0, n))
        self.intercept_ = torch.empty((0))
        if self.classes_.shape[0] == 2:
            self._fit_with_one_class(
                X, y, self.classes_[1], sample_weight=sample_weight
            )
        else:
            for i, x in enumerate(self.classes_):
                self._fit_with_one_class(X, y, x, sample_weight=sample_weight)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """
        ## Description

        Predict confidence scores for samples.

        ## Arguments
        * `X` (torch.Tensor): the data set for which we want to get the confidence scores.

        """
        scores = X @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores 

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        ## Description

        Predict the class labels for the provided data.

        ## Arguments

        * `X` (torch.Tensor): the target point
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).long()
        else:
            indices = scores.argmax(dim=1)
        return self.classes_[indices]

    def _fit_with_one_class(
            self, X: torch.Tensor, y: torch.Tensor, fitting_class: any, sample_weight=None
    ):

        m, n = X.shape

        y = torch.unsqueeze(y, 1)

        y = (y == fitting_class).float()
        y *= 2
        y -= 1

        w = cp.Variable((n, 1))
        if self.fit_intercept:
            b = cp.Variable()
        X_param = cp.Parameter((m, n))
        ones = torch.ones((m, 1))

        loss = cp.multiply((1 / 2.0), cp.norm(w, 2))

        if self.fit_intercept:
            hinge = cp.pos(ones - cp.multiply(y, X_param @ w + b))
        else:
            hinge = cp.pos(ones - cp.multiply(y, X_param @ w))

        if self.loss == "squared_hinge":
            loss += cp.multiply(self.C, cp.sum(cp.square(hinge)))
        elif self.loss == "hinge":
            loss += cp.multiply(self.C, cp.sum(hinge))

        objective = loss

        # set up constraints
        constraints = []

        prob = cp.Problem(cp.Minimize(objective), constraints)
        assert prob.is_dpp()

        # convert into pytorch layer
        if self.fit_intercept:
            fit_lr = CvxpyLayer(prob, [X_param], [w, b])
        else:
            fit_lr = CvxpyLayer(prob, [X_param], [w])

        # prob.solve(solver="ECOS", abstol=self.tol, max_iters=self.max_iter)
        if self.fit_intercept:
            weight, intercept = fit_lr(X, solver_args={"solve_method": "ECOS", "abstol": self.tol,
                                                       "max_iters": self.max_iter})
        else:
            weight = fit_lr(X, solver_args={"solve_method": "ECOS", "abstol": self.tol, "max_iters": self.max_iter})

        self.coef_ = torch.cat((self.coef_, torch.t(weight)))

        if self.fit_intercept:
            self.intercept_ = torch.cat(
                (self.intercept_, torch.unsqueeze(intercept, 0))
            )
        return self
