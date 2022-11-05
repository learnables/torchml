import torch

import torchml as ml
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class LinearSVC(ml.Model):
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
        return X @ self.coef_.T + self.intercept_

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).int()
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
            loss += cp.multiply(self.C,  cp.sum(cp.square(hinge)))
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
            weight, intercept = fit_lr(X, solver_args={"solve_method": "ECOS", "abstol": self.tol, "max_iters": self.max_iter})
        else:
            weight = fit_lr(X, solver_args={"solve_method": "ECOS", "abstol": self.tol, "max_iters": self.max_iter})

        self.coef_ = torch.cat((self.coef_, torch.t(weight)))


        if self.fit_intercept:
            self.intercept_ = torch.cat(
                (self.intercept_, torch.unsqueeze(intercept, 0))
            )
        return self
