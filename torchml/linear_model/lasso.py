import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import torchml as ml


class Lasso(ml.Model):
    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = False,
        positive: bool = False
    ):
        super(Lasso, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.positive = positive

    def fit(self, X: torch.Tensor, y: torch.Tensor):

        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

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
        X, y = X.type(torch.float64), y.type(torch.float64)
        X.requires_grad_(True)
        y.requires_grad_(True)

        # this object is now callable with pytorch tensors
        if self.fit_intercept:
            self.weight, self.intercept = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float32))
        else:
            self.weight = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float32))
        self.weight = torch.stack(list(self.weight), dim=0)

    def predict(self, X: torch.Tensor):
        if self.fit_intercept:
            return X @ self.weight + self.intercept
        return X @ self.weight
