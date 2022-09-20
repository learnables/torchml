import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import torchml as ml


class Lasso(ml.Model):
    def __init__(
        self,
        *,
        alpha: float = 1.0
    ):
        super(Lasso, self).__init__()
        self.alpha = alpha

    def fit(self, X: torch.Tensor, y: torch.Tensor):

        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        # when alpha == 0, L1 penalty term will not apply
        if self.alpha == 0:
            self.weight = torch.pinverse(X.T @ X) @ X.T @ y
        else:
            m, n = X.shape

            w = cp.Variable((n, 1))
            X_param = cp.Parameter((m, n))
            y_param = cp.Parameter((m, 1))
            alpha = cp.Parameter(nonneg=True)

            # set up objective
            loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w - y_param))
            penalty = alpha * cp.norm1(w)
            objective = loss + penalty

            # set up constraints
            constraints = []

            prob = cp.Problem(cp.Minimize(objective), constraints)

            # convert into pytorch layer in one line
            fit_lr = CvxpyLayer(prob, [X_param, y_param, alpha], [w])

            # process input data
            X, y = X.type(torch.float64), y.type(torch.float64)
            X.requires_grad_(True)
            y.requires_grad_(True)

            # this object is now callable with pytorch tensors
            self.weight = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float64))

            self.weight = torch.stack(list(self.weight), dim=0)

    def predict(self, X: torch.Tensor):
        return X @ self.weight
