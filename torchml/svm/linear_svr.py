import torch

import torchml as ml
import cvxpy as cp


class LinearSVR(ml.Model):
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
        y_param = cp.Parameter((m, 1))
        C_param = cp.Parameter(nonneg=True)
        epi_param = cp.Parameter()

        loss = cp.multiply((1 / 2.0), cp.norm(w, 2))

        if self.fit_intercept:
            hinge = cp.pos(cp.abs(y_param - (X_param @ w + b)) - epi_param)
        else:
            hinge = cp.pos(cp.abs(y_param - (X_param @ w + b)) - epi_param)

        if self.loss == "epsilon_insensitive":
            loss += C_param * cp.sum(cp.square(hinge))
        elif self.loss == "squared_epsilon_insensitive":
            loss += C_param * cp.sum(hinge)

        objective = loss

        # set up constraints
        constraints = []

        prob = cp.Problem(cp.Minimize(objective), constraints)
        X_param.value = X.numpy()
        y_param.value = y.numpy()
        C_param.value = self.C
        epi_param.value = self.epsilon
        prob.solve(solver="ECOS", abstol=self.tol, max_iters=self.max_iter)

        self.coef_, self.intercept_ = torch.flatten(
            torch.from_numpy(w.value)
        ), torch.flatten(torch.from_numpy(b.value))
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.coef_ + self.intercept_
