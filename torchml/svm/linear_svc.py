import torch
from sklearn.datasets import make_classification

import torchml as ml
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from sklearn import svm


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
        self.y_ = None
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
            raise ValueError(
                "Penalty term must be positive; got (C=%r)" % self.C)
        self.classes_ = torch.unique(y)
        self.y_ = y
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        # m, n = X.shape
        #
        # w = cp.Variable(n)
        # if self.fit_intercept:
        #     b = cp.Variable()
        #
        # obj = 0
        # for i in range(m):
        #     if y[i] == self.classes_[1]:
        #         yi = 1
        #     else:
        #         yi = -1
        #     if self.fit_intercept:
        #         obj += cp.square(cp.pos(1 - yi * (w.T @ X[i] + b)))
        #     else:
        #         obj += cp.sqaure(cp.pos(1 - yi * (w.T @ X[i])))
        #
        # obj *= self.C
        # obj += cp.multiply((1 / 2.0), cp.norm(w, 2))
        #
        # prob = cp.Problem(cp.Minimize(obj), [])
        # prob.solve()
        # self.coef_, self.intercept_ = torch.from_numpy(w.value), torch.from_numpy(b.value)
        # if self.fit_intercept:
        #     fit_lr = CvxpyLayer(prob, [], [w, b])
        # else:
        #     fit_lr = CvxpyLayer(prob, [], [w])
        #
        # self.weight, self.intercept = fit_lr()
        y = torch.unsqueeze(y, 1)

        y = (y == self.classes_[1]).float()
        y *= 2
        y -= 1

        m, n = X.shape

        w = cp.Variable((n, 1))
        if self.fit_intercept:
            b = cp.Variable()
        X_param = cp.Parameter((m, n))
        y_param = cp.Parameter((m, 1))
        C_param = cp.Parameter(nonneg=True)
        ones = torch.ones((m, 1))

        # set up objective
        if self.fit_intercept:
            loss = cp.multiply((1 / 2.0),
                               cp.norm(w, 2)) + C_param * cp.sum(cp.square(cp.pos(ones -
                                                                                   cp.multiply(y_param,
                                                                                               X_param @ w + b))))
        else:
            loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w - y_param))

        objective = loss

        # set up constraints
        constraints = []

        prob = cp.Problem(cp.Minimize(objective), constraints)
        X_param.value = X.numpy()
        y_param.value = y.numpy()
        C_param.value = self.C
        prob.solve(solver='ECOS', abstol=self.tol, max_iters=self.max_iter)

        # convert into pytorch layer
        # if self.fit_intercept:
        #     fit_lr = CvxpyLayer(prob, [X_param, y_param, C_param], [w, b])
        # else:
        #     fit_lr = CvxpyLayer(prob, [X_param, y_param, C_param], [w])

        # process input data
        # if self.require_grad:
        #     X.requires_grad_(True)
        #     y.requires_grad_(True)

        # this object is now callable with pytorch tensors

        # if self.fit_intercept:
        #     self.weight, self.intercept = fit_lr(
        #         X, y, self.C
        #     )
        # else:
        #     self.weight = fit_lr(X, y, torch.tensor(
        #         self.alpha, dtype=torch.float64))
        self.coef_, self.intercept_ = torch.from_numpy(w.value), torch.from_numpy(b.value)
        self.coef_ = torch.t(self.coef_)
        return self
