import torch
import warnings

import torchml as ml


class LinearDiscriminantAnalysis(ml.Model):
    def __init__(
        self,
        *,
        n_components: int = None,
        priors: torch.Tensor = None,
        tol: float = 1e-4,
        solver: str = "svd"
    ):
        super(LinearDiscriminantAnalysis, self).__init__()
        self.n_components = n_components
        self.priors = priors
        self.tol = tol
        self.solver = solver

        if solver != "svd":
            raise NotImplementedError("other methods have not been implemented.")

    def _classes_means(self, X, y):
        _, y = torch.unique(y, return_inverse=True)
        means = torch.zeros(self.classes_.shape[0], X.shape[1])
        for i in range(self.classes_.shape[0]):
            means[i, :] = torch.mean(X[y == i], 0)

        means = torch.FloatTensor(means)
        return means

    def _classes_cov(self, X, y):
        cov = torch.zeros(X.shape[1], X.shape[1])
        for idx, label in enumerate(self.classes_):
            Xg = X[y == label, :]
            if Xg.shape[0] == 1:
                class_cov = torch.zeros(X.shape[1], X.shape[1])
            else:
                class_cov = self.priors_[idx] * torch.atleast_2d(torch.cov(Xg.T, correction=0))
            cov += class_cov
        cov = cov.to(torch.float64)
        return cov

    def _solve_svd(self, X, y):
        svd = torch.linalg.svd

        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group]
            Xc.append(Xg - self.means_[idx, :])

        self.xbar_ = self.priors_ @ self.means_

        Xc = torch.cat(Xc, dim=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = torch.std(Xc, dim=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.0
        fac = torch.asarray(1.0 / (n_samples - n_classes))

        # 2) Within variance scaling
        X = torch.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        U, S, Vt = svd(X, full_matrices=False)

        rank = torch.sum((S > self.tol).int())
        # Scaling of within covariance is: V' 1/S
        scalings = ((Vt[:rank, :] / std).T / S[:rank])
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        # 3) Between variance scaling
        # Scale weighted centers
        X = (
            (torch.sqrt((n_samples * self.priors_) * fac)) * (self.means_ - self.xbar_).T
        ).T @ scalings.float()
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, Vt = svd(X, full_matrices=False)

        if self._max_components == 0:
            self.explained_variance_ratio_ = torch.empty((0,), dtype=S.dtype)
        else:
            self.explained_variance_ratio_ = (S**2 / torch.sum(S**2))[
                : self._max_components
            ]

        rank = torch.sum((S > self.tol * S[0]).int())
        self.scalings_ = scalings.float() @ Vt.T[:, :rank]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = -0.5 * torch.sum(coef**2, dim=1) + torch.log(self.priors_)
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # data validation check
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        self.classes_, counts = torch.unique(y, return_counts=True)
        n_samples = X.shape[0]
        n_classes = self.classes_.shape[0]

        if n_samples <= n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        # get priors
        if self.priors is None:
            self.priors_ = counts / n_samples
        else:
            self.priors_ = self.priors

        if torch.any(self.priors_ < 0):
            raise ValueError("priors must be non-negative")

        if torch.abs(torch.sum(self.priors_) - 1.0) > 1e-5:
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / torch.sum(self.priors_)

        # get number of components for dimensionality reduction
        max_components = min(n_classes - 1, X.shape[1])
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        # get input data means and covariances
        self.means_ = self._classes_means(X, y)
        self.covariance_ = self._classes_cov(X, y)

        self._solve_svd(X, y)

        # treat binary case as a special case
        if self.classes_.shape[0] == 2:
            coef_ = torch.asarray(self.coef_[1, :] - self.coef_[0, :], dtype=X.dtype)
            self.coef_ = torch.reshape(coef_, (1, -1))
            intercept_ = torch.asarray(
                self.intercept_[1] - self.intercept_[0], dtype=X.dtype
            )
            self.intercept_ = torch.reshape(intercept_, (1, 1))
        self._n_features_out = self._max_components

    def _decision_function(self, X: torch.Tensor):
        scores = torch.mm(X.double(), self.coef_.T.double()) + self.intercept_
        dec_func = scores.view(-1) if scores.shape[1] == 1 else scores

        return dec_func

    def predict(self, X: torch.Tensor):
        scores = self._decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).int()
        else:
            indices = torch.argmax(scores, dim=1)

        return torch.take(self.classes_, indices.long())

    def transform(self, X: torch.Tensor):
        if self.solver == "svd":
            X_new = (X - self.xbar_).float() @ self.scalings_

        return X_new[:, : self._max_components].float()
