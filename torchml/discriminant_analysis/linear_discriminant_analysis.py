import torch
import warnings

import torchml as ml

class LinearDiscriminantAnalysis(ml.Model):
    def __init__(
        self,
        *,
        solver: str = "svd",
        n_components: int = None,
        priors: torch.Tensor = None
    ):
        super(LinearDiscriminantAnalysis, self).__init__()
        self.solver = solver
        self.n_components = n_components
        self.priors = priors

    def _classes_means(self, X, y):
        means = []
        for label in self.classes_:
            tmp = torch.mean(X[y == label])
            means.append(tmp)
        return means

    def _classes_cov(self, X, y):
        cov = torch.zeros(X.shape[1], X.shape[1])
        for idx, label in self.classes_:
            Xg = X[y == label, :]
            cov += self.priors_[idx] * torch.atleast_2d(torch.cov(Xg))
        return cov
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):

        # data validation check
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        self.classes_, counts = torch.unique(y, return_counts=True)
        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]

        if n_samples <= n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        # get priors
        if self.priors is None:
            self.priors_ = counts / y.shape[0].to(torch.float64)
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
        self.means_ = self._classes_means(self, X, y)
        self.covariances_ = self._classes_cov(self, X, y)
        
        # solve
        Sw = self.covariances_
        St = torch.cov(X)
        Sb = St - Sw
        
        temp = Sw.pinverse().matmul(Sb)
        evals, evecs = torch.eig(temp, eigenvectors=True)

        evals, inc_idx = torch.sort(evals)
        evecs = evecs[:, inc_idx]
        
        self.scalings = evecs
        self.coef_ = torch.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * torch.diag(torch.dot(self.means_, self.coef_.T)) + torch.log(
            self.priors_
        )     
