from __future__ import annotations
import math
import torch

import torchml as ml


class GaussianNB(ml.Model):

    """
    <a href="" class="https://scikit-learn.org/stable/modules/naive_bayes.html">[Source]</a>

    ## Description

    Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes' theorem
    with the “naive” assumption of conditional independence between every pair of features given the
    value of the class variable.

    GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the
    features is assumed to be Gaussian.

    ## References

    1. H. Zhang (2004). The optimality of Naive Bayes. Proc. FLAIRS.

    ## Arguments

    *  `priors` : array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    * `var_smoothing` : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

    ## Example

    ~~~python
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    ~~~
    """

    def __init__(
        self,
        *,
        priors: torch.Tensor | None = None,
        var_smoothing: float = 1e-9,
    ):
        super().__init__()
        self.priors = priors
        self.var_smoothing = var_smoothing

    def _update_mean_variance(
        self, n_past: int, mu: float, var: float, X: torch.Tensor
    ):
        if X.shape[0] == 0:
            return mu, var

        n_new = X.shape[0]
        new_mu = torch.mean(X, dim=0)
        new_var = torch.var(X, dim=0, unbiased=False)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.

        ## Example

        ~~~python
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        ~~~
        """
        self.n_feature = X.shape[1]
        self.classes = y.unique()
        self.n_class = len(self.classes)

        self.theta = torch.zeros((self.n_class, self.n_feature), dtype=torch.float64)
        self.var = torch.zeros((self.n_class, self.n_feature), dtype=torch.float64)
        self.class_count = torch.zeros(self.n_class, dtype=torch.float64)

        for i, y_val in enumerate(self.classes):
            X_i = X[y == y_val]
            N_i = X_i.shape[0]
            new_theta, new_sigma = self._update_mean_variance(
                self.class_count[i], self.theta[i, :], self.var[i, :], X_i
            )
            self.theta[i, :] = new_theta
            self.var[i, :] = new_sigma
            self.class_count[i] += N_i

        # from sklearn doc:
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon = self.var_smoothing * torch.var(X, dim=0, unbiased=False).max()
        self.var[:, :] += self.epsilon

        # Empirical prior, with sample_weight taken into account
        self.class_prior = (
            self.class_count / self.class_count.sum()
            if self.priors is None
            else self.priors
        )

    def predict(self, X: torch.Tensor):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        ~~~
        """
        joint_log_likelihood = []
        for i in range(self.classes.shape[0]):
            jointi = torch.log(self.class_prior[i])
            # XXX: not sure why [1] and [N, 1] are not broadcastable
            n_ij = -0.5 * torch.sum(
                torch.log(2.0 * math.pi * self.var[i, :]), 0, True
            ).unsqueeze(0).repeat(X.shape[0], 1)
            n_ij -= 0.5 * torch.sum(
                ((X - self.theta[i, :]) ** 2) / (self.var[i, :]), 1, True
            )
            joint_log_likelihood.append((jointi + n_ij).squeeze(1))

        joint_log_likelihood = torch.stack(joint_log_likelihood).T
        return self.classes[joint_log_likelihood.argmax(1)]
