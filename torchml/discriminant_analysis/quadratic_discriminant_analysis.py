import torch
import warnings

import torchml as ml


class QuadraticDiscriminantAnalysis(ml.Model):

    """
    <a href="https://github.com/learnables/torchml/blob/master/torchml/discriminant_analysis/quadratic_discriminant_analysis.py">[Source]</a>

    ## Description

    Quadratic Discriminant Analysis is a classifier with a quadratic decision boundary, which is calculated by fitting class conditional densities to the data and using Bayes' rule. This model fits a Gaussian density to each class.
    This current implementation only includes "svd" solver.

    ## References

    1. Carl J Huberty's Discriminant Analysis [paper](https://www.jstor.org/stable/1170065#metadata_info_tab_contents)
    2. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)

    ## Arguments

    * `priors` (torch.Tensor, default=None) - The class prior probabilities. By default, the class proportions are calculated from the input training data.
    * `reg_param` (float, default=0.0) - Regularizes the per-class covariance estimates by transforming S2 as `S2 = ((1 - reg_param) * S2) + reg_param`, where S2 corresponds to the scaling_ attribute of a given class.
    * `store_covariance` (bool, default=False) - If True, the class covariance matrices will be explicitly computed and stored in the self.covariance_ attribute.
    * `tol` (float, default=1e-4) - Absolute threshold for a singular value to be considered significant. This parameter does not affect the predictions. It controls a warning that is raised when features are considered to be colinear.

    ## Example

    ~~~python
    qda = QuadraticDiscriminantAnalysis()
    ~~~
    """

    def __init__(
        self,
        *,
        priors: torch.Tensor = None,
        reg_param: float = 0.0,
        store_covariance: bool = False,
        tol: float = 1e-4
    ):
        super(QuadraticDiscriminantAnalysis, self).__init__()
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    def _classes_means(self, X: torch.Tensor, y: torch.Tensor):

        """
        ## Description
        Compute class means.

        ## Arguments
        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.
        """

        means = torch.zeros(self.classes_.shape[0], X.shape[1])
        for i in range(self.classes_.shape[0]):
            means[i, :] = torch.mean(X[y == i], 0)

        means = torch.FloatTensor(means)
        return means

    def fit(self, X: torch.Tensor, y: torch.Tensor):

        """
        ## Description
        Fit the Quadratic Discriminant Analysis model.

        ## Arguments
        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.

        ## Example

        ~~~python
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        ~~~
        """

        # data validation check
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        self.classes_, y = torch.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]

        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )
        if self.priors is None:
            self.priors_ = torch.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        cov = None
        store_covariance = self.store_covariance
        if store_covariance:
            cov = []
        means = []
        scalings = []
        rotations = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if Xg.shape[0] == 1:
                raise ValueError(
                    "y has only 1 sample in class %s, covariance is ill defined."
                    % str(self.classes_[ind])
                )
            Xgc = Xg - meang
            # Xgc = U * S * V.T
            _, S, Vt = torch.linalg.svd(Xgc, full_matrices=False)
            rank = torch.sum(S > self.tol)
            if rank < n_features:
                warnings.warn("Variables are collinear")
            S2 = (S**2) / ((Xg.shape[0]) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append((S2 * Vt.T) @ Vt)
            scalings.append(S2)
            rotations.append(Vt.T)
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        self.means_ = torch.stack(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X: torch.Tensor):
        norm2 = []
        for i in range(self.classes_.shape[0]):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = Xm @ (R * (S ** (-0.5)))
            norm2.append(torch.sum(X2**2, dim=1))
        norm2 = torch.stack(norm2).T
        u = torch.tensor([torch.sum(torch.log(s)) for s in self.scalings_])
        return -0.5 * (norm2 + u) + torch.log(self.priors_)

    def decision_function(self, X: torch.Tensor):

        """
        ## Description
        Apply decision function to an array of samples.

        ## Arguments
        * `X` (Tensor) - Input data.

        ## Example

        ~~~python
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        qda_dec_func = qda.decision_function(X_test)
        ~~~
        """

        dec_func = self._decision_function(X)
        # handle special case of two classes
        if self.classes_.shape[0] == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    def predict(self, X: torch.Tensor):

        """
        ## Description
        Predict using Quadratic Discriminant Analysis model.

        ## Arguments
        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        qda_pred = qda.predict(X_test)
        ~~~
        """

        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred

    def predict_proba(self, X: torch.Tensor):

        """
        ## Description
        Calculate and return posterior probabilities of classification.

        ## Arguments
        * `X` (Tensor) - Input data.

        ## Example

        ~~~python
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        qda_predict_proba = qda.predict_proba(X_test)
        ~~~
        """

        values = self._decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = torch.exp(values - torch.max(values, dim=1)[0][:, None])
        # compute posterior probabilities
        return likelihood / torch.sum(likelihood, dim=1)[:, None]

    def predict_log_proba(self, X: torch.Tensor):

        """
        ## Description
        Calculate and return log of posterior probabilities of classification.

        ## Arguments
        * `X` (Tensor) - Input data.

        ## Example

        ~~~python
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        qda_predict_log_proba = qda.predict_log_proba(X_test)
        ~~~
        """

        probas_ = self.predict_proba(X)
        return torch.log(probas_)
