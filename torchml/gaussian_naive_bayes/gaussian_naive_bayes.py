
import math
import torch

import torchml as ml


class GaussianNB(ml.Model):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    One-liner description (eg, "Ordinary least-square model with bias term").

    You can write a more thorough description here, including references and even equations.
    Solves:

    $$ \\min_w \\vert \\vert  Xw - y \\vert \\vert^2 $$

    ## References

    1. Gauss, for OLS?

    ## Arguments

    *  `priors` : array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    * `var_smoothing` : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

    ## Example

    ~~~python
    ~~~
    """

    def __init__(
        self,
        *,
        priors=None, 
        var_smoothing=1e-9,
    ):
        super().__init__()
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
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
        # from sklearn doc:
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        # self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

        size = X.shape[0]
        self.n_feature = X.shape[1]
        y_unique_val = y.unique()
        self.n_class = len(y_unique_val)
        
        # Initialize the class prior
        if self.priors is not None:
            self.priors = torch.from_numpy(self.priors)
            # Check that the provided prior matches the number of classes
            if len(self.priors) != self.n_class:
                raise ValueError("Number of priors must match number of classes.")
            # Check that the sum is 1
            if not torch.isclose(self.priors.sum(), 1.0):
                raise ValueError("The sum of the priors should be 1.")
            # Check that the priors are non-negative
            if (self.priors < 0).any():
                raise ValueError("Priors must be non-negative.")
            self.class_probs = self.priors
        else:
            # Probability of each class in the training set
            self.class_probs = y.int().bincount().float() / size

        # All the posterior probabilites
        cond_probs = [] 
        for i in range(self.n_class):
            cond_probs.append([])
            # Group samples by class
            idx = torch.where(y == y_unique_val[i])[0]
            elts = X[idx]
            # Compute mean and std of each feature
            for j in range(self.n_feature):
                cond_probs[i].append([])
                features_class = elts[:, j]
                mean = features_class.mean()
                std = (features_class - mean).pow(2).mean().sqrt()
                # Store these value to use them for the gaussian likelihood
                cond_probs[i][j] = [mean, std]
        self.cond_probs = cond_probs

    def gaussian_likelihood(self, X, mean, std):
        """Computes the gaussian likelihood
        
        ## Arguments
            * `X` - A torch tensor for the data.
            * `mean` - A float for the mean of the gaussian.
            * `std` - A flot for the standard deviation of the gaussian.
        """
        return (1 / (2 * math.pi * std.pow(2))) * torch.exp(-0.5 * ((X - mean) / std).pow(2))

    def predict(self, X):
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
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        n_sample = X.shape[0]
        # import ipdb; ipdb.set_trace()
        pred_probs = torch.zeros((n_sample, self.n_class), dtype=torch.float32)
        for k in range(n_sample):
            elt = X[k]
            for i in range(self.n_class):
                # Set probability by the prior (class probability)
                pred_probs[k][i] = self.class_probs[i]
                prob_feature_per_class = self.cond_probs[i]
                for j in range(self.n_feature):
                    # multiply by the gaussian likelihood with parameters
                    # mean and std of the class i on feature j
                    mean, std = prob_feature_per_class[j]
                    pred_probs[k][i] *= self.gaussian_likelihood(elt[j], mean, std)     
        # Get to highest probability among all classes
        return pred_probs.argmax(dim=1)