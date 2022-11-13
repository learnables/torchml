from random import random
import torch
import math

import torchml as ml


class RBFSampler(ml.Model):

    """
    <a href="https://github.com/learnables/torchml/blob/master/torchml/kernel_approximations/rbfsampler.py">[Source]</a>

    ## Description

    RBFSampler constructs an approximate mapping for Random Kitchen Sinks.

    ## References

    1. Ali Rahimi and Benjamin Rechti's Weighted Sums of Random Kitchen Sinks [paper](https://papers.nips.cc/paper/2008/hash/0efe32849d230d7f53049ddc4a4b0c60-Abstract.html)
    2. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html)

    ## Arguments

    * `gamma` (float, default=1.0) - Parameter of RBF kernel.
    * `n_components` (int, default=100) - Dimensionality of the computed feature space.
    * `random_state` (int, default=None) - Passed in seed that controls the generation of the random weights and random offset when fitting the training data.

    ## Example

    ~~~python
    rbfsampler = RBFSampler()
    ~~~
    """

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        n_components: int = 100,
        random_state: int = None
    ):
        super(RBFSampler, self).__init__()
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.random_weights_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor or None = None):

        """
        ## Description

        Fit the model with training data X.

        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates (None for unsupervised transformation).

        ## Example

        ~~~python
        rbfsampler = RBFSampler()
        rbfsampler.fit(X_train, y_train)
        ~~~
        """

        n_features = X.size(dim=1)

        # get random seed
        seed = self.random_state
        if seed is None:
            torch.random.seed()
        else:
            torch.manual_seed(seed)

        self.random_normal_matrix = torch.empty((n_features, self.n_components)).normal_()
        self.random_uniform_matrix = torch.empty((self.n_components)).uniform_(0, 2 * math.pi)

        self.random_weights_ = math.sqrt(self.gamma * 2) * self.random_normal_matrix
        self.random_offset_ = self.random_uniform_matrix

        self._n_features_out = self.n_components
        return self

    def transform(self, X: torch.Tensor):

        """
        ## Description

        Apply the approximate feature mapping to X.

        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        rbfsampler = RBFSampler()
        rbfsampler.fit(X_train, y_train)
        rbfsampler.transform(X_train)
        ~~~
        """

        assert self.random_weights_ is not None, "This RBFSampler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."

        self.random_weights_ = self.random_weights_.double()
        self.random_offset_ = self.random_offset_.double()

        projection = torch.mm(X, self.random_weights_)
        projection += self.random_offset_
        projection = torch.cos(projection)
        projection *= math.sqrt(2.0) / math.sqrt(self.n_components)
        return projection

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor or None = None):

        """
        ## Description

        Fit and then transform X.

        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates (None for unsupervised transformation).

        ## Example

        ~~~python
        rbfsampler = RBFSampler()
        rbfsampler.fit_transform(X_train, y_train)
        ~~~
        """

        return self.fit(X, y).transform(X)
