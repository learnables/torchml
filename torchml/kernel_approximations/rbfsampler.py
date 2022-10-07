import torch
import math

import torchml as ml


class RBFSampler(ml.Model):
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

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # TODO: validate parameter
        n_features = X.size(dim=1)

        # get random seed
        seed = self.random_state
        if seed is None:
            torch.random.seed()
        else:
            torch.manual_seed(seed)

        self.random_weights_ = ((2 * self.gamma) ** 2) * torch.empty((n_features, self.n_components)).normal_()
        self.random_offset_ = torch.empty((self.n_components)).uniform_(0, 2 * math.pi)

        if X.dtype == torch.float64:
            self.random_weights_ = self.random_weights_.astype(X.dtype, copy=False)
            self.random_offset_ = self.random_offset_.astype(X.dtype, copy=False)

        self._n_features_out = self.n_components
        return self

    def transform(self, X: torch.Tensor):
        # TODO: check if it is fitted
        X = X.type(torch.float64)
        self.random_weights_ = self.random_weights_.type(torch.float64)
        self.random_offset_ = self.random_offset_.type(torch.float64)

        projection = torch.mm(X, self.random_weights_)
        projection += self.random_offset_
        projection = torch.cos(projection)
        projection *= (2 ** 2) / (self.n_components ** self.n_components)
        return projection

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor):
        return self.fit(X, y).transform(X)
