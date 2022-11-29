import cvxpy as cp
import torch
from tqdm import tqdm
from cvxpylayers.torch import CvxpyLayer
import numpy as np

import torchml as ml


class LogisticRegression(ml.Model):

    """
    ## Description

    Logistic regression is a linear model for classification.
    In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

    ## References

    1. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

    ## Arguments

    * `max_iter` (int) - Maximum number of iterations taken for training, default=100000

    ## Example

    ~~~python
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg.predict(X_test)
    ~~~
    """

    def __init__(
        self,
        *,
        max_iter: int = 100,
    ):
        super(LogisticRegression, self).__init__()
        self.max_iter = max_iter

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.03):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.
        * `y` (Tensor) - Target covariates.
        * `lr` (float) - Learning rate, default=0.01.

        ## Example

        ~~~python
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        ~~~
        """
        m, n = X.shape

        # Solve with cvxpylayers
        lambda1_tch = torch.tensor([[0.1]], requires_grad=True)
        lambda2_tch = torch.tensor([[0.1]], requires_grad=True)

        a = cp.Variable((n, 1))
        b = cp.Variable((1, 1))
        lambda1 = cp.Parameter((1, 1), nonneg=True)
        lambda2 = cp.Parameter((1, 1), nonneg=True)
        x = cp.Parameter((m, n))
        Y = y.numpy()[:, np.newaxis]

        log_likelihood = (1.0 / m) * cp.sum(
            cp.multiply(Y, x @ a + b)
            - cp.log_sum_exp(
                cp.hstack([np.zeros((m, 1)), x @ a + b]).T, axis=0, keepdims=True
            ).T
        )
        regularization = -lambda1 * cp.norm(a, 1) - lambda2 * cp.sum_squares(a)
        prob = cp.Problem(cp.Maximize(log_likelihood + regularization))
        fit_logreg = CvxpyLayer(prob, [x, lambda1, lambda2], [a, b])

        loss = torch.nn.BCELoss()
        optimizer = torch.optim.SGD([lambda1_tch, lambda2_tch], lr=lr)

        with tqdm(range(self.max_iter)) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                a_tch, b_tch = fit_logreg(X, lambda1_tch, lambda2_tch)
                outputs = torch.sigmoid(X @ a_tch + b_tch)
                l = loss(outputs.squeeze(), y)
                l.backward()
                optimizer.step()
                pbar.set_description(f"loss: {l.item():.4f}")

        self.slope = a_tch
        self.intercept = b_tch
        return self

    def predict(self, X: torch.Tensor):
        """
        ## Arguments

        * `X` (Tensor) - Input variates.

        ## Example

        ~~~python
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        logreg.predict(X_test)
        ~~~
        """
        with torch.no_grad():
            outputs = torch.sigmoid(X @ self.slope + self.intercept)
            outputs = torch.where(
                outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs)
            )
        return outputs.squeeze()
