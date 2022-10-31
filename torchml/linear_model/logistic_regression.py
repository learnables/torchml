import torch
from tqdm import tqdm

import torchml as ml


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class LogisticRegression(ml.Model):

    """
    ## Description

    Logistic regression is a linear model for classification.
    In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

    ## References

    1. The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

    ## Arguments

    * `max_iterint` (int) - Maximum number of iterations taken for training, default=100000

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
        max_iter: int = 100000,
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
        self.model = LogisticRegressionModel(2, 1)  # binary classification
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        for epoch in tqdm(range(self.max_iter), desc="Training"):
            # Forward pass
            outputs = self.model(X)
            loss = criterion(torch.squeeze(outputs), y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            outputs = self.model(X)
            return torch.squeeze(outputs).round().numpy()
