# Ridge

## Introduction

The "Ridge" is a supervised learning method on OLS adding a L2 penalty. This method of adding a small positive quantity (square of coefficients) to the OLS objective function is able to obtain an equation with smaller mean square error.

## Probablistic Derivation

We first set up the maximum likelihood estimation equation:
$$\max_w p(w \vert D)$$
where
$$D = \{(x, y)\}^N$$

As we select our priors based on normal distribution (L2 penalty), the likelihood function could be reduced through the following equation:
$$\min_w \frac{1}{2} \vert \vert  Xw - y \vert \vert^2 I + \frac{1}{2} \lambda \vert \vert w \vert \vert^2$$

After simplification, the objective function becomes:
$$\min_w (X^TXw + w^TX^Ty)^T - X^Ty + \lambda w$$

While this equation is differentiable, we are able to deduct the closed form equation by taking the gradient of the objective function and set it to 0:

$$w = (X^TX + \lambda I)^{-1}X^Ty$$

## Implementation

If user ask for the intercept term, we prepend the a column of 1 to input data X.

```
if self.fit_intercept:
    X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
```

The weight of the model is calculated through the closed-form equation derived from the obejctive function. Similar to the implementation of Sklearn, intercept of the model is not penalized (not adding the `ridge` term).

```
# L2 penalty term will not apply when alpha is 0
if self.alpha == 0:
    self.weight = torch.pinverse(X.T @ X) @ X.T @ y
else:
    ridge = self.alpha * torch.eye(X.shape[1])
    # intercept term is not penalized when fit_intercept is true
    if self.fit_intercept:
        ridge[0][0] = 0
    self.weight = torch.pinverse((X.T @ X) + ridge) @ X.T @ y
```

## The `torchml` Interface
First fit the model with training data, then make predictions.

```
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge.predict(X_test)
```

## References
- Arthur E. Hoerl and Robert W. Kennard's introduction to Ridge Regression [paper](https://www.jstor.org/stable/1271436)
- The scikit-learn [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=lasso#sklearn.linear_model.ridge)

# Lasso

## Introduction

The "Lasso" is a supervised learning method used on OLS (ordinary least squares) that performs both variable selection and regularization. Mathematically, it consists of a linear model with added L1 penalty term.

The objective function for Lasso is listed below:

$$\min_w \frac{1}{2m} \vert \vert  Xw + b - y \vert \vert^2 I + \frac{1}{2} \lambda \vert w \vert$$

## Implementation

As Lasso objective function is not differentiable, it is hard to obtain a closed form equation that could be used directly to calculate the coefficients. In torchml, we utilizes [Cvxpylayers](https://github.com/cvxgrp/cvxpylayers) to construct a differentiable convex optimization layer to compute the problem. 

One problem of using Cvxpylayers is that the coefficients are not able to reduce to 0 exactly, but Cvxpylayers is capable of minimizing the impact of certain features by assigning them extremely small coefficients).

We first set up objective function of Lasso for further construction of cvxpylayer.

```
if self.fit_intercept:
	loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w + b - y_param))
else:
	loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w - y_param))

penalty = alpha * cp.norm1(w)
objective = loss + penalty
```

Then we set up constraints, create the problem to minimize the objective function, and construct a pytorch layer.

```
constraints = []
if self.positive:
	constraints = [w >= 0]

prob = cp.Problem(cp.Minimize(objective), constraints)

if self.fit_intercept:
	fit_lr = CvxpyLayer(prob, [X_param, y_param, alpha], [w, b])
else:
	fit_lr = CvxpyLayer(prob, [X_param, y_param, alpha], [w])
```

At last, we obtain weight and intercept (if asked by user) by fitting in the input training data into the constructed pytorch layer.

```
if self.fit_intercept:
	self.weight, self.intercept = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float64))
else:
	self.weight = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float64))
	self.weight = torch.stack(list(self.weight), dim=0)
```

## The `torchml` Interface
First fit the model with training data, then make predictions.

```
lasso = Lasso()
lasso.fit(X_train, y_train)
lasso.predict(X_test)
```

## References
- The scikit-learn [tutorial page](https://scikit-learn.org/stable/modules/linear_model.html#lasso)
