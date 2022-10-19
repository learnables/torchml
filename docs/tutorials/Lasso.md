# Lasso

## Overview

The "Lasso" is a coefficien reducing method used on OLS (ordinary least squares). First propsed by Robert Tibshirani from University of Toronto, this method has been widely used in machine learning today. As it minimizes the residual sum of squares subject to the sum of the absolute value of the coeffients being less than a constant, Lasso is able to reduce some coefficients to exactly 0. For further reading and understanding the algorithm, the original paper is linked here: https://www.jstor.org/stable/2346178

The objective function for Lasso is listed below:

$$ \\min_w \\frac{1}{2m} \\vert \\vert  Xw + b - y \\vert \\vert^2 I + \\frac{1}{2} \\lambda \\vert w \\vert $$

* `m` - number of input samples

* `X` - variates

* `w` - weights of the linear regression with L1 penalty

* `b` - intercept

* `y` - covariates

* `λ` - constant that multiplies the L1 penalty term

## Implementation

As Lasso objective function is not differentiable, it is hard to obtain a closed form equation that could be used directly to calculate the coefficients. In torchml, we utilizes Cvxpylayers (https://github.com/cvxgrp/cvxpylayers) to construct a differentiable convex optimization layer to compute the problem. 

One problem of using Cvxpylayers is that the coefficients reduced is not able to reduce to 0 exactly, but Cvxpylayers is capable of minimizing the impact of certain features (extremely small coefficients).

##### Set up objective function for further construction of cvxpylayer

```
if self.fit_intercept:
	loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w + b - y_param))
else:
	loss = (1 / (2 * m)) * cp.sum(cp.square(X_param @ w - y_param))

penalty = alpha * cp.norm1(w)
objective = loss + penalty
```

While Lasso is penalized with L1 term, we utilized Cvxpylayers' norm1 function to retrieve the term and add to the objective function. For further understading on L1 and L2 penalty terms: https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning. 

##### Set up constraints, create the problem, construct a pytorch layer

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

One way to find out the weights and intercept (if fit_intercept is True) is by constructing a pytorch layer. To create such a layer, we have to set up any constraints and the objective function that we want to minimize. For further understanding how to utilize Cvxpylayers, the example of using it is linked: 

##### Obtain weight and intercept

```
if self.fit_intercept:
	self.weight, self.intercept = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float64))
else:
	self.weight = fit_lr(X, y, torch.tensor(self.alpha, dtype=torch.float64))
	self.weight = torch.stack(list(self.weight), dim=0)
```

By fitting in X and y training data into the constructed layer, we are able to retreieve the weight and the intercept (if applicable).

## Conclusion

This tutorial explained the "Lasso" and the code of constructing a optimizing layer and retrieving weight and intercept out of it. Hope this tutorial help you have a better understanding towards the algorithm.