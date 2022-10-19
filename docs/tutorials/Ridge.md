# Ridge

## Overview

The "Ridge" is a method proposed by Arthur E. HOERL and Robert W. KENNARD. Since OLS (ordinary least squares) is unsatisfactory most of the times, especially when putting under the context of other fields like physics or chemistry. The method of adding a small positive quantity to the objective function is able to obtain an equation with smaller mean square error. For further reading and understanding the algorithm, the original paper is linked here: https://www.jstor.org/stable/1271436.

Different from Lasso, Ridge applies L2 penalty term in its objective function:

$$ \\min_w \\frac{1}{2} \\vert \\vert  Xw - y \\vert \\vert^2 I + \\frac{1}{2} \\lambda \\vert \\vert w \\vert \\vert^2 $$

​    \* `w` - weights of the linear regression with L2 penalty

​    \* `X` - variates

​    \* `λ`- constant that multiplies the L2 term

​    \* `y` - covariates

While this equation is differentiable, we are able to deduct the closed form equation. By taking the gradient of the objective function and set it to 0, we are able represent weight with the following equation.

$$ w = (X^TX + \\lambda I)^{-1}X^Ty $$

## Implementation

Since Ridge has closed form equation, the implementation is simply matrix multiplication and other basic math arithmetic.

##### fit_intercept

```
if self.fit_intercept:
	X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)

if self.alpha == 0:
	self.weight = torch.pinverse(X.T @ X) @ X.T @ y
else:
	ridge = self.alpha * torch.eye(X.shape[1])
	if self.fit_intercept:
		ridge[0][0] = 0
		self.weight = torch.pinverse((X.T @ X) + ridge) @ X.T @ y
```

The fit_intercept function for ridge is implemented by appednding a list of 1 to input training data X. In this case, we are adding a constant column that can be used to calculate the intercepts. Similar to the implementation of Sklearn, the intercept calculated is not being penalized (not adding the `ridge` term to the intercepts).

## Conclusion

This tutorial explained the "Ridge" and the code of calculating the weight and intercept of the equation. Hope this tutorial help you have a better understanding towards the algorithm and the logic of math behind it.