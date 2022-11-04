# Gaussian Naive Bayes

## Introduction

Gaussian Naive Bayes is a supervised learning algorithm based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. It is typically used for classification.

## Probabilistic Derivation

### Naive Bayes
Bayes’ theorem states the following relationship, given class variable $y$ and dependent feature vector $x_1$ through $x_n$ :

$$P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots, x_n \mid y)}
                                 {P(x_1, \dots, x_n)}$$

Using the naive conditional independence assumption that

$$P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y),$$

for all $i$, this relationship is simplified to

$$P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                 {P(x_1, \dots, x_n)}$$

Since $P(x_1, \dots, x_n)$ is constant given the input, we can use the following classification rule:

$$\begin{align}\begin{aligned}P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)\\\Downarrow\\\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),\end{aligned}\end{align}$$

and we can use the relative frequency of class $y$ in the training set to estimate $P(y)$.

### *Gaussian* Naive Bayes
The likelihood of the features $P(x_i \mid y)$ is assumed to be Gaussian:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$$

The parameters $\sigma_y$ and $\mu_y$ are estimated using maximum likelihood.

In practice, instead of likelihood, we compute log likelihood.


## Implementation
We first compute the mean and variance of each feature for all data points in the same class (`X[i]` for the `i`th class):
```
mu = torch.mean(X[i], dim=0)
var = torch.var(X[i], dim=0, unbiased=False)
```

Then we compute the joint log likelihood for the given data point to belong to each of the classes:
```
n_ij = -0.5 * torch.sum(torch.log(2.0 * math.pi * self.var[i, :]), 0, True)
n_ij -= 0.5 * torch.sum(((X - self.theta[i, :]) ** 2) / (self.var[i, :]), 1, True)
```

Finally, we take an argmax for the joint log likelihood amongst all classes:
```
return self.classes[joint_log_likelihood.argmax(1)]
```

## The `torchml` Interface
First fit the classifier with training data, then make prediction using the classifier.
```
clf = GaussianNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

## References
- [scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html)
- H. Zhang (2004). [The optimality of Naive Bayes](https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf). Proc. FLAIRS.