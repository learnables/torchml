# Nearest Neighbors

## Introduction
`torchml.neighbors` currently supports Unsupervised learnings on classification problem. It currently supports K Nearest Neighbors classification with `torchml.neighbors.NearestNeighbors` that implement `sklearn.neighbors.NearestNeighbors`'s brute force solution with TorchML.

## Probabilistic Derivation

### K Nearest Neighbors classification

The principle behind Nearest Neighbors algorithms is, given a distance function and a new test point $x$, the algorithm find k closest samples in the known sample set, and use them to estimate the $x$. The number $k$ can be user-defined and tuned according to the particular problem. The distance function can be any arbitrary metric function, and standard Euclidean distance is the most common choice.

One important thing about this algorithm is that its not based on any probabilistic framework, but the algorithm is able to estimate probability for each class given a test point $x$ and its k neighbors.

Given a dataset with $n$ samples and $b$ distinct classes, and a new point $x$ we wish to classify: 
$\{x_i, y_i\}, i=1,2....n, y_i \in \{c_1, c_2, c_3... c_b\}$

We calculate the number of samples that fall into a class for all classes:
$\{n_a, a=1,2,3...b\}, \Sigma_{a=1}^{b}n_a = n$

We first find the $k$ nearest neighbors of $x$:
$\{x_j, y_j\}, i=1,2....k, y_j \in \{c_1, c_2, c_3... c_c\}$

We then count the number of points in the $k$ neighbors that are in the class $c$:
$\{nk_a, a=1,2,3...b\}, \Sigma_{a=1}^{b}nk_a = k$

The probability that $x$ is of class $c_c$ is simply:
$P(c_c | x)= {nk_c\over k}$

This estimation is often accurate in practice, even though the algorithm is not built with probability in mind. 

### KNN from a bayesian stand point

Even though the KNN algorithm is not built on top of probabilistic framework, we can gain intuition behind its shockingly good estimation by framing it in the bayesian framework.

What we want is:
$P(c_c | x), c=1,2,3...b$
and in bayesian terms, what we need is:
$P(c_c | x) = {{P(x | c_c)*P(c_c)} \over {P(x)}}$
Given nothing but our samples, $P(c_c)$, or the prior, is simply $n_c \over n$

$P(x)$ is the probabilistic density of random variable $x$, and we need to borrow some knowledge from density estimation for this analysis:

Since we don't know $P(x)$, we need to conduct discrete trials on $P(x)$. Suppose that the density $P(x)$ lies in a D-Dimensional space, and we assume it to be Euclidean. We conduct trials in this space by drawing $n$ points on it according to $P(x)$ (these $n$ points are our samples). By principle of locality, for a given point $x_t$ we've drawn on the space, we can assume that the density have some correlations with points in the small space surrounding it. Let's draw a small sphere around the point, and name the space in the sphere $R$.

The total probability that a test point can end up inside $R$ is the sum of probability that a point can be in a point in $R$ over all the small points in $R$, or the probability mass of $P(x)$ in $R$:
$P_{in R} = {\int_{R} P(x)dx}$

For the $n$ samples we gathered, each sample has a probability $P_{in R}$ of being inside $R$, then the total number of $k$ points that successfully end up in $R$ can be modeled using binomial distribution:
$Bin(k|n,P_{in R}) = {n! \over {k!(n-k)!}}{P_{in R}^k}{(1-P_{in R})}^{n-k}$

We also have:
$E(k) = n*P_{in R}$
$P_{in R} = {{E(k)} \over n}$

For our algorithm we supply the parameter $k$, so we can just sub in our well-chosen $k$ instead of the expectation, which gives us:
$k \approx n*P_{in R}$
$P_{in R} \approx {k \over n}$

We further assume that $R$ is quite small, thus $P(x)$ changes very little inside $R$, and we assume $P(x)$ to follow a uniform distribution, then we can derive that:
$P_{in R} \approx P(x)V$ Where $V$ is the volume of $R$.

Then our final estimation of $P(x)$ will be:
$P(x) = {{k}\over{nV}}$

We repeat the process for a specific class $c_c$, and we will get:
$P(x|c_c) = {{nk_c}\over{n_c V}}$

substitute both $P(x|c_c) = {{nk_c}\over{n_c V}}$ and $P(x) = {{k}\over{nV}}$ into our bayesian, we will get:
$P(c_c | x)= {nk_c\over k}$

## Algorithmic Implementation

Given a new sample, the brute-force algorithm is to:
1. Calculate all pairwise distances between the sample point and the labeled examples
2. Find the `k` neighboring samples with the least `k` smallest distances
3. For each class, obtain the ratio of number of points in that class in the `k` neighbors and the number `k`, and that ratio will be the probability that the new sample belongs to this class.

## The torchml Interface

~~~python
import numpy as np
import torchml as ml
samples = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])
point = np.array([1.1])
neigh = ml.neighbors.KNeighborsClassifier(n_neighbors=3)
neigh.fit(torch.from_numpy(samples), torch.from_numpy(y))
neigh.predict(torch.from_numpy(point)) # returns the most likely class label
neigh.predict_proba(torch.from_numpy(point)) # returns all the class probabilities 
~~~

## References

* [Christopher M. Bishop. 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
* [MIT Lecture on KNN](https://youtu.be/09mb78oiPkA)

 










