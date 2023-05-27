# Nearest Centroids

Written by [Bairen Chen](https://bairenc.github.io) on 10/18/2022.

In this article, we will dive into a classification algorithm called Nearest Centroid and explain how to implement it with torchml. If you want to learn more about the NC Classifer's applications in research, [ScienceDirect](https://www.sciencedirect.com/topics/computer-science/nearest-centroid) has a summary page.

!!! note
    This tutorial is written for python users who are familiar with pytorch.

## Overview

* Introduction of NC Classifer and some intuition of how it works
* The implementation code step-be-step, making it easy for users to understand and use NC classifer in torchml

## Nearest Centroids Classifier & Intuition

If you are familiar with the K-Nearest Neighbor Classifer, Nearest Centroids will seem quite familiar. 

In short, For each class in the data set that we train, we keep a representation of the class. Hence, if we have 'N' classes, we would have 'N' representations of the classes, and we call representations 'centroids'. 

$$ \\ \vec\mu_\ell = \frac {1}{|C_\ell|}\underset{i\in C_\ell}{\sum} \vec {x}_i$$

- 'μ' : per-class centroids
- '$\vec{x}$' : training vector
- '$C_\ell$' :  the set of indices of samples belonging to class $\ell\in\mathbf{Y}$



Then when asked to predict with new data points, we compare each of the point with the representations of each class. We simiply find the 'Nearest' centroid and return the class that centroid represents as a the prediction.

$$ \\ \hat{y} = \argmin _{\ell\in \mathbf{Y}}||\vec{\mu_\ell} - \vec{x}|| $$

- '$\mu_\ell$' : per-class centroids 
- '$\vec{x}$' : observation
- '$\mathbf{Y}$' :  all classes in traning set
- '$\hat{y}$' : prediction classes

*Note:*
Often, we use mean/median of the training vectors to calculate centroids as they describe the class well, and Euclidean Distance to measure the distance between the centroids and the unseen data points for prediction.


## NC Classifer Implementation

This section breaks down step-by-step the NC Classifier implementation with example code.

**Import torchml**

~~~python
import torch
import torchml as ml
~~~
Note: importing numpy for numpy arrays demonstration.

**Creating dataset**

~~~python
X = torch.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = torch.tensor([1, 1, 1, 2, 2, 2])
~~~
Here we have:

'X' is a 6x2 training vector: we have 6 samples in our data set and each has 2 features.

'y' is a 1x6 vector of target values: we have 6 target values for the 6 samples in our training vector X.


**Creating model & Fit**

~~~python
centroid = ml.neighbors.NearestCentroid()
centroid.fit(X,y)
~~~

Next, we instantiate the Nearest Centroid Classifer instance in torchml, call it 'centroid'.

Then, we call the fit function in the Nearest Centroid class to fit the Nearest Centroid model according to the given data 'X' and 'y'.

**Predict**

~~~python
test = torch.tensor([[-0.8, -1]])
print(centroid.predict(test))
~~~

*Output*
~~~python
[1]
~~~

Finally, we want to use our NC Classifer model to predict the class of a test vector called 'test'. Calling the predict function in the Nearest Centroid class will perform the classification and output the predicted class.

## Conclusion

Having explained the background of Nearest Centroid and its code implementation with torchml, I hope this tutorial will be helpful to those who are interested in using Nearest Centroid for their classification tasks.

## References

1. scikit-learn Nearest Centroid Documentation.
   https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
2. Wikipedia Page on Nearest Centroid. https://en.wikipedia.org/wiki/Nearest_centroid_classifier
