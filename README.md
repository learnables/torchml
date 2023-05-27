<p align="center"><img src="/assets/images/torchml-logo.png" height="120px" /></p>

--------------------------------------------------------------------------------

![Test Status](https://github.com/learnables/torchml/workflows/Testing/badge.svg?branch=master)

`torchml` implements the scikit-learn API on top of PyTorch.
This we automatically get GPU support for scikit-learn and, when possible, differentiability.

## Resources

- GitHub: [github.com/learnables/torchml](http://github.com/learnables/torchml)
- Documentation: [learnables.net/torchml](http://learnables.net/torchml/)
- Tutorials: [learnables.net/torchml/tutorials](http://learnables.net/torchml/tutorials/linear_model/)
- Examples: [learnables.net/torchml/examples](https://github.com/learnables/torchml/tree/master/examples)

## Getting Started

`pip install torchml`

### Minimal Linear Regression Example

~~~python
import torchml as ml

(X_train, y_train), (X_test, y_test) = generate_data()

# API closely follows scikit-learn
linreg = ml.linear_mode.LinearRegression()
linreg.fit(X_train, y_train)
linreg.predict(X_test)
~~~

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](./CHANGELOG.md) file.

## Citing

To cite `torchml` repository in your academic publications, please use the following reference.

>  Sébastien M. R. Arnold, Lucy Xiaoyang Shi, Xinran Gao, Zhiheng Zhang, and Bairen Chen. 2023. "torchml: a scikit-learn implementation on top of PyTorch".

You can also use the following Bibtex entry:

~~~bib
@misc{torchml,
  author={Arnold, S{\'e}bastien M R and Shi, Lucy Xiaoyang and Gao, Xinran and Zhang, Zhiheng and Chen, Bairen},
  title={torchml: A scikit-learn implementation on top of PyTorch},
  year={2023},
  url={https://github.com/learnables/torchml},
}
~~~
