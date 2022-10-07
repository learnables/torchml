# torchml


<div>
  <p align="center">
    <img src="docs/assets/images/logo.png" height="200"> 
  </p>
</div>
 torchml is a one-for-all library for few-shot learning on top of PyTorch. The library includes standard algorithms from [scikit-learn](https://scikit-learn.org/stable/index.html), and also advanced differentiable solvers/models of current popular few-shot learning models.

--------------------------------------------------------------------------------

![Test Status](https://github.com/learnables/torchml/workflows/Testing/badge.svg?branch=master)

## Table of contents
- [TODOs](#todos)
- [Getting started](#getting-started)
- [Tests](#tests)
- [Docs](#docs)
- [Torchml example usage](#torchml-example-usage)
- [Implementation](#implementation)
- [Installation](#installation)
- [License](#license)
- [Credits](#credits)



## TODOs

* Replace `myproject` with `torchml` everywhere such that:
    * All tests pass.
    * The docs render nicely.
* Implement algorithms from scikit-learn with PyTorch.
    * including tests for feature parity (ie, same API, same predictions) and gradient correctness (ie, finite differences).
    * including docs.
    * preliminary example: [Linear Regression](torchml/linear_model/linear_regression.py) with some [tests](tests/unit/linear_model/linear_regression_tests.py).
* A logo?

## Getting started

Install the library locally in development mode:

```
make dev
```

## Tests

Add your own unit tests in: `tests/unit/module/submodule_test.py` and run:

```
make tests
```

## Docs

Add your own to: `mkdocs.yaml` and in `docs/api/module.md`, run:

```
make docs
```

and open [http://localhost:8000](http://localhost:8000).


## Torchml example usage

## Implementation

## Installation

## License

## Credits

