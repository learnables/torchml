# torchml

--------------------------------------------------------------------------------

![Test Status](https://github.com/learnables/torchml/workflows/Testing/badge.svg?branch=master)

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

Create a virtual environment:

```
conda create -n torchml python=3.9
conda activate torchml
```

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
