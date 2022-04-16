# -*- coding=utf-8 -*-

"""
Some useful utilities for mypackage.
"""


def echo(msg):
    """
    <a class="source-link" href="">[Source]</a>

    First, a short description.

    Throw in an equation, for good measure:

    $$ \\int_\\Omega f(x) p(x) dx $$

    ## Arguments

    * `msg` (string): The message to be printed.

    ## Example

    ~~~python
    echo('Hello world!')
    ~~~
    """
    print(msg)


class Example(list):
    """
    <a class="source-link" href="">[Source]</a>

    General wrapper for gradient-based meta-learning implementations.

    A variety of algorithms can simply be implemented by changing the kind
    of `transform` used during fast-adaptation.
    For example, if the transform is `Scale` we recover Meta-SGD [2] with `adapt_transform=False`
    and Alpha MAML [4] with `adapt_transform=True`.
    If the transform is a Kronecker-factored module (e.g. neural network, or linear), we recover
    KFO from [5].

    ## References

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.”
    2. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.”
    3. Park & Oliva. 2019. “Meta-Curvature.”
    4. Behl et al. 2019. “Alpha MAML: Adaptive Model-Agnostic Meta-Learning.”
    5. Arnold et al. 2019. “When MAML Can Adapt Fast and How to Assist When It Cannot.”

    ## Example

    ~~~python
    model = SmallCNN()
    transform = l2l.optim.ModuleTransform(torch.nn.Linear)
    gbml = l2l.algorithms.GBML(
        module=model,
        transform=transform,
        lr=0.01,
        adapt_transform=True,
    )
    ~~~
    """

    def __init__(
            self,
            module,
            transform,
            lr=1.0,
            adapt_transform=False,
            first_order=False,
            allow_unused=False,
            allow_nograd=False,
            **kwargs,
            ):
        """
        ## Arguments

        * `module` (Module) - Module to be wrapped.
        * `tranform` (Module) - Transform used to update the module.
        * `lr` (float) - Fast adaptation learning rate.
        """
        pass

    def adapt(
            self,
            loss,
            first_order=None,
            allow_nograd=None,
            allow_unused=None,
            ):
        """
        Takes a gradient step on the loss and updates the cloned parameters in place.

        The parameters of the transform are only adapted if `self.adapt_update` is `True`.

        ## Arguments

        * `loss` (Tensor) - Loss to minimize upon update.
        * `first_order` (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * `allow_unused` (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * `allow_nograd` (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        pass
