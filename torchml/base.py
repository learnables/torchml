import torch


class Model(torch.nn.Module):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    ## Arguments

    * `arg1` (int) - The first argument.

    ## Example

    ~~~python
    ~~~
    """

    def fit(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
