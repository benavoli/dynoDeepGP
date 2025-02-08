#!/usr/bin/env python3

import torch
from torch.distributions import Laplace, Cauchy
from torch.nn import Module as TModule
from torch.distributions import constraints, HalfCauchy, Normal
from gpytorch.priors import Prior
from gpytorch.priors.utils import _bufferize_attributes, _del_attributes
from torch.nn import Module as TModule
import math
from numbers import Number


MVN_LAZY_PROPERTIES = ("covariance_matrix", "scale_tril", "precision_matrix")


class LaplacePrior(Prior, Laplace):
    """
   Laplace Prior

    """

    def __init__(self, loc, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        Laplace.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        _bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return LaplacePrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))


class CauchyPrior(Prior, Cauchy):
    """
   Cauchy Prior

    """

    def __init__(self, loc, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        Cauchy.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        _bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return CauchyPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))
#!/usr/bin/env python3




class HorseshoePrior(Prior):
    r"""Horseshoe prior.

    There is no analytical form for the horeshoe prior's pdf, but it
    satisfies a tight bound of the form `lb(x) <= pdf(x) <= ub(x)`, where

        lb(x) = K/2 * log(1 + 4 * (scale / x) ** 2)
        ub(x) = K * log(1 + 2 * (scale / x) ** 2)

    with `K = 1 / sqrt(2 pi^3)`. Here, we simply use

        pdf(x) \sim (lb(x) + ub(x)) / 2

    Reference: C. M. Carvalho, N. G. Polson, and J. G. Scott.
        The horseshoe estimator for sparse signals. Biometrika, 2010.
    """

    arg_constraints = {"scale": constraints.positive}
    support = constraints.real
    _validate_args = True

    def __init__(self, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        if isinstance(scale, Number):
            scale = torch.tensor(float(scale))
        self.K = 1 / math.sqrt(2 * math.pi**3)
        self.scale = scale
        super().__init__(scale.shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.scale
        self.register_buffer("scale", scale)
        self._transform = transform

    def log_prob(self, X):
        self.transform(X)
        A = (self.scale / self.transform(X)) ** 2
        lb = self.K / 2 * torch.log(1 + 4 * A)
        ub = self.K * torch.log(1 + 2 * A)
        return torch.log((lb + ub) / 2)

    def rsample(self, sample_shape=torch.Size([])):
        local_shrinkage = HalfCauchy(1).rsample(self.scale.shape)
        param_sample = Normal(0, local_shrinkage * self.scale).rsample(sample_shape)
        return param_sample

    def expand(self, expand_shape, _instance=None):
        batch_shape = torch.Size(expand_shape)
        return HorseshoePrior(self.scale.expand(batch_shape))