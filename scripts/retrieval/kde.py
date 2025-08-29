# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a modified version of the implementation from JAX for numerical stability.
It uses a linear solve instead of inverse following the scipy implementation.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.tree_util import register_pytree_node_class
from jax.scipy import special


@register_pytree_node_class
@dataclass(frozen=True, init=False)
class gaussian_kde:  # noqa: N801
    """Gaussian Kernel Density Estimator

    JAX implementation of :class:`scipy.stats.gaussian_kde`.

    Parameters:
      dataset: arraylike, real-valued. Data from which to estimate the distribution.
        If 1D, shape is (n_data,). If 2D, shape is (n_dimensions, n_data).
      bw_method: string, scalar, or callable. Either "scott", "silverman", a scalar
        value, or a callable function which takes ``self`` as a parameter.
      weights: arraylike, optional. Weights of the same shape as the dataset.
    """

    neff: Any
    dataset: Any
    weights: Any
    covariance: Any
    cho_cov: Any

    def __init__(self, dataset, bw_method=None, weights=None):
        check_arraylike("gaussian_kde", dataset)
        dataset = jnp.atleast_2d(dataset)
        if jnp.issubdtype(lax.dtype(dataset), np.complexfloating):
            raise NotImplementedError("gaussian_kde does not support complex data")
        if not dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        d, n = dataset.shape
        if weights is not None:
            check_arraylike("gaussian_kde", weights)
            dataset, weights = promote_dtypes_inexact(dataset, weights)
            weights = jnp.atleast_1d(weights)
            weights /= jnp.sum(weights)
            if weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(weights) != n:
                raise ValueError("`weights` input should be of length n")
        else:
            (dataset,) = promote_dtypes_inexact(dataset)
            weights = jnp.full(n, 1.0 / n, dtype=dataset.dtype)

        self._setattr("dataset", dataset)
        self._setattr("weights", weights)
        neff = self._setattr("neff", 1 / jnp.sum(weights**2))

        bw_method = "scott" if bw_method is None else bw_method
        if bw_method == "scott":
            factor = jnp.power(neff, -1.0 / (d + 4))
        elif bw_method == "silverman":
            factor = jnp.power(neff * (d + 2) / 4.0, -1.0 / (d + 4))
        elif jnp.isscalar(bw_method) and not isinstance(bw_method, str):
            factor = bw_method
        elif callable(bw_method):
            factor = bw_method(self)
        else:
            raise ValueError("`bw_method` should be 'scott', 'silverman', a scalar, or a callable.")

        data_covariance = jnp.atleast_2d(jnp.cov(dataset, rowvar=1, bias=False, aweights=weights))
        data_cho_cov = jnp.linalg.cholesky(data_covariance)
        covariance = data_covariance * factor**2
        self._setattr("covariance", covariance)
        self._setattr("cho_cov", data_cho_cov * factor)

    def _setattr(self, name, value):
        # Frozen dataclasses don't support setting attributes so we have to
        # overload that operation here as they do in the dataclass implementation
        object.__setattr__(self, name, value)
        return value

    def tree_flatten(self):
        return ((self.neff, self.dataset, self.weights, self.covariance, self.inv_cov), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        kde = cls.__new__(cls)
        kde._setattr("neff", children[0])
        kde._setattr("dataset", children[1])
        kde._setattr("weights", children[2])
        kde._setattr("covariance", children[3])
        kde._setattr("inv_cov", children[4])
        return kde

    @property
    def d(self):
        return self.dataset.shape[0]

    @property
    def n(self):
        return self.dataset.shape[1]

    def evaluate(self, points):
        """Evaluate the Gaussian KDE on the given points."""
        check_arraylike("evaluate", points)
        points = self._reshape_points(points)
        result = _gaussian_kernel_eval(False, self.dataset.T, self.weights[:, None], points.T, self.cho_cov)
        return result[:, 0]

    def __call__(self, points):
        return self.evaluate(points)

    def pdf(self, x):
        """Probability density function"""
        return self.evaluate(x)

    def logpdf(self, x):
        """Log probability density function"""
        check_arraylike("logpdf", x)
        x = self._reshape_points(x)
        result = _gaussian_kernel_eval(True, self.dataset.T, self.weights[:, None], x.T, self.cho_cov)
        return result[:, 0]

    def _reshape_points(self, points):
        if jnp.issubdtype(lax.dtype(points), np.complexfloating):
            raise NotImplementedError("gaussian_kde does not support complex coordinates")
        points = jnp.atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                points = jnp.reshape(points, (self.d, 1))
            else:
                raise ValueError("points have dimension {}, dataset has dimension {}".format(d, self.d))
        return points


@partial(jit, static_argnums=0)
def _gaussian_kernel_eval(in_log, points, values, xi, precision):
    points, values, xi, precision = promote_dtypes_inexact(points, values, xi, precision)
    d = points.shape[1]

    if xi.shape[1] != d:
        raise ValueError("points and xi must have same trailing dim")
    if precision.shape != (d, d):
        raise ValueError("precision matrix must match data dims")

    # New Implementation
    points = jax.scipy.linalg.solve_triangular(precision, points.T, lower=True).T
    xi = jax.scipy.linalg.solve_triangular(precision, xi.T, lower=True).T
    log_norm = -0.5 * d * jnp.log(2 * np.pi) - jnp.sum(jnp.log(jnp.diag(precision)))

    def kernel(x_test, x_train, y_train):
        arg = log_norm - 0.5 * jnp.sum(jnp.square(x_train - x_test))
        if in_log:
            return jnp.log(y_train) + arg
        return y_train * jnp.exp(arg)

    reduce = special.logsumexp if in_log else jnp.sum
    reduced_kernel = lambda x: reduce(vmap(kernel, in_axes=(None, 0, 0))(x, points, values), axis=0)  # noqa: E731
    mapped_kernel = vmap(reduced_kernel)

    return mapped_kernel(xi)
