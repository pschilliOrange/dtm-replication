import sys
import os

_parent_dir = os.path.abspath("..")
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

import jax
import jax.numpy as jnp


def autocorr_1d(x):
    """
    Computes the autocorrelation of a 1D time series.
    Expects x to be a 2D array of shape (batch_dim, time_dim).
    """
    assert x.ndim == 2
    n = x.shape[1] // 2
    x = x / jnp.std(x)
    x0 = x[:, :n]
    mean_0 = jnp.mean(x0)
    x_infinity = x[:, -(n // 2) :]
    mean_infinity = jnp.mean(x_infinity)

    @jax.jit
    def corr_fn(lag):
        x1 = jax.lax.dynamic_slice(x, [0, lag], (x.shape[0], n))
        return jnp.mean(x0 * x1)

    vec_corr = jax.lax.map(corr_fn, jnp.arange(0, n + 1))
    return vec_corr - (mean_0 * mean_infinity)


def autocorr_fn(x, backend):
    """
    Computes the autocorrelation of a multi-dimensional time series.
    Input x should have shape (n_cores, n_reps, n_chains, n_samples, data_dim).
    """

    @jax.jit
    def inner_vmap_fn(x):
        # Map autocorr_1d over the n_reps dimension.
        out = jax.vmap(autocorr_1d, in_axes=0, out_axes=0)(x)
        return jnp.mean(out, axis=0)

    @jax.jit
    def outer_vmap_fn(x):
        # Map over the data_dim dimension.
        out = jax.vmap(inner_vmap_fn, in_axes=3, out_axes=0)(x)
        return jnp.mean(out, axis=0)

    out = jax.pmap(outer_vmap_fn, in_axes=0, out_axes=0, backend=backend)(x)
    return jnp.mean(out, axis=0)
