import math

import jax
import jax.numpy as jnp
import functools
import jax.random as jr
import numpy as np
import scipy
from jaxtyping import ArrayLike

from .inception import InceptionV3

### Attribution
# This project includes code adapted from
# [jax-fid by Matthias Wright et al.] (Apache License 2.0),
# a JAX/Flax implementation of FID. The original project is
# available under the Apache License, Version 2.0.

# https://github.com/matthias-wright/jax-fid/blob/main/LICENSE

def compute_statistics(images: ArrayLike, params, apply_fn, batch_size):
    images = np.array(images, np.float32)  # They are assumed to be in [0, 1]
    assert images.ndim == 4, f"images.shape = {images.shape}"
    tot_len = images.shape[0]
    num_batches = math.ceil(tot_len / batch_size)
    n_channels = images.shape[3]
    assert images.shape[1] == images.shape[2]

    @jax.jit
    def compute_act(_x):
        _x = jnp.array(_x)
        side_len = _x.shape[1]
        if side_len != 256:
            _x = jax.image.resize(
                _x,
                (_x.shape[0], 256, 256, 1),
                method="bilinear",
                antialias=True,
            )

        if n_channels == 1:
            _x = jnp.tile(_x, (1, 1, 1, 3))
        elif n_channels != 3:
            raise ValueError("Images must have 1 or 3 channels.")

        _x = 2 * _x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(_x))
        return pred.squeeze(axis=1).squeeze(axis=1)

    act = []
    for i in range(num_batches):
        x = images[i * batch_size : i * batch_size + batch_size]
        act.append(np.array(compute_act(x)))
    act = np.concatenate(act, axis=0)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    first_term = diff.dot(diff)
    second_term = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return first_term + second_term, first_term, second_term


def get_apply_fn():
    model = InceptionV3(pretrained=True)
    params = model.init(jr.key(0), jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    return params, apply_fn

def get_fid_fn(batch_size, ref_stats_filename):
    ref_stats = np.load(ref_stats_filename)
    ref_mu, ref_sigma = ref_stats["mu"], ref_stats["sigma"]

    params, apply_fn = get_apply_fn()

    def compute_fid_fn(images: ArrayLike):
        mu, sigma = compute_statistics(images, params, apply_fn, batch_size)
        fid_score = compute_frechet_distance(mu, ref_mu, sigma, ref_sigma)
        return fid_score

    return compute_fid_fn

def bootstrap_fid_fn(images, ref_stats_path):

    fid_fn = get_fid_fn(100, ref_stats_path)
    images = images.reshape(-1, 28, 28, 1)  #hardcoded channel=1 for fid calculation for now, hardcoded 28 side len
    assert images.ndim == 4, images.shape  # (n_samples, h, w, c)
    # Compute FID and its components on the full dataset.
    fid, fid_term1, fid_term2 = fid_fn(images)
    return float(fid), float(fid_term1), float(fid_term2)