"""Evaluation metrics for the GP surrogate.

Three quantities:

* the mean width of the posterior band over the candidate points, which is the
  objective variance-driven placement is implicitly minimising;
* k-fold cross-validated R^2;
* empirical coverage of the posterior at a set of nominal credible levels.

All of them work for any number of input features and any number of outputs,
and all of them evaluate the GP at the points they actually care about rather
than at the nearest node of some grid.
"""

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

from .gp import MultiOutputGP, whiten


def _as_2d(y):
    """Present ``y`` as (n, k) without copying when it is already 2-D."""
    y = np.asarray(y, dtype=float)

    return y.reshape(-1, 1) if y.ndim == 1 else y


def mean_band_width(lower, upper, weights=None):
    """Mean posterior band width over the candidate points.

    The Monte Carlo estimate of the average band width across the sampling
    distribution.  With random candidates it carries sampling error of order
    ``1 / sqrt(n_candidates)``.

    Args:
        lower, upper: (m,) or (m, k) band edges
        weights: optional per-output weights.  When given, the per-output means
            are combined as a weighted average, so the result stays in band
            units rather than growing with the number of outputs.

    Returns:
        float when ``weights`` is given or the input is single-output,
        otherwise an array of k per-output means.
    """
    widths = _as_2d(upper) - _as_2d(lower)
    per_output = widths.mean(axis=0)

    if weights is None:
        return float(per_output[0]) if per_output.size == 1 else per_output

    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError('weights must sum to something positive')

    return float(np.dot(per_output, weights) / total)


def coverage(y, predictions, sigma, alphas=(0.68, 0.95, 0.99)):
    """Fraction of points falling inside each nominal credible interval.

    A calibrated posterior returns roughly ``alphas`` back.  Under-coverage
    means the error bars driving placement are too small to trust.

    Args:
        y, predictions, sigma: (n,) or (n, k) arrays at matching locations
        alphas: nominal levels

    Returns:
        (len(alphas),) for a single output, otherwise (len(alphas), k)
    """
    y, predictions, sigma = _as_2d(y), _as_2d(predictions), _as_2d(sigma)
    z = np.abs((y - predictions) / np.where(sigma > 0.0, sigma, np.inf))
    z_bounds = norm.ppf(np.asarray(alphas, dtype=float) / 2.0 + 0.5)

    #-- (alphas, n, k) -> (alphas, k)
    inside = z[None, :, :] <= z_bounds[:, None, None]
    result = inside.mean(axis=1)

    return result[:, 0] if result.shape[1] == 1 else result


def kfold_r2(X, y, num_folds=6, random_state=42, gp_kwargs=None,
             verbose=False):
    """k-fold cross-validated R^2, per output.

    Each fold refits the GP on the training part and predicts at the held-out
    inputs directly.  R^2 is measured against the mean of each fold's own
    held-out values.

    Args:
        X: (n, d) inputs in physical units
        y: (n,) or (n, k) outputs in physical units
        gp_kwargs: passed through to each :class:`~astral_gp.gp.SklearnGP`

    Returns:
        (num_folds,) for a single output, otherwise (num_folds, k)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = _as_2d(y)
    gp_kwargs = gp_kwargs or {}

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        #-- Whitening constants come from the training part only, so the
        #-- held-out points are genuinely unseen.
        X_white, x_std, x_mean = whiten(X[train_idx], axis=0)
        gp = MultiOutputGP(**gp_kwargs).train(X_white, y[train_idx])

        query = (X[test_idx] - x_mean) / x_std
        predicted, _, _ = gp.predict_physical(query)

        truth = y[test_idx]
        mse = np.mean((predicted - truth) ** 2, axis=0)
        baseline = np.mean((truth.mean(axis=0) - truth) ** 2, axis=0)
        scores.append(1.0 - mse / np.where(baseline > 0.0, baseline, np.inf))

        if verbose:
            print(f'  fold {fold}: R^2 = {np.round(scores[-1], 4)}')

    scores = np.array(scores)

    return scores[:, 0] if scores.shape[1] == 1 else scores


def fit_report(X, y, n_sigma=1.96, alphas=(0.68, 0.95, 0.99), num_folds=6,
               log_x=False, gp_kwargs=None, verbose=True):
    """Fit once, then report MSE, coverage and cross-validated R^2.

    Args:
        X: (n,) or (n, d) inputs
        y: (n,) or (n, k) outputs
        log_x: fit against ``log(X)`` instead of ``X``
        n_sigma: band half-width used to recover sigma for the coverage table

    Returns:
        dict with ``mse``, ``coverage``, ``kfold_r2`` and ``mean_r2``
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1:
        X = X.reshape(-1, 1)
    if log_x:
        X = np.log(X)
    y = _as_2d(y)
    gp_kwargs = gp_kwargs or {}

    X_white, x_std, x_mean = whiten(X, axis=0)
    gp = MultiOutputGP(**gp_kwargs).train(X_white, y)

    predicted, lower, upper = gp.predict_physical(X_white, n_sigma=n_sigma)
    sigma = (upper - lower) / (2.0 * n_sigma)

    mse = np.mean((predicted - y) ** 2, axis=0)
    cov = coverage(y, predicted, sigma, alphas)

    if verbose:
        print(f'N = {len(X)}, {X.shape[1]} feature(s), {y.shape[1]} output(s)')
        print(f'  MSE      : {np.round(mse, 8)}')
        for alpha, c in zip(alphas, np.atleast_2d(cov.T).T):
            print(f'  coverage({alpha:.2f}) = {np.round(np.ravel(c), 4)}')

    scores = kfold_r2(X, y, num_folds=num_folds, gp_kwargs=gp_kwargs,
                      verbose=verbose)
    mean_r2 = np.mean(scores, axis=0)

    if verbose:
        print(f'  mean {num_folds}-fold R^2 = {np.round(mean_r2, 4)}')

    return {'mse': mse[0] if mse.size == 1 else mse,
            'coverage': cov,
            'kfold_r2': scores,
            'mean_r2': float(mean_r2) if np.size(mean_r2) == 1 else mean_r2}
