"""Random candidate points for variance-driven placement.

The GP's posterior variance has to be evaluated *somewhere* before the argmax
means anything.  A dense grid works in one dimension and becomes hopeless by
about four, so candidates are drawn at random instead: ``n_test`` points per
step, redrawn every step, from a seeded generator.

The default draws uniformly over the bounding box of the data seen so far.
Anything else is expressed as a distribution, which is where scipy comes in --
:func:`build_sampler` accepts frozen ``scipy.stats`` distributions directly, so
a prior over where sampling is worthwhile costs one line:

    Astral(sampler=stats.norm(loc=[20.0, 0.5], scale=[3.0, 0.1]))

Candidates are produced in physical units; whitening happens downstream, so
bounds and distributions are written in the units of the problem.
"""

import numpy as np
from scipy import stats


def bounds_from_data(X, margin=0.0):
    """Per-feature ``(lo, hi)`` covering ``X``.

    Args:
        X: (n, d) array
        margin: fractional padding added to each side of the span

    Returns:
        list of d (lo, hi) tuples
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    return [(float(a), float(b))
            for a, b in zip(lo - margin * span, hi + margin * span)]


def uniform_over(bounds):
    """A frozen uniform distribution over a per-feature box.

    scipy broadcasts array ``loc``/``scale`` across ``size=(n, d)``, so one
    frozen object covers every feature at once.
    """
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)

    return stats.uniform(loc=lo, scale=hi - lo)


def _is_multivariate(dist, n_features):
    """True when ``dist.rvs(size=n)`` already returns one row per draw.

    ``multivariate_normal`` and friends interpret ``size`` as a number of
    vectors; univariate distributions interpret it as a shape.  Probing once is
    cheaper than trying to enumerate which is which.
    """
    try:
        probe = np.asarray(dist.rvs(size=3))
    except Exception:
        return False

    return probe.ndim == 2 and probe.shape == (3, n_features)


def _from_distribution(dist, n_features):
    """Wrap a frozen scipy distribution as ``callable(n, rng) -> (n, d)``."""
    multivariate = _is_multivariate(dist, n_features)

    def sample(n, rng):
        size = n if multivariate else (n, n_features)
        drawn = np.asarray(dist.rvs(size=size, random_state=rng), dtype=float)
        drawn = drawn.reshape(n, -1)

        if drawn.shape[1] != n_features:
            raise ValueError(
                f'sampler produced {drawn.shape[1]} features, expected '
                f'{n_features}. Give one distribution per feature, or a '
                'distribution whose rvs() returns rows of the right width.')

        return drawn

    return sample


def _from_distribution_list(dists, n_features):
    """Wrap one frozen distribution per feature."""
    if len(dists) != n_features:
        raise ValueError(
            f'got {len(dists)} distributions for {n_features} features')

    def sample(n, rng):
        columns = [np.asarray(d.rvs(size=n, random_state=rng), dtype=float)
                   for d in dists]
        return np.column_stack(columns)

    return sample


def build_sampler(sampler=None, bounds=None, X=None, n_features=None):
    """Normalise the accepted sampler forms into one callable.

    Accepts:
        None: uniform over ``bounds``, or over the bounding box of ``X``
        a frozen scipy distribution: anything exposing ``rvs``
        a sequence of frozen distributions: one per feature
        a callable ``f(n, rng) -> (n, d)``: used as given

    Returns:
        callable(n, rng) -> (n, d) array of candidates in physical units
    """
    if n_features is None:
        if bounds is not None:
            n_features = len(bounds)
        elif X is not None:
            n_features = np.atleast_2d(X).shape[1]
        else:
            raise ValueError('need one of n_features, bounds or X')

    if sampler is None:
        if bounds is None:
            if X is None:
                raise ValueError('need bounds or X to build a default sampler')
            bounds = bounds_from_data(X)
        return _from_distribution(uniform_over(bounds), n_features)

    if hasattr(sampler, 'rvs'):
        return _from_distribution(sampler, n_features)

    if isinstance(sampler, (list, tuple)) and all(
            hasattr(d, 'rvs') for d in sampler):
        return _from_distribution_list(sampler, n_features)

    if callable(sampler):
        def sample(n, rng):
            drawn = np.asarray(sampler(n, rng), dtype=float).reshape(n, -1)
            if drawn.shape[1] != n_features:
                raise ValueError(
                    f'sampler returned {drawn.shape[1]} features, expected '
                    f'{n_features}')
            return drawn
        return sample

    raise TypeError(
        'sampler must be None, a frozen scipy distribution, a sequence of '
        f'them, or a callable(n, rng); got {type(sampler).__name__}')
