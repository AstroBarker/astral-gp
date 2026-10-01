"""Variance-driven placement of training data for a GP surrogate.

The idea the whole package is built around: when each new training point is
expensive, spend it where the surrogate is least certain.  Fit a GP to what you
have, evaluate its posterior variance at a set of random candidate points, and
take the next sample at the argmax.  Repeat.

Inputs are ``(n, d)`` for any number of features and outputs are ``(n, k)`` for
any number of targets, each target getting its own independent GP.  Placement
maximises a weighted sum of the per-output posterior variances, so you can say
which targets you actually care about.

A single class carries the data, the whitening constants, the fitted GPs and
the run history.  New target values arrive one of three ways -- from a callable,
from a pool of already-evaluated points, or handed back by the caller:

    ast = Astral(test_name='demo', n_test=2000, seed=0)
    ast.build_initial_design([(0.0, 1.0), (0.0, 1.0)], 12, simulator=f)
    ast.run_loop(30, simulator=f)

    ast.step(pool_X=X_done, pool_y=y_done)   # pool of finished evaluations

    X_new = ast.place_next()                 # hand off and come back
    ast.add_point(X_new, expensive_code(X_new))
"""

import json
from pathlib import Path

import numpy as np

from .gp import (
    LENGTH_SCALE_BOUNDS,
    N_RESTARTS,
    MultiOutputGP,
    unwhiten,
    whiten,
)
from .metrics import coverage, kfold_r2, mean_band_width
from .sampling import bounds_from_data, build_sampler


class Astral():
    """A GP surrogate that chooses where to sample next.

    Args:
        test_name: run label; output goes to ``{runs_dir}/{test_name}``
        n_test: number of random candidate points drawn per step
        n_eval: size of a fixed evaluation set used only for reporting
            progress.  ``None`` reports on the placement candidates themselves,
            which adds Monte Carlo jitter of order ``1/sqrt(n_test)`` to the
            uncertainty curve; setting it gives a curve comparable across steps.
        num_new_samples: points placed per step
        weights: per-output weights for the placement score.  Defaults to ones,
            i.e. every output counts equally.
        sampler: ``None`` for uniform over the data's bounding box, or a frozen
            scipy distribution, a sequence of them (one per feature), or a
            callable ``f(n, rng) -> (n, d)``.  See :mod:`astral_gp.sampling`.
        bounds: per-feature ``(lo, hi)`` for the default sampler.  ``None``
            derives them from the data at each step.
        seed: seeds the candidate draws, making a run reproducible
        n_sigma: band half-width in posterior standard deviations.  Placement is
            unaffected; the reported band width scales linearly with it.
        length_scale_bounds: passed to each GP; see
            :class:`~astral_gp.gp.SklearnGP`
    """

    def __init__(self, test_name='astral', n_test=1000, n_eval=None,
                 num_new_samples=1, weights=None, sampler=None, bounds=None,
                 seed=None, noise_level=0.1, noise_bounds=(1e-3, 10.0),
                 constant_value=0.1, constant_bounds=(1e-3, 1e1),
                 length_scale_bounds=LENGTH_SCALE_BOUNDS,
                 n_restarts=N_RESTARTS, n_sigma=1.0,
                 max_iter=None, runs_dir='runs', verbose=True):

        self.test_name = test_name
        self.n_test = int(n_test)
        self.n_eval = None if n_eval is None else int(n_eval)
        self.num_new_samples = int(num_new_samples)
        self.n_sigma = float(n_sigma)
        self.bounds = None if bounds is None else [tuple(b) for b in bounds]
        self.sampler_spec = sampler
        self.seed = seed
        self.verbose = bool(verbose)

        self.gp_kwargs = {'noise_level': noise_level,
                          'noise_bounds': noise_bounds,
                          'constant_value': constant_value,
                          'constant_bounds': constant_bounds,
                          'length_scale_bounds': length_scale_bounds,
                          'n_restarts': n_restarts,
                          'max_iter': max_iter,
                          'random_state': seed}

        self._rng = np.random.default_rng(seed)
        self._weights = None if weights is None else np.asarray(weights,
                                                                dtype=float)

        #-- Create output directory
        self.output_dir = Path(runs_dir) / self.test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        #-- Data.  X is (n, d) and y is (n, k), both in physical units; the
        #-- whitening constants are recomputed from scratch on every fit.
        self.X = None
        self.y = None
        self.iteration = 0

        #-- Pool of pre-computed evaluations, when stepping against finished data
        self.pool_X = None
        self.pool_y = None
        self.pool_used = None

        #-- Populated by train_gp()
        self.gp = None
        self.x_std, self.x_mean = None, None
        self.test_X, self.test_X_physical = None, None
        self.mean, self.lower, self.upper = None, None, None
        self._eval_X = None

        self.history = {'iteration': [], 'n': [], 'uncertainty': [],
                        'placement': [], 'added': [], 'kernels': []}

        if self.verbose:
            print(f'astral: {self.test_name} -> {self.output_dir}')

    #-- Shapes -----------------------------------------------------------

    @property
    def n_features(self):
        return None if self.X is None else self.X.shape[1]

    @property
    def n_outputs(self):
        return None if self.y is None else self.y.shape[1]

    @property
    def weights(self):
        """Per-output placement weights, defaulting to ones."""
        if self._weights is None:
            return np.ones(self.n_outputs or 1)

        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = None if value is None else np.asarray(value,
                                                              dtype=float)

    #-- Data --------------------------------------------------------------

    @staticmethod
    def _as_X(X):
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)

    @staticmethod
    def _as_y(y):
        y = np.asarray(y, dtype=float)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def set_data(self, X, y, weights=None):
        """Install training data.

        Args:
            X: (n,) or (n, d) inputs
            y: (n,) or (n, k) outputs
            weights: optional per-output placement weights
        """
        self.X = self._as_X(X)
        self.y = self._as_y(y)

        if len(self.X) != len(self.y):
            raise ValueError(
                f'X has {len(self.X)} rows but y has {len(self.y)}')

        if weights is not None:
            self.weights = weights
        if self._weights is not None and len(self._weights) != self.n_outputs:
            raise ValueError(
                f'{len(self._weights)} weights for {self.n_outputs} outputs')

        self.mean = None

        return self

    def build_initial_design(self, bounds, n_initial, simulator=None,
                             pool_X=None, pool_y=None):
        """Seed with ``n_initial`` random points drawn over ``bounds``.

        Values come from ``simulator`` or from the nearest members of a pool,
        exactly as :meth:`step` gets them.

        Args:
            bounds: per-feature ``(lo, hi)``.  A bare ``(lo, hi)`` is accepted
                for one feature.
        """
        bounds = self._normalise_bounds(bounds)
        if self.bounds is None:
            self.bounds = bounds

        sampler = build_sampler(self.sampler_spec, bounds=bounds,
                                n_features=len(bounds))
        X_init = sampler(int(n_initial), self._rng)

        if self.verbose:
            print(f'building initial design: {n_initial} points in '
                  f'{len(bounds)}-D')

        if pool_X is not None:
            self.set_pool(pool_X, pool_y)

        X_init, y_init = self._evaluate(X_init, simulator)
        self.set_data(X_init, y_init)
        self.iteration = 0

        return self

    @staticmethod
    def _normalise_bounds(bounds):
        """Accept ``(lo, hi)`` or ``[(lo, hi), ...]``."""
        bounds = list(bounds)
        if len(bounds) == 2 and np.isscalar(bounds[0]):
            return [tuple(bounds)]

        return [tuple(b) for b in bounds]

    def set_pool(self, pool_X, pool_y):
        """Register a pool of finished evaluations to draw new points from."""
        self.pool_X = self._as_X(pool_X)
        self.pool_y = self._as_y(pool_y)
        self.pool_used = np.zeros(len(self.pool_X), dtype=bool)

        if self.verbose:
            print(f'pool: {len(self.pool_X)} finished evaluations, '
                  f'{self.pool_X.shape[1]} feature(s)')

        return self

    #-- Fitting -----------------------------------------------------------

    def _draw_candidates(self, n):
        """``n`` random candidate points, in physical units."""
        bounds = self.bounds or bounds_from_data(self.X)
        sampler = build_sampler(self.sampler_spec, bounds=bounds, X=self.X,
                                n_features=self.n_features)

        return sampler(int(n), self._rng)

    def train_gp(self):
        """Whiten, fit one GP per output, and predict on fresh candidates.

        Whitening constants are recomputed per feature and per output from the
        current data every time.
        """
        if self.X is None:
            raise RuntimeError('no data; call set_data() or '
                               'build_initial_design()')

        X_white, self.x_std, self.x_mean = whiten(self.X, axis=0)

        self.test_X_physical = self._draw_candidates(self.n_test)
        self.test_X = (self.test_X_physical - self.x_mean) / self.x_std

        self.gp = MultiOutputGP(**self.gp_kwargs).train(X_white, self.y)
        self.mean, self.lower, self.upper = self.gp.predict_band(
            self.test_X, n_sigma=self.n_sigma)

        return self.mean, self.lower, self.upper

    def band_physical(self, X_white=None):
        """The fitted band at the candidates, in the original output units."""
        query = self.test_X if X_white is None else X_white

        return self.gp.predict_physical(query, n_sigma=self.n_sigma)

    #-- Progress ----------------------------------------------------------

    def _evaluation_points(self):
        """Whitened points at which progress is measured.

        With ``n_eval`` set, a fixed set is drawn once and reused so the
        uncertainty curve is comparable across steps; otherwise the placement
        candidates are used directly.
        """
        if self.n_eval is None:
            return self.test_X

        if self._eval_X is None:
            self._eval_X = self._draw_candidates(self.n_eval)

        return (self._eval_X - self.x_mean) / self.x_std

    def mean_uncertainty(self, per_output=False):
        """Mean posterior band width over the evaluation points.

        Computed on whitened outputs so that the single number stays comparable
        across outputs carrying different units.

        Args:
            per_output: return the k band widths in physical units instead of
                one weighted average

        Returns:
            float, or an array of k physical band widths
        """
        if self.mean is None:
            self.train_gp()

        query = self._evaluation_points()

        if per_output:
            _, lower, upper = self.gp.predict_physical(query,
                                                       n_sigma=self.n_sigma)
            return mean_band_width(lower, upper)

        _, lower, upper = self.gp.predict_band(query, n_sigma=self.n_sigma)

        return mean_band_width(lower, upper, weights=self.weights)

    #-- Placement ---------------------------------------------------------

    def placement_score(self):
        """Weighted sum of per-output posterior variances at the candidates.

        ``score(x) = sum_j w_j * var_j(x)``, with the variances in whitened
        units so the weights compare like with like.
        """
        if self.mean is None:
            self.train_gp()

        variances = self.gp.variances(self.test_X)

        return variances @ self.weights

    def place_next(self, num_new_samples=None):
        """Candidate points of largest weighted variance, in physical units.

        Returns:
            (num_new_samples, d) array
        """
        k = (self.num_new_samples if num_new_samples is None
             else int(num_new_samples))
        score = self.placement_score()
        indices = np.argsort(-score, kind='stable')[:k]

        return self.test_X_physical[indices]

    #-- Stepping ----------------------------------------------------------

    def _evaluate(self, X_new, simulator=None):
        """Obtain y for ``X_new`` from a callable or from the pool.

        Returns the possibly-adjusted X alongside y: in pool mode the returned
        X is the pool member actually used, not the requested location.
        """
        X_new = self._as_X(X_new)

        if simulator is not None:
            y_new = np.asarray(simulator(X_new), dtype=float)
            return X_new, y_new.reshape(len(X_new), -1)

        if self.pool_X is None:
            raise ValueError(
                'no simulator and no pool. Pass simulator=..., or '
                'pool_X/pool_y, or use place_next() and add_point() to supply '
                'values yourself.')

        #-- Nearest unused pool member, by distance in whitened input space so
        #-- features with different units contribute comparably.
        scale = self.x_std if self.x_std is not None else np.ones(
            self.pool_X.shape[1])

        chosen = []
        for row in X_new:
            available = np.flatnonzero(~self.pool_used)
            if len(available) == 0:
                raise RuntimeError('pool exhausted')
            gap = (self.pool_X[available] - row) / scale
            idx = available[np.einsum('ij,ij->i', gap, gap).argmin()]
            self.pool_used[idx] = True
            chosen.append(idx)

        chosen = np.array(chosen)

        return self.pool_X[chosen], self.pool_y[chosen]

    def add_point(self, X, y):
        """Append points. New rows go on the end; data is never re-sorted."""
        X = self._as_X(X)
        y = np.asarray(y, dtype=float).reshape(len(X), -1)

        self.X = np.vstack([self.X, X])
        self.y = np.vstack([self.y, y])
        self.mean = None

        return self

    def step(self, simulator=None, pool_X=None, pool_y=None, record=True):
        """Advance by one placement.

        Fits the GPs, places at the largest weighted posterior variance,
        obtains the new target value(s) and appends them.

        Returns:
            (X_new, y_new) as actually added
        """
        if pool_X is not None:
            self.set_pool(pool_X, pool_y)

        self.train_gp()

        uncertainty = self.mean_uncertainty()
        X_request = self.place_next()
        X_new, y_new = self._evaluate(X_request, simulator)

        if record:
            self.history['iteration'].append(self.iteration)
            self.history['n'].append(len(self.X))
            self.history['uncertainty'].append(uncertainty)
            self.history['placement'].append(X_request[0].tolist())
            self.history['added'].append(np.ravel(X_new[0]).tolist())
            self.history['kernels'].append(
                [str(k) for k in self.gp.kernels_])

        if self.verbose:
            print(f'  step {self.iteration:3d}  N={len(self.X):4d}  '
                  f'U={uncertainty:.6g}  placing at '
                  f'{np.array2string(X_request[0], precision=4)}')

        self.add_point(X_new, y_new)
        self.iteration += 1

        return X_new, y_new

    def run_loop(self, n_steps, simulator=None, pool_X=None, pool_y=None,
                 save=False):
        """Repeat :meth:`step` ``n_steps`` times.

        Args:
            save: also write ``data_at_step_{i}.json`` at every step
        """
        if pool_X is not None:
            self.set_pool(pool_X, pool_y)

        for _ in range(int(n_steps)):
            if save:
                self.save_data()
            self.step(simulator=simulator)

        #-- Fit once more so the final data has trained GPs behind it
        self.train_gp()
        if save:
            self.save_data()

        return self.history

    #-- Persistence -------------------------------------------------------

    def save_data(self, step=None, data_dir=None):
        """Write the current data as ``data_at_step_{i}.json``."""
        step = self.iteration if step is None else int(step)
        data_dir = self.output_dir if data_dir is None else Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        path = data_dir / f'data_at_step_{step}.json'
        with open(path, 'w') as f:
            json.dump({'X': self.X.tolist(), 'y': self.y.tolist()}, f, indent=4)

        return path

    def load_data(self, step, data_dir=None, n_features=1):
        """Read data written by :meth:`save_data`.

        Also accepts the older ``{"data": [[x, y...], ...]}`` layout, in which
        the first ``n_features`` columns are inputs and the rest outputs.
        """
        data_dir = self.output_dir if data_dir is None else Path(data_dir)
        path = Path(data_dir) / f'data_at_step_{step}.json'
        if not path.exists():
            path = Path(data_dir) / f'grid_at_step_{step}.json'

        with open(path, 'r') as f:
            payload = json.load(f)

        if 'X' in payload:
            self.set_data(np.array(payload['X']), np.array(payload['y']))
        else:
            matrix = np.array(payload['data'])
            self.set_data(matrix[:, :n_features], matrix[:, n_features:])

        self.iteration = int(step)

        return self

    def save_history(self, path=None):
        """Write the run history as JSON."""
        path = self.output_dir / 'history.json' if path is None else Path(path)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)

        return path

    #-- Evaluation --------------------------------------------------------

    def kfold_r2(self, num_folds=6, random_state=42):
        """k-fold cross-validated R^2 on the current data, per output."""
        return kfold_r2(self.X, self.y, num_folds=num_folds,
                        random_state=random_state, gp_kwargs=self.gp_kwargs,
                        verbose=self.verbose)

    def coverage(self, alphas=(0.68, 0.95, 0.99)):
        """Empirical coverage of the posterior at the training points."""
        if self.mean is None:
            self.train_gp()

        X_white = (self.X - self.x_mean) / self.x_std
        predicted, lower, upper = self.gp.predict_physical(
            X_white, n_sigma=self.n_sigma)
        sigma = (upper - lower) / (2.0 * self.n_sigma)

        return coverage(self.y, predicted, sigma, alphas)

    #-- Plotting ----------------------------------------------------------

    def make_plots(self, xlabel='x', ylabel='y', output=0, dpi=300):
        """Write diagnostic figures to the run directory."""
        from .plotting import plot_fit, plot_slice, plot_uncertainty_curve

        if self.mean is None:
            self.train_gp()

        if self.n_features == 1:
            paths = [plot_fit(self, xlabel=xlabel, ylabel=ylabel,
                              output=output, dpi=dpi)]
        else:
            paths = [plot_slice(self, feature=0, xlabel=xlabel, ylabel=ylabel,
                                output=output, dpi=dpi)]

        if self.history['uncertainty']:
            paths.append(plot_uncertainty_curve(self, dpi=dpi))

        if self.verbose:
            for path in paths:
                print(f'wrote {path}')

        return paths
