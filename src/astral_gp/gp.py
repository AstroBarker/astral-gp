"""Gaussian process regression on top of scikit-learn.

Two layers.  :class:`SklearnGP` wraps a single ``GaussianProcessRegressor`` with
the kernel

    WhiteKernel(0.1, (1e-3, 10)) + Constant(0.1, (1e-3, 10)) * RBF(l, l_bounds)

fit to whitened inputs and a single whitened output.  :class:`MultiOutputGP`
holds one of those per output column.

Independent fits are not an implementation detail.  sklearn's own multi-output
support shares a single kernel across every column and returns the *same*
posterior standard deviation for all of them, which makes per-output
uncertainty -- and therefore any weighting of it -- meaningless.  One GP per
output costs k fits and gives k genuinely different variances.
"""

import numpy as np
import scipy.optimize
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

#-- Default kernel hyperparameters.
NOISE_LEVEL = 0.1
NOISE_BOUNDS = (1e-3, 10.0)
CONSTANT_VALUE = 0.1
CONSTANT_BOUNDS = (1e-3, 1e1)

#-- Length-scale bounds on whitened inputs.  Whitened data has unit variance,
#-- so a length scale near the lower bound decorrelates every test point from
#-- every training point and the GP falls back to its prior; near the upper
#-- bound the kernel is constant across the domain.  Four decades either side of
#-- unity is generous for real data and keeps the degenerate limits bounded.
#--
#-- A fixed floor is not by itself a cure.  The marginal likelihood is
#-- multimodal, and a single L-BFGS start can still land in the short-length-
#-- scale mode -- it just cannot run away to zero any more.  N_RESTARTS is the
#-- other half of the answer; see below.
LENGTH_SCALE_BOUNDS = (1e-2, 1e2)

#-- Restarts sample additional initial hyperparameters inside the bounds, which
#-- is only safe because the lower bound is strictly positive.  Measured over
#-- three test functions x six seeds x twelve steps, restarts cut degenerate
#-- fits from 5/216 to 2/216; the survivors are a decaying signal where a short
#-- length scale is arguably the honest answer.  Each restart is another
#-- L-BFGS run per output, so this trades roughly 3x fit time for the
#-- reliability.  Set n_restarts=0 if you would rather have the speed.
N_RESTARTS = 2


def whiten(values, axis=None):
    """Standardise to zero mean and unit variance.

    Args:
        values: array of any shape
        axis: ``None`` uses one global mean/std for the whole array; ``0``
            standardises each column independently, which is what multi-output
            targets need if their units differ.

    Returns:
        (whitened, std, mean)
    """
    values = np.asarray(values, dtype=float)
    std = values.std(axis=axis)
    mean = values.mean(axis=axis)
    std = np.where(std > 0.0, std, 1.0) if np.ndim(std) else (std or 1.0)

    return (values - mean) / std, std, mean


def unwhiten(values, std, mean):
    """Undo :func:`whiten` given its returned constants."""
    return np.asarray(values, dtype=float) * std + mean


def lbfgs_optimizer(obj_func, initial_theta, bounds, max_iter=1000):
    """L-BFGS-B with an explicit iteration cap.

    scikit-learn's built-in optimizer runs to ``maxiter=15000``.
    """
    theta_opt, func_min, info = scipy.optimize.fmin_l_bfgs_b(
        obj_func, initial_theta, bounds=bounds, maxiter=max_iter)

    return theta_opt, func_min


class SklearnGP():
    """A single-output GP regressor.

    Args:
        length_scale_bounds: ``(lo, hi)`` on whitened inputs, or the string
            ``'data'`` to derive them from ``(min|x|, max|x|)``.  The latter
            reproduces an older convention in which the lower bound was
            whatever the nearest training point's distance to the mean happened
            to be -- frequently ~0, which lets the length scale collapse.  It is
            kept reachable, not recommended.
    """

    def __init__(self, noise_level=NOISE_LEVEL, noise_bounds=NOISE_BOUNDS,
                 constant_value=CONSTANT_VALUE,
                 constant_bounds=CONSTANT_BOUNDS,
                 length_scale_bounds=LENGTH_SCALE_BOUNDS,
                 n_restarts=N_RESTARTS, max_iter=None, random_state=None):

        self.noise_level = float(noise_level)
        self.noise_bounds = tuple(noise_bounds)
        self.constant_value = float(constant_value)
        self.constant_bounds = tuple(constant_bounds)
        self.length_scale_bounds = length_scale_bounds
        self.n_restarts = int(n_restarts)
        self.max_iter = None if max_iter is None else int(max_iter)
        #-- Restarts draw their initial hyperparameters at random.  Left unset,
        #-- sklearn uses fresh entropy and two fits of identical data disagree
        #-- in the last few digits, so seed it to keep campaigns reproducible.
        self.random_state = random_state

        self.gpr = None
        self.kernel_ = None

    def _bounds_for(self, inputs):
        """Resolve the length-scale bounds against the training inputs."""
        if not isinstance(self.length_scale_bounds, str):
            return tuple(self.length_scale_bounds)

        if self.length_scale_bounds != 'data':
            raise ValueError(
                f"length_scale_bounds must be a (lo, hi) pair or 'data', got "
                f'{self.length_scale_bounds!r}')

        bounds = (np.min(np.abs(inputs)), np.max(np.abs(inputs)))
        if self.n_restarts > 0 and bounds[0] <= 0.0:
            raise ValueError(
                f"length_scale_bounds='data' gave a lower bound of "
                f'{bounds[0]}, and n_restarts={self.n_restarts} samples the '
                'initial theta in log space. Use the default bounds or '
                'n_restarts=0.')

        return bounds

    def train(self, inputs, targets):
        """Fit the GP to whitened ``inputs`` (n, d) and ``targets`` (n,)."""
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        targets = np.asarray(targets, dtype=float).ravel()

        length_scale = np.std(inputs, axis=0)
        length_scale = np.where(length_scale > 0.0, length_scale, 1.0)

        kernel = (
            WhiteKernel(noise_level=self.noise_level,
                        noise_level_bounds=self.noise_bounds)
            + C(self.constant_value, self.constant_bounds)
            * RBF(length_scale=length_scale,
                  length_scale_bounds=self._bounds_for(inputs)))

        optimizer = 'fmin_l_bfgs_b'
        if self.max_iter is not None:
            def optimizer(obj_func, initial_theta, bounds):
                return lbfgs_optimizer(obj_func, initial_theta, bounds,
                                       max_iter=self.max_iter)

        self.gpr = gaussian_process.GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=self.n_restarts,
            optimizer=optimizer,
            random_state=self.random_state).fit(inputs, targets)
        self.kernel_ = self.gpr.kernel_

        return self

    def evaluate(self, inputs, return_std=True):
        """Posterior mean (and standard deviation) at ``inputs``."""
        return self.gpr.predict(np.atleast_2d(np.asarray(inputs, dtype=float)),
                                return_std=return_std)

    def predict_band(self, inputs, n_sigma=1.0):
        """Posterior mean and a +/- ``n_sigma`` band.

        Returns:
            (mean, lower, upper)
        """
        mean, std = self.evaluate(inputs, return_std=True)

        return mean, mean - n_sigma * std, mean + n_sigma * std

    @property
    def hyperparameters(self):
        """Fitted (noise, constant, length scale) of the composite kernel."""
        return (self.kernel_.k1.noise_level,
                self.kernel_.k2.k1.constant_value,
                self.kernel_.k2.k2.length_scale)


class MultiOutputGP():
    """One independent :class:`SklearnGP` per output column.

    Each output is standardised by its own mean and standard deviation before
    fitting, so the per-output posterior variances come back on a common scale
    and can be weighted against each other.
    """

    def __init__(self, **gp_kwargs):
        self.gp_kwargs = gp_kwargs
        self.gps = []
        self.y_std = None
        self.y_mean = None

    def __len__(self):
        return len(self.gps)

    def train(self, inputs, targets):
        """Fit one GP per column of ``targets`` (n, k).

        ``inputs`` should already be whitened; ``targets`` are whitened here,
        per column, and the constants retained for :meth:`predict_physical`.
        """
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        whitened, self.y_std, self.y_mean = whiten(targets, axis=0)

        self.gps = [SklearnGP(**self.gp_kwargs).train(inputs, whitened[:, j])
                    for j in range(targets.shape[1])]

        return self

    def predict(self, inputs, return_std=True):
        """Whitened posterior mean, and std, as (m, k) arrays."""
        results = [gp.evaluate(inputs, return_std=True) for gp in self.gps]
        mean = np.column_stack([r[0] for r in results])
        std = np.column_stack([r[1] for r in results])

        return (mean, std) if return_std else mean

    def variances(self, inputs):
        """Per-output posterior variance, (m, k), in whitened units."""
        _, std = self.predict(inputs, return_std=True)

        return std ** 2

    def predict_band(self, inputs, n_sigma=1.0):
        """Whitened mean and +/- ``n_sigma`` band, each (m, k)."""
        mean, std = self.predict(inputs, return_std=True)

        return mean, mean - n_sigma * std, mean + n_sigma * std

    def predict_physical(self, inputs, n_sigma=1.0):
        """The same band, converted back to the original output units."""
        mean, lower, upper = self.predict_band(inputs, n_sigma=n_sigma)

        return (unwhiten(mean, self.y_std, self.y_mean),
                unwhiten(lower, self.y_std, self.y_mean),
                unwhiten(upper, self.y_std, self.y_mean))

    @property
    def kernels_(self):
        """The fitted kernel of each output's GP."""
        return [gp.kernel_ for gp in self.gps]

    @property
    def hyperparameters(self):
        """Per-output (noise, constant, length scale)."""
        return [gp.hyperparameters for gp in self.gps]


def sklearn_gpr(X, y, X_test, n_sigma=1.0, return_hyperparameters=False,
                max_iter=None, n_restarts=N_RESTARTS):
    """Fit a GP on ``(X, y)`` and predict a band on ``X_test``.

    A functional shortcut for the single-output case; inputs and targets are
    used as given, with no whitening.

    Returns:
        (mean, lower, upper), and optionally (noise, constant, length scale).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and np.ndim(y) and len(np.ravel(y)) != 1:
        X = X.reshape(-1, 1)
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    if X_test.shape[1] != X.shape[1]:
        X_test = X_test.reshape(-1, X.shape[1])

    gp = SklearnGP(n_restarts=n_restarts, max_iter=max_iter).train(X, y)
    mean, lower, upper = gp.predict_band(X_test, n_sigma=n_sigma)

    if return_hyperparameters:
        noise, constant_value, lengthscale = gp.hyperparameters
        return mean, lower, upper, noise, constant_value, lengthscale

    return mean, lower, upper
