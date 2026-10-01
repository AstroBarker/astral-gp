"""The GP layer: independent per-output fits, and the length-scale floor."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from astral_gp import MultiOutputGP, SklearnGP, sklearn_gpr, unwhiten, whiten
from astral_gp.gp import LENGTH_SCALE_BOUNDS


def sample_problem(n=30, seed=0):
    """Three outputs of deliberately different smoothness."""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0.0, 6.0, n)).reshape(-1, 1)
    y = np.column_stack([np.sin(X[:, 0]),
                         np.sin(8.0 * X[:, 0]),
                         100.0 * X[:, 0]])

    return X, y


#-- Whitening --------------------------------------------------------------


def test_whiten_global_vs_per_column():
    values = np.array([[1.0, 100.0], [3.0, 300.0]])

    _, global_std, _ = whiten(values)
    _, column_std, _ = whiten(values, axis=0)

    assert np.ndim(global_std) == 0
    assert column_std.shape == (2,)
    assert column_std[1] > column_std[0]


def test_whiten_roundtrip():
    values = np.array([[1.0, 100.0], [3.0, 300.0], [2.0, 250.0]])

    w, std, mean = whiten(values, axis=0)

    np.testing.assert_allclose(unwhiten(w, std, mean), values)
    np.testing.assert_allclose(w.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(w.std(axis=0), 1.0)


def test_whiten_survives_a_constant_column():
    values = np.array([[5.0, 1.0], [5.0, 2.0]])

    w, std, mean = whiten(values, axis=0)

    assert np.all(np.isfinite(w))
    assert std[0] == 1.0


#-- Independent per-output fits --------------------------------------------


def test_sklearn_multioutput_shares_one_std():
    """The behaviour MultiOutputGP exists to avoid.

    sklearn fits one kernel across all columns and reports the same posterior
    std for each, so any per-output weighting would be a no-op.
    """
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)
    yw, _, _ = whiten(y, axis=0)

    kernel = WhiteKernel(0.1, (1e-3, 10)) + C(0.1, (1e-3, 10)) * RBF(
        1.0, LENGTH_SCALE_BOUNDS)
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=0).fit(Xw, yw)
    _, std = gpr.predict(Xw, return_std=True)

    np.testing.assert_allclose(std[:, 0], std[:, 1])


def test_multioutput_gp_gives_per_output_variances():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    gp = MultiOutputGP().train(Xw, y)
    variances = gp.variances(Xw)

    assert len(gp) == 3
    assert variances.shape == (len(X), 3)
    assert not np.allclose(variances[:, 0], variances[:, 1]), (
        'a rough and a smooth output should not share a variance')


def test_multioutput_gp_fits_a_separate_kernel_per_output():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    gp = MultiOutputGP().train(Xw, y)

    assert len(gp.kernels_) == 3
    assert len({str(k) for k in gp.kernels_}) == 3


def test_rough_output_gets_the_shorter_length_scale():
    X, y = sample_problem(n=60)
    Xw, _, _ = whiten(X, axis=0)

    gp = MultiOutputGP().train(Xw, y)
    smooth = float(np.atleast_1d(gp.hyperparameters[0][2])[0])
    rough = float(np.atleast_1d(gp.hyperparameters[1][2])[0])

    assert rough < smooth


def test_predict_physical_restores_output_units():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    gp = MultiOutputGP().train(Xw, y)
    mean, lower, upper = gp.predict_physical(Xw)

    assert mean.shape == y.shape
    assert np.all(lower <= mean) and np.all(mean <= upper)
    #-- the third output spans 0..600; a whitened mean would not
    assert mean[:, 2].max() > 100.0


def test_single_output_accepted_as_flat_array():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    gp = MultiOutputGP().train(Xw, y[:, 0])

    assert len(gp) == 1
    assert gp.variances(Xw).shape == (len(X), 1)


#-- Length-scale bounds ----------------------------------------------------


def test_default_bounds_floor_the_length_scale():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    gp = SklearnGP().train(Xw, y[:, 0])
    ell = float(np.atleast_1d(gp.hyperparameters[2])[0])

    assert LENGTH_SCALE_BOUNDS[0] <= ell <= LENGTH_SCALE_BOUNDS[1]


def test_legacy_data_bounds_still_reachable():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    gp = SklearnGP(length_scale_bounds='data', n_restarts=0).train(Xw, y[:, 0])

    assert gp.kernel_ is not None


def test_legacy_bounds_reject_restarts_when_the_floor_is_zero():
    """A training point on the mean gives min|x| == 0, i.e. log(0)."""
    X = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    y = X[:, 0] ** 2

    gp = SklearnGP(length_scale_bounds='data', n_restarts=3)

    with pytest.raises(ValueError, match='n_restarts'):
        gp.train(X, y)


def test_unknown_bounds_string_is_rejected():
    X, y = sample_problem()

    with pytest.raises(ValueError, match='must be'):
        SklearnGP(length_scale_bounds='auto').train(X, y[:, 0])


#-- Functional shortcut ----------------------------------------------------


def test_sklearn_gpr_returns_an_ordered_band():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    mean, lower, upper = sklearn_gpr(Xw, y[:, 0], Xw)

    assert mean.shape == (len(X),)
    assert np.all(lower <= mean) and np.all(mean <= upper)


def test_sklearn_gpr_can_return_hyperparameters():
    X, y = sample_problem()
    Xw, _, _ = whiten(X, axis=0)

    result = sklearn_gpr(Xw, y[:, 0], Xw, return_hyperparameters=True)

    assert len(result) == 6
