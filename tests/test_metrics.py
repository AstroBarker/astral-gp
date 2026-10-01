"""Evaluation metrics, for one output and for several."""

import numpy as np
import pytest

from astral_gp import Astral, coverage, fit_report, kfold_r2, mean_band_width


def smooth(X):
    return np.sin(X[:, 0]) + 0.2 * X[:, 0]


def three_outputs(X):
    x = X[:, 0]

    return np.column_stack([np.sin(x), np.cos(x), 0.5 * x])


#-- Band width -------------------------------------------------------------


def test_mean_band_width_single_output():
    lower, upper = np.zeros(100), np.full(100, 2.0)

    assert mean_band_width(lower, upper) == pytest.approx(2.0)


def test_mean_band_width_per_output():
    lower = np.zeros((100, 2))
    upper = np.column_stack([np.full(100, 2.0), np.full(100, 6.0)])

    np.testing.assert_allclose(mean_band_width(lower, upper), [2.0, 6.0])


def test_mean_band_width_weighted_is_an_average_not_a_sum():
    lower = np.zeros((100, 2))
    upper = np.column_stack([np.full(100, 2.0), np.full(100, 6.0)])

    assert mean_band_width(lower, upper, weights=[1.0, 1.0]) == pytest.approx(4.0)
    assert mean_band_width(lower, upper, weights=[1.0, 0.0]) == pytest.approx(2.0)
    assert mean_band_width(lower, upper, weights=[3.0, 1.0]) == pytest.approx(3.0)


def test_mean_band_width_rejects_degenerate_weights():
    lower, upper = np.zeros((10, 2)), np.ones((10, 2))

    with pytest.raises(ValueError, match='positive'):
        mean_band_width(lower, upper, weights=[0.0, 0.0])


#-- Coverage ---------------------------------------------------------------


def test_coverage_of_a_perfect_gaussian():
    rng = np.random.default_rng(0)
    n = 200000
    y = rng.standard_normal(n)

    got = coverage(y, np.zeros(n), np.ones(n), (0.68, 0.95, 0.99))

    np.testing.assert_allclose(got, [0.68, 0.95, 0.99], atol=5e-3)


def test_coverage_detects_overconfidence():
    rng = np.random.default_rng(0)
    n = 50000
    y = rng.standard_normal(n)

    #-- claim sigma is half its true value
    got = coverage(y, np.zeros(n), np.full(n, 0.5), (0.68, 0.95))

    assert np.all(got < [0.68, 0.95])


def test_coverage_shape_for_multiple_outputs():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((5000, 3))

    got = coverage(y, np.zeros((5000, 3)), np.ones((5000, 3)), (0.68, 0.95))

    assert got.shape == (2, 3)
    np.testing.assert_allclose(got[0], 0.68, atol=0.02)
    np.testing.assert_allclose(got[1], 0.95, atol=0.02)


#-- Cross-validation -------------------------------------------------------


def test_kfold_r2_on_a_smooth_target():
    X = np.linspace(0.0, 4.0, 60).reshape(-1, 1)

    scores = kfold_r2(X, smooth(X))

    assert scores.shape == (6,)
    assert scores.mean() > 0.9, 'a smooth target should cross-validate well'


def test_kfold_r2_shape_for_multiple_outputs():
    X = np.linspace(0.0, 6.0, 60).reshape(-1, 1)

    scores = kfold_r2(X, three_outputs(X))

    assert scores.shape == (6, 3)


def test_kfold_r2_on_multidimensional_inputs():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(80, 2))
    y = X[:, 0] + 0.5 * X[:, 1]

    scores = kfold_r2(X, y)

    assert scores.mean() > 0.8


def test_kfold_r2_is_poor_on_noise():
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 60).reshape(-1, 1)
    y = rng.standard_normal(60)

    assert kfold_r2(X, y).mean() < 0.5


#-- Reports ----------------------------------------------------------------


def test_fit_report_single_output():
    X = np.linspace(0.5, 4.0, 50).reshape(-1, 1)

    report = fit_report(X, smooth(X), verbose=False)

    assert np.isscalar(report['mse']) or report['mse'].size == 1
    assert report['coverage'].shape == (3,)
    assert report['kfold_r2'].shape == (6,)
    assert report['mean_r2'] > 0.9


def test_fit_report_multi_output():
    X = np.linspace(0.5, 6.0, 50).reshape(-1, 1)

    report = fit_report(X, three_outputs(X), verbose=False)

    assert report['mse'].shape == (3,)
    assert report['coverage'].shape == (3, 3)
    assert report['kfold_r2'].shape == (6, 3)


def test_fit_report_log_x():
    X = np.linspace(1.0, 30.0, 40).reshape(-1, 1)
    y = np.log(X[:, 0])

    report = fit_report(X, y, log_x=True, verbose=False)

    assert report['mean_r2'] > 0.9


#-- Through the class ------------------------------------------------------


def test_class_metrics_single_output(tmp_path):
    X = np.linspace(0.0, 4.0, 40).reshape(-1, 1)

    ast = Astral(test_name='t', runs_dir=tmp_path, seed=0, n_test=300,
                 n_sigma=1.96, verbose=False)
    ast.set_data(X, smooth(X))

    assert ast.kfold_r2().shape == (6,)

    cov = ast.coverage()
    assert cov.shape == (3,)
    assert np.all((cov >= 0.0) & (cov <= 1.0))


def test_class_metrics_multi_output(tmp_path):
    X = np.linspace(0.0, 6.0, 40).reshape(-1, 1)

    ast = Astral(test_name='t', runs_dir=tmp_path, seed=0, n_test=300,
                 n_sigma=1.96, verbose=False)
    ast.set_data(X, three_outputs(X))

    assert ast.kfold_r2().shape == (6, 3)
    assert ast.coverage().shape == (3, 3)


def test_uncertainty_is_reported_in_whitened_units(tmp_path):
    """Rescaling an output must not change the weighted progress number."""
    X = np.linspace(0.0, 6.0, 30).reshape(-1, 1)
    y = smooth(X)

    plain = Astral(test_name='a', runs_dir=tmp_path, seed=3, n_test=400,
                   verbose=False)
    plain.set_data(X, y)

    scaled = Astral(test_name='b', runs_dir=tmp_path, seed=3, n_test=400,
                    verbose=False)
    scaled.set_data(X, 1000.0 * y)

    assert plain.mean_uncertainty() == pytest.approx(scaled.mean_uncertainty(),
                                                     rel=1e-6)


def test_fixed_evaluation_set_is_stable_across_steps(tmp_path):
    """n_eval pins the evaluation points so the curve is comparable."""
    ast = Astral(test_name='t', runs_dir=tmp_path, seed=0, n_test=200,
                 n_eval=500, verbose=False)
    ast.build_initial_design([(0.0, 6.0)], 10, simulator=smooth)

    ast.train_gp()
    ast.mean_uncertainty()
    first = ast._eval_X.copy()

    ast.train_gp()
    ast.mean_uncertainty()

    np.testing.assert_array_equal(first, ast._eval_X)
