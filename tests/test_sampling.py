"""Candidate sampling: the four accepted forms, and reproducibility."""

import numpy as np
import pytest
from scipy import stats

from astral_gp import Astral, bounds_from_data, build_sampler, uniform_over


def rng(seed=0):
    return np.random.default_rng(seed)


def test_default_sampler_covers_the_data_box():
    X = np.array([[9.0, 0.0], [30.0, 1.0], [20.0, 0.4]])

    sample = build_sampler(None, X=X)
    drawn = sample(4000, rng())

    assert drawn.shape == (4000, 2)
    assert drawn.min(axis=0) == pytest.approx([9.0, 0.0], abs=0.2)
    assert drawn.max(axis=0) == pytest.approx([30.0, 1.0], abs=0.2)


def test_explicit_bounds_override_the_data():
    X = np.array([[0.0], [1.0]])

    drawn = build_sampler(None, bounds=[(100.0, 200.0)], X=X)(500, rng())

    assert drawn.min() >= 100.0 and drawn.max() <= 200.0


def test_n_test_is_honoured():
    sample = build_sampler(None, bounds=[(0.0, 1.0)])

    for n in (1, 17, 2500):
        assert sample(n, rng()).shape == (n, 1)


def test_frozen_scipy_distribution():
    dist = stats.norm(loc=[5.0, -2.0], scale=[1.0, 0.5])

    drawn = build_sampler(dist, n_features=2)(20000, rng())

    assert drawn.shape == (20000, 2)
    assert drawn.mean(axis=0) == pytest.approx([5.0, -2.0], abs=0.05)
    assert drawn.std(axis=0) == pytest.approx([1.0, 0.5], abs=0.05)


def test_multivariate_frozen_distribution():
    dist = stats.multivariate_normal(mean=[0.0, 10.0],
                                     cov=[[1.0, 0.0], [0.0, 4.0]])

    drawn = build_sampler(dist, n_features=2)(5000, rng())

    assert drawn.shape == (5000, 2)
    assert drawn.mean(axis=0) == pytest.approx([0.0, 10.0], abs=0.1)


def test_one_distribution_per_feature():
    dists = [stats.uniform(0.0, 1.0), stats.norm(100.0, 1.0)]

    drawn = build_sampler(dists, n_features=2)(4000, rng())

    assert drawn.shape == (4000, 2)
    assert 0.0 <= drawn[:, 0].min() and drawn[:, 0].max() <= 1.0
    assert drawn[:, 1].mean() == pytest.approx(100.0, abs=0.1)


def test_plain_callable():
    def fixed(n, generator):
        return np.tile([1.0, 2.0, 3.0], (n, 1))

    drawn = build_sampler(fixed, n_features=3)(10, rng())

    assert drawn.shape == (10, 3)
    assert np.all(drawn == [1.0, 2.0, 3.0])


def test_wrong_width_is_rejected():
    def too_narrow(n, generator):
        return np.zeros((n, 2))

    with pytest.raises(ValueError, match='features'):
        build_sampler(too_narrow, n_features=5)(3, rng())


def test_unusable_sampler_type_is_rejected():
    with pytest.raises(TypeError, match='sampler must be'):
        build_sampler('linspace', n_features=1)


def test_same_seed_gives_the_same_draw():
    sample = build_sampler(None, bounds=[(0.0, 1.0), (0.0, 1.0)])

    np.testing.assert_array_equal(sample(100, rng(4)), sample(100, rng(4)))
    assert not np.array_equal(sample(100, rng(4)), sample(100, rng(5)))


def test_bounds_from_data_pads_by_margin():
    X = np.array([[0.0], [10.0]])

    assert bounds_from_data(X) == [(0.0, 10.0)]
    assert bounds_from_data(X, margin=0.1) == [(-1.0, 11.0)]


def test_bounds_from_data_survives_a_constant_feature():
    X = np.array([[5.0, 1.0], [5.0, 2.0]])

    lo, hi = bounds_from_data(X, margin=0.5)[0]

    assert np.isfinite(lo) and np.isfinite(hi)


def test_uniform_over_matches_requested_box():
    drawn = uniform_over([(2.0, 3.0)]).rvs(size=(5000, 1), random_state=rng())

    assert drawn.min() >= 2.0 and drawn.max() <= 3.0


#-- Sampling as the Astral class uses it -----------------------------------


def target(X):
    return np.sin(X[:, 0])


def test_candidates_are_redrawn_each_fit(tmp_path):
    ast = Astral(test_name='t', runs_dir=tmp_path, seed=0, n_test=200,
                 verbose=False)
    ast.set_data(np.linspace(0, 10, 12), target(np.linspace(0, 10, 12).reshape(-1, 1)))

    ast.train_gp()
    first = ast.test_X_physical.copy()
    ast.train_gp()

    assert not np.array_equal(first, ast.test_X_physical)


def test_campaign_is_reproducible_under_a_seed(tmp_path):
    def run(seed):
        ast = Astral(test_name='t', runs_dir=tmp_path, seed=seed, n_test=200,
                     verbose=False)
        ast.build_initial_design([(0.0, 10.0)], 6, simulator=target)
        for _ in range(5):
            ast.step(simulator=target)
        return ast.X

    np.testing.assert_array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_sampler_reaches_the_class(tmp_path):
    ast = Astral(test_name='t', runs_dir=tmp_path, seed=0, n_test=1000,
                 sampler=stats.norm(loc=[5.0], scale=[0.5]), verbose=False)
    grid = np.linspace(0, 10, 12)
    ast.set_data(grid, target(grid.reshape(-1, 1)))
    ast.train_gp()

    assert ast.test_X_physical.mean() == pytest.approx(5.0, abs=0.1)


def test_bounds_reach_the_class(tmp_path):
    ast = Astral(test_name='t', runs_dir=tmp_path, seed=0, n_test=500,
                 bounds=[(20.0, 30.0)], verbose=False)
    grid = np.linspace(0, 10, 12)
    ast.set_data(grid, target(grid.reshape(-1, 1)))
    ast.train_gp()

    assert ast.test_X_physical.min() >= 20.0
    assert ast.place_next()[0][0] >= 20.0
