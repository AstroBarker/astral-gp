"""Placement: multi-dimensional inputs, multi-output weighting, pools."""

import numpy as np
import pytest

from astral_gp import Astral


def smooth_1d(X):
    return np.sin(X[:, 0]) + 0.3 * X[:, 0]


def three_outputs(X):
    """Outputs of deliberately different roughness."""
    x = X[:, 0]

    return np.column_stack([np.sin(x),
                            np.sin(6.0 * x),
                            0.5 * x])


def build(tmp_path, **kwargs):
    kwargs.setdefault('seed', 0)
    kwargs.setdefault('verbose', False)
    kwargs.setdefault('n_test', 400)

    return Astral(test_name='t', runs_dir=tmp_path, **kwargs)


#-- Shapes -----------------------------------------------------------------


@pytest.mark.parametrize('d', [1, 2, 3])
def test_placement_shape_follows_input_dimension(d, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(25, d))
    y = X.sum(axis=1)

    ast = build(tmp_path, num_new_samples=3)
    ast.set_data(X, y)

    assert ast.place_next().shape == (3, d)


def test_one_dimensional_input_accepted_as_flat_array(tmp_path):
    ast = build(tmp_path)
    ast.set_data(np.linspace(0, 10, 15), np.sin(np.linspace(0, 10, 15)))

    assert ast.n_features == 1
    assert ast.n_outputs == 1
    assert ast.place_next().shape == (1, 1)


def test_multi_output_shapes(tmp_path):
    X = np.linspace(0, 6, 30).reshape(-1, 1)

    ast = build(tmp_path)
    ast.set_data(X, three_outputs(X))
    ast.train_gp()

    assert ast.n_outputs == 3
    assert ast.mean.shape == (400, 3)
    assert ast.gp.variances(ast.test_X).shape == (400, 3)
    assert len(ast.mean_uncertainty(per_output=True)) == 3
    assert np.isscalar(ast.mean_uncertainty())


def test_mismatched_lengths_are_rejected(tmp_path):
    ast = build(tmp_path)

    with pytest.raises(ValueError, match='rows'):
        ast.set_data(np.zeros((5, 2)), np.zeros(4))


def test_wrong_weight_count_is_rejected(tmp_path):
    X = np.linspace(0, 6, 20).reshape(-1, 1)

    ast = build(tmp_path, weights=[1.0, 1.0])

    with pytest.raises(ValueError, match='weights'):
        ast.set_data(X, three_outputs(X))


#-- Weighting --------------------------------------------------------------


def test_weights_default_to_ones(tmp_path):
    X = np.linspace(0, 6, 20).reshape(-1, 1)

    ast = build(tmp_path)
    ast.set_data(X, three_outputs(X))

    np.testing.assert_array_equal(ast.weights, np.ones(3))


def test_single_output_weight_matches_fitting_that_output_alone(tmp_path):
    """w = [1, 0, 0] must place exactly as a run on column 0 would."""
    X = np.linspace(0, 6, 24).reshape(-1, 1)
    y = three_outputs(X)

    both = build(tmp_path, weights=[1.0, 0.0, 0.0])
    both.set_data(X, y)
    both.train_gp()

    alone = build(tmp_path)
    alone.set_data(X, y[:, 0])
    alone.train_gp()

    #-- Same seed and same data, so the candidate draws coincide exactly.
    #-- The scores differ only in the last bits: one is a matmul against
    #-- [1, 0, 0], the other reads the variance straight out.
    np.testing.assert_array_equal(both.test_X_physical, alone.test_X_physical)
    np.testing.assert_allclose(both.placement_score(),
                               alone.placement_score(), rtol=1e-9)
    np.testing.assert_array_equal(both.place_next(), alone.place_next())


def test_different_weights_give_different_campaigns(tmp_path):
    X = np.linspace(0, 6, 24).reshape(-1, 1)
    y = three_outputs(X)

    scores = []
    for w in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]):
        ast = build(tmp_path, weights=w)
        ast.set_data(X, y)
        ast.train_gp()
        scores.append(ast.placement_score())

    assert not np.allclose(scores[0], scores[1])
    assert not np.allclose(scores[0], scores[2])


def test_zero_weights_are_rejected_by_the_uncertainty_metric(tmp_path):
    X = np.linspace(0, 6, 20).reshape(-1, 1)

    ast = build(tmp_path, weights=[0.0, 0.0, 0.0])
    ast.set_data(X, three_outputs(X))
    ast.train_gp()

    with pytest.raises(ValueError, match='positive'):
        ast.mean_uncertainty()


#-- Behaviour --------------------------------------------------------------


def test_uncertainty_falls_as_points_are_added(tmp_path):
    ast = build(tmp_path, n_test=600)
    ast.build_initial_design([(0.0, 5.0)], 6, simulator=smooth_1d)
    ast.run_loop(20, simulator=smooth_1d)

    u = ast.history['uncertainty']

    assert u[-1] < u[0], 'mean band width should shrink with N'


def test_uncertainty_falls_in_two_dimensions(tmp_path):
    def surface(X):
        return np.sin(3.0 * X[:, 0]) * np.cos(2.0 * X[:, 1])

    ast = build(tmp_path, n_test=800)
    ast.build_initial_design([(0.0, 1.0), (0.0, 1.0)], 12, simulator=surface)
    ast.run_loop(25, simulator=surface)

    u = ast.history['uncertainty']

    assert ast.n_features == 2
    assert u[-1] < u[0]


def test_placement_goes_where_variance_is_highest(tmp_path):
    """Leave half the domain empty; the next point must land in the hole."""
    X = np.concatenate([np.linspace(0.0, 4.0, 20),
                        np.linspace(9.0, 10.0, 5)]).reshape(-1, 1)

    ast = build(tmp_path, n_test=2000, bounds=[(0.0, 10.0)])
    ast.set_data(X, smooth_1d(X))

    assert 4.0 < ast.place_next()[0][0] < 9.0


def test_n_sigma_does_not_move_placement(tmp_path):
    X = np.linspace(0, 10, 20).reshape(-1, 1)

    places = []
    for n_sigma in (1.0, 1.96, 3.0):
        ast = build(tmp_path, n_sigma=n_sigma)
        ast.set_data(X, smooth_1d(X))
        places.append(ast.place_next())

    np.testing.assert_array_equal(places[0], places[1])
    np.testing.assert_array_equal(places[0], places[2])


def test_points_are_appended_not_sorted(tmp_path):
    X = np.linspace(0, 10, 10).reshape(-1, 1)

    ast = build(tmp_path)
    ast.set_data(X, smooth_1d(X))
    ast.add_point([[0.5]], [[1.0]])

    assert ast.X[-1, 0] == 0.5
    np.testing.assert_array_equal(ast.X[:-1], X)


#-- Pools ------------------------------------------------------------------


def test_pool_mode_consumes_each_entry_once(tmp_path):
    pool_X = np.linspace(0.0, 10.0, 60).reshape(-1, 1)
    pool_y = smooth_1d(pool_X)

    ast = build(tmp_path)
    ast.set_data(pool_X[::10], pool_y[::10])
    ast.run_loop(10, pool_X=pool_X, pool_y=pool_y)

    added = ast.X[6:]

    assert len(np.unique(added)) == len(added)
    assert ast.pool_used.sum() == 10


def test_pool_mode_works_in_two_dimensions(tmp_path):
    rng = np.random.default_rng(0)
    pool_X = rng.uniform(size=(80, 2))
    pool_y = pool_X[:, 0] ** 2 + np.sin(4.0 * pool_X[:, 1])

    ast = build(tmp_path)
    ast.set_data(pool_X[:10], pool_y[:10])
    ast.set_pool(pool_X, pool_y)
    ast.pool_used[:10] = True
    ast.run_loop(8)

    assert len(ast.X) == 18
    assert ast.pool_used.sum() == 18


def test_exhausted_pool_raises(tmp_path):
    pool_X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    pool_y = smooth_1d(pool_X)

    ast = build(tmp_path)
    ast.set_data(pool_X[:3], pool_y[:3])
    ast.set_pool(pool_X, pool_y)
    ast.pool_used[:] = True

    with pytest.raises(RuntimeError, match='exhausted'):
        ast.step()


def test_no_simulator_and_no_pool_is_a_clear_error(tmp_path):
    X = np.linspace(0, 10, 12).reshape(-1, 1)

    ast = build(tmp_path)
    ast.set_data(X, smooth_1d(X))

    with pytest.raises(ValueError, match='no simulator and no pool'):
        ast.step()


def test_history_records_request_and_delivery(tmp_path):
    pool_X = np.linspace(0.0, 10.0, 40).reshape(-1, 1)
    pool_y = smooth_1d(pool_X)

    ast = build(tmp_path)
    ast.set_data(pool_X[::8], pool_y[::8])
    ast.run_loop(4, pool_X=pool_X, pool_y=pool_y)

    assert len(ast.history['placement']) == 4
    assert len(ast.history['added']) == 4
    assert len(ast.history['kernels'][0]) == 1


#-- Persistence ------------------------------------------------------------


def test_data_roundtrip(tmp_path):
    X = np.linspace(0, 6, 12).reshape(-1, 1)
    y = three_outputs(X)

    ast = build(tmp_path)
    ast.set_data(X, y)
    ast.save_data(step=3)

    other = build(tmp_path)
    other.load_data(3)

    np.testing.assert_array_equal(other.X, X)
    np.testing.assert_array_equal(other.y, y)


def test_legacy_data_layout_still_reads(tmp_path):
    """The older {"data": [[x, y], ...]} files remain loadable."""
    import json

    matrix = [[9.0, 1.45], [10.0, 1.47], [11.0, 1.50]]
    (tmp_path / 'grid_at_step_0.json').write_text(json.dumps({'data': matrix}))

    ast = build(tmp_path)
    ast.load_data(0, data_dir=tmp_path, n_features=1)

    assert ast.X.shape == (3, 1)
    assert ast.y.shape == (3, 1)
    assert ast.X[0, 0] == 9.0 and ast.y[0, 0] == 1.45
