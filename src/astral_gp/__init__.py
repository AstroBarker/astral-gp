"""astral-gp: variance-driven placement of training data for GP surrogates.

Fit a Gaussian process to what you have, find where its posterior variance is
largest, and spend the next expensive evaluation there.

Inputs are (n, d) for any number of features, outputs are (n, k) for any number
of targets, and placement maximises a weighted sum of the per-output variances.

    from astral_gp import Astral

    ast = Astral(test_name='demo', n_test=2000, seed=0)
    ast.build_initial_design([(0.0, 1.0)], 5, simulator=my_function)
    ast.run_loop(20, simulator=my_function)
"""

from .astral import Astral
from .gp import MultiOutputGP, SklearnGP, sklearn_gpr, unwhiten, whiten
from .metrics import coverage, fit_report, kfold_r2, mean_band_width
from .sampling import bounds_from_data, build_sampler, uniform_over

__version__ = '0.2.0'

__all__ = [
    'Astral',
    'SklearnGP',
    'MultiOutputGP',
    'sklearn_gpr',
    'whiten',
    'unwhiten',
    'coverage',
    'fit_report',
    'kfold_r2',
    'mean_band_width',
    'build_sampler',
    'bounds_from_data',
    'uniform_over',
    '__version__',
]
