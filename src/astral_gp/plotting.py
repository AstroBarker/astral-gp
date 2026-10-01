"""Diagnostic plots for a running :class:`~astral_gp.astral.Astral`.

Three views, written into the run directory: the fit with its posterior band
(one input feature), a one-feature slice through a higher-dimensional fit, and
the mean posterior uncertainty against the number of training points.

These are diagnostics, not publication figures.  Every function accepts an
``ax`` and returns it instead of writing a file, so they compose into whatever
layout you actually want.
"""

import matplotlib.pyplot as plt
import numpy as np

#-- Okabe-Ito, colour-blind safe
MEAN_COLOR = '#0072B2'
BAND_COLOR = '#56B4E9'
SAMPLE_COLOR = '#333333'
ADDED_COLOR = '#D55E00'


def _sorted_band(ast, output):
    """Candidate abscissa and band for one output, sorted for line drawing.

    Candidates are drawn at random, so they arrive unordered; a line plot needs
    them in order.
    """
    mean, lower, upper = ast.band_physical()
    x = ast.test_X_physical[:, 0]
    order = np.argsort(x)

    return (x[order], mean[order, output], lower[order, output],
            upper[order, output])


def plot_fit(ast, xlabel='x', ylabel='y', output=0, n_initial=None,
             filename=None, dpi=300, ax=None):
    """Posterior mean, band and training points for a one-feature fit.

    Args:
        ast: a fitted Astral instance with ``n_features == 1``
        output: which output column to draw
        n_initial: if given, points beyond this index are drawn as 'added'
        ax: draw into an existing axes instead of a new figure

    Returns:
        the output path, or the axes when ``ax`` was supplied
    """
    if ast.n_features != 1:
        raise ValueError(
            f'plot_fit needs one input feature, got {ast.n_features}. '
            'Use plot_slice() for higher-dimensional fits.')

    x, mean, lower, upper = _sorted_band(ast, output)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(x, lower, upper, alpha=0.3, color=BAND_COLOR,
                    label=f'{ast.n_sigma:g}$\\sigma$ confidence')
    ax.plot(x, mean, color=MEAN_COLOR, lw=2, label='GP mean')

    _scatter_training(ast, ax, output, n_initial, feature=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if not own_figure:
        return ax

    fig.tight_layout()
    filename = filename or f'fit_step_{ast.iteration}.png'
    path = ast.output_dir / filename
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path


def plot_slice(ast, feature=0, at=None, n_line=400, xlabel=None, ylabel='y',
               output=0, n_initial=None, filename=None, dpi=300, ax=None):
    """One feature varied, the others held fixed.

    The only honest 2-D view of a higher-dimensional surrogate: the band shown
    is the posterior along a line through input space, not a projection of the
    whole fit. Training points are scattered against the same feature, so they
    generally do not lie on the line.

    Args:
        feature: index of the feature to vary
        at: values to hold the other features at; defaults to their medians
        n_line: points along the slice

    Returns:
        the output path, or the axes when ``ax`` was supplied
    """
    held = np.median(ast.X, axis=0) if at is None else np.asarray(at, float)

    lo, hi = ast.X[:, feature].min(), ast.X[:, feature].max()
    line = np.linspace(lo, hi, n_line)
    query = np.tile(held, (n_line, 1))
    query[:, feature] = line

    whitened = (query - ast.x_mean) / ast.x_std
    mean, lower, upper = ast.gp.predict_physical(whitened, n_sigma=ast.n_sigma)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(line, lower[:, output], upper[:, output], alpha=0.3,
                    color=BAND_COLOR,
                    label=f'{ast.n_sigma:g}$\\sigma$ confidence')
    ax.plot(line, mean[:, output], color=MEAN_COLOR, lw=2, label='GP mean')

    _scatter_training(ast, ax, output, n_initial, feature=feature)

    ax.set_xlabel(xlabel or f'feature {feature}')
    ax.set_ylabel(ylabel)
    held_text = ', '.join(f'{v:.3g}' for i, v in enumerate(held)
                          if i != feature)
    ax.set_title(f'slice along feature {feature}  (others at {held_text})',
                 fontsize=10)
    ax.legend(fontsize=9)

    if not own_figure:
        return ax

    fig.tight_layout()
    filename = filename or f'slice_f{feature}_step_{ast.iteration}.png'
    path = ast.output_dir / filename
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path


def _scatter_training(ast, ax, output, n_initial, feature):
    """Training points, split into the initial design and what was placed."""
    x = ast.X[:, feature]
    y = ast.y[:, output]

    if n_initial is None or n_initial >= len(x):
        ax.scatter(x, y, color=SAMPLE_COLOR, edgecolor='k', s=30, zorder=3,
                   alpha=0.6, label='Training data')
        return

    ax.scatter(x[:n_initial], y[:n_initial], color=SAMPLE_COLOR, edgecolor='k',
               s=30, zorder=3, alpha=0.6, label='Initial design')
    ax.scatter(x[n_initial:], y[n_initial:], color=ADDED_COLOR, edgecolor='k',
               s=40, zorder=4, label='Placed at max variance')


def plot_uncertainty_curve(ast, relative=True, baseline=None,
                           filename='uncertainty-vs-n.png', dpi=300, ax=None):
    """Mean posterior uncertainty against training set size.

    Args:
        relative: plot as a percentage of the first recorded value
        baseline: optional comparison value drawn as a horizontal line

    Returns:
        the output path, or the axes when ``ax`` was supplied
    """
    n = np.asarray(ast.history['n'])
    u = np.asarray(ast.history['uncertainty'], dtype=float)

    if len(u) == 0:
        raise RuntimeError('no history; run step() or run_loop() first')

    scale = 100.0 / u[0] if relative else 1.0
    ylabel = ('Relative uncertainty (%)' if relative
              else 'Mean posterior band width')

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(n, u * scale, color=MEAN_COLOR)
    ax.scatter(n, u * scale, color=MEAN_COLOR, s=18)

    if relative:
        ax.axhline(y=100.0, color='darkgray', ls='-.', lw=1.5,
                   label='Initial design')
    if baseline is not None:
        ax.axhline(y=baseline * scale, color='lightgray', ls='--', lw=1.5,
                   label='Baseline')

    ax.set_xlabel('Number of training points $N$')
    ax.set_ylabel(ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best')

    if not own_figure:
        return ax

    fig.tight_layout()
    path = ast.output_dir / filename
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
