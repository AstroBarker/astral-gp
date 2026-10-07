# astral-gp

**Variance-driven placement of training data for Gaussian process surrogates.**

When every training point costs a full simulation, an experiment, or a day of
compute, the question is not *how do I fit this data* but *where do I spend the
next evaluation*. `astral-gp` answers it the simplest way that works: fit a
Gaussian process to what you have, evaluate its posterior variance at a set of
random candidate points, and sample at the argmax. Repeat.

Inputs are `(n_samples, n_features)` for any number of features. Outputs are
`(n_samples, n_outputs)` for any number of targets, each getting its own
independent GP, and placement maximises a **weighted** sum of their variances —
so you can say which targets you actually care about.

---

## Contents

- [Install](#install)
- [Trivial example](#trivial-example)
- [How it works](#how-it-works)
- [Multiple inputs and outputs](#multiple-inputs-and-outputs)
- [Choosing where candidates are drawn](#choosing-where-candidates-are-drawn)
- [Getting new target values](#getting-new-target-values)
- [Evaluating the surrogate](#evaluating-the-surrogate)
- [Command line](#command-line)
- [API reference](#api-reference)
- [Design notes](#design-notes)
- [Examples](#examples)
- [Testing](#testing)

---

## Install

```bash
pip install astral-gp
```

From a checkout, with the example notebooks and test dependencies:

```bash
pip install -e ".[examples,dev]"
```

Requires Python 3.10+. The only runtime dependencies are numpy, scipy,
scikit-learn and matplotlib.

---

## Trivial example

Refine a 1-D target from eight starting points, letting the GP choose where the
next twelve samples go:

```python
import numpy as np
from astral_gp import Astral

def target(X):
    return np.sin(X[:, 0]) + 0.3 * X[:, 0]

ast = Astral(test_name='demo', seed=0, verbose=False)
ast.build_initial_design([(0.0, 10.0)], 8, simulator=target)

for _ in range(12):
    ast.train_gp()
    print(f'N={len(ast.X):3d}  '
          f'uncertainty={ast.mean_uncertainty():.6f}  '
          f'placing at {ast.place_next()[0][0]:.4f}')
    ast.step(simulator=target)
```

```
   N   mean band width   placed at
   8          0.174508      9.9950
   9          0.170302      4.2255
  10          0.096421      1.3018
  11          0.084023      0.0011
  12          0.083200      4.9677
  13          0.081157      9.9974
  14          0.080694      3.2172
  15          0.079076      1.7978
  16          0.077925      9.2006
  17          0.076998      0.8342
  18          0.076326      0.0003
  19          0.076097      7.8206
  20          0.075251
```

The first placements go to the **domain edges** — 9.995, then 0.0011 — where the
GP has data on one side only and its variance is therefore largest. After that it
fills the widest interior gaps. Uncertainty falls by more than half, though not
perfectly monotonically: the kernel hyperparameters are refit at every step, and
the candidates are redrawn, so a little wobble is expected.

The same thing from a terminal:

```bash
astral-gp demo --plot
```

---

## How it works

Each step does five things.

1. **Whiten.** Every input feature and every output column is standardised
   independently, using the current data. Constants are recomputed each step.
2. **Fit.** One GP per output, each with the kernel

   ```
   WhiteKernel(0.1, (1e-3, 10)) + Constant(0.1, (1e-3, 10)) * RBF(l, (1e-2, 1e2))
   ```

3. **Draw candidates.** `n_test` random points from the sampler — by default
   uniform over the bounding box of the data seen so far.
4. **Score and place.** `score(x) = Σ wⱼ · varⱼ(x)` across outputs; take the
   top `num_new_samples` candidates. Posterior variance does not depend on the
   observed `y` at all, so placement is a pure function of *where* the training
   points sit.
5. **Evaluate and append.** Obtain `y` at the new location and append it.

Progress is tracked by `mean_uncertainty()`, the mean posterior band width over
the candidates. It is computed on whitened outputs, so a single number stays
meaningful when outputs carry different units.

---

## Multiple inputs and outputs

```python
X = np.column_stack([temperature, pressure, density])   # (n, 3)
y = np.column_stack([yield_fraction, peak_flux])        # (n, 2)

ast = Astral(n_test=5000, weights=[3.0, 1.0], seed=0)
ast.set_data(X, y)

ast.place_next()          # -> (1, 3), a point in input space
ast.mean_uncertainty()                  # one weighted number
ast.mean_uncertainty(per_output=True)   # two, in their own units
```

Weights are the knob that decides what the campaign is *for*:

| `weights` | Behaviour |
|---|---|
| `None` (default) | every output counts equally |
| `[3.0, 1.0]` | the first output matters three times as much |
| `[1.0, 0.0]` | place for the first output alone |

Because each output is standardised before fitting, the per-output variances are
already commensurate, and the weights mean what they look like they mean.

---

## Choosing where candidates are drawn

The default samples uniformly over the bounding box of the data. When you know
more than that, say so with a distribution — anything exposing `rvs` works, so
the whole of `scipy.stats` is available:

```python
from scipy import stats

Astral(bounds=[(9.0, 30.0), (0.0, 1.0)])          # explicit box, still uniform
Astral(sampler=stats.norm(loc=[20.0], scale=3.0))  # concentrate where it matters
Astral(sampler=stats.truncnorm(-2, 2, loc=20.0, scale=3.0))
Astral(sampler=stats.multivariate_normal(mean=[0, 10], cov=np.eye(2)))

Astral(sampler=[stats.uniform(9, 21), stats.beta(2, 5)])   # one per feature
Astral(sampler=lambda n, rng: my_latin_hypercube(n, rng))  # anything else
```

`n_test` sets how many candidates are drawn per step, and `seed` makes the whole
campaign reproducible. Candidates are redrawn every step; if you want the
progress curve measured on a *fixed* set instead, pass `n_eval`.

---

## Getting new target values

Three ways, depending on how your evaluations are actually run.

**A callable**, when the target is cheap enough to evaluate inline. It receives
an `(m, d)` array and returns `(m,)` or `(m, k)`:

```python
ast.step(simulator=lambda X: np.sin(X[:, 0]))
```

**A pool of finished evaluations**, when the results already exist and you are
studying how few of them you needed. The placement is computed as usual and the
nearest unused pool member is consumed:

```python
ast.set_pool(pool_X, pool_y)
ast.run_loop(50)
```

**Place and hand back**, when an evaluation means submitting a job:

```python
X_new = ast.place_next()          # where to run next
y_new = my_expensive_code(X_new)  # ... hours later ...
ast.add_point(X_new, y_new)
```

Data round-trips through JSON, so a campaign can be stopped and resumed:

```python
ast.save_data()                          # runs/<name>/data_at_step_<i>.json
ast.load_data(7, data_dir='my-runs')
```

---

## Evaluating the surrogate

```python
ast.kfold_r2(num_folds=6)                # cross-validated R^2, per output
ast.coverage(alphas=(0.68, 0.95, 0.99))  # empirical vs nominal credible levels
```

Coverage is the honest check on a GP: if the posterior is calibrated, 68% of
points should fall inside the 1σ band. Since those error bars are what drives
every placement decision, systematic under-coverage means the campaign is being
steered by numbers you cannot trust.

For a one-shot report on a dataset without running a campaign:

```python
from astral_gp import fit_report

fit_report(X, y)
```

---

## Command line

```bash
astral-gp demo                                   # the example above
astral-gp place   --data data_at_step_10.json    # where should the next run go?
astral-gp inspect --data-dir runs/mine --stop 50 # uncertainty across a campaign
astral-gp step    --data-dir runs/mine --step 0 --pool finished.dat
```

```bash
$ astral-gp place --data data_at_step_10.json --seed 0
astral: astral -> runs/astral
  mean band width: 0.0801287180116
  place next at [0.0019000161]
```

---

## API reference

### `Astral(...)`

| Argument | Default | Meaning |
|---|---|---|
| `test_name` | `'astral'` | Run label; output goes to `{runs_dir}/{test_name}` |
| `n_test` | `1000` | Random candidate points drawn per step |
| `n_eval` | `None` | Fixed evaluation set for progress reporting |
| `num_new_samples` | `1` | Points placed per step |
| `weights` | `None` | Per-output placement weights; `None` means equal |
| `sampler` | `None` | Distribution, list of them, or callable |
| `bounds` | `None` | Per-feature `(lo, hi)`; `None` derives from the data |
| `seed` | `None` | Seeds candidate draws and optimizer restarts |
| `n_sigma` | `1.0` | Band half-width in posterior standard deviations |
| `length_scale_bounds` | `(1e-2, 1e2)` | On whitened inputs; `'data'` for the legacy rule |
| `n_restarts` | `2` | Optimizer restarts per fit |
| `runs_dir` | `'runs'` | Parent directory for output |

| Method | Returns |
|---|---|
| `set_data(X, y, weights=None)` | Install training data |
| `build_initial_design(bounds, n, ...)` | Seed with random points |
| `set_pool(pool_X, pool_y)` | Register finished evaluations |
| `train_gp()` | `(mean, lower, upper)`, each `(n_test, k)` |
| `band_physical()` | The same, in the original output units |
| `placement_score()` | Weighted variance at each candidate |
| `place_next(num_new_samples=None)` | `(k, d)` points of largest score |
| `add_point(X, y)` | Append to the data |
| `step(...)` | One iteration; returns `(X_new, y_new)` |
| `run_loop(n_steps, ...)` | Repeat `step`; returns the history dict |
| `mean_uncertainty(per_output=False)` | Mean posterior band width |
| `kfold_r2()`, `coverage()` | Evaluation metrics, per output |
| `save_data()`, `load_data()`, `save_history()` | JSON persistence |
| `make_plots()` | Diagnostic figures into the run directory |

`n_sigma` does **not** affect placement — the argmax is over a monotone
rescaling of the same variances — but the reported band width scales with it.

### Module functions

`whiten`, `unwhiten`, `SklearnGP`, `MultiOutputGP`, `sklearn_gpr`,
`mean_band_width`, `coverage`, `kfold_r2`, `fit_report`, `build_sampler`,
`bounds_from_data`, `uniform_over`.

---

## Design notes

**One GP per output, not one shared GP.** scikit-learn's multi-output
`GaussianProcessRegressor` fits a *single* kernel across all columns and returns
an identical posterior standard deviation for every one of them. Under that
model a weight vector would be a no-op. Fitting `k` independent GPs, each with
its own kernel and its own standardisation, is what makes per-output uncertainty
real. It costs `k` fits per step.

**Random candidates, not a grid.** A dense grid is fine in one dimension and
infeasible as dimensionality increases. Random sampling costs nothing in dimension, 
and `n_test` gives direct control over how hard you look. The price is Monte Carlo 
noise of order `1/sqrt(n_test)` on the reported uncertainty; `n_eval` removes it 
from the progress curve by fixing the evaluation set.

**Length-scale bounds are floored at `(1e-2, 1e2)`.** On whitened inputs a
length scale near zero decorrelates every candidate from every training point,
and the GP falls back to predicting its prior — the band width jumps by
an order of magnitude for one step and recovers. A fixed floor bounds the
damage. If you see a single wild spike in an otherwise smooth uncertainty curve, 
check whether the fitted length scale is sitting on its lower bound.

**Uncertainty is reported on whitened outputs.** Averaging band widths across
outputs measured in different units would be meaningless, so the single
progress number is computed after standardisation. Use
`mean_uncertainty(per_output=True)` for physical band widths.

---

## Examples

Three notebooks in [`examples/`](examples/):

1. **`01-analytic-single-output.ipynb`** — the method end to end on analytic
   targets, in one dimension and then in two, with a diagnostic section on
   spotting a degenerate fit. Runs in seconds, needs no data.
2. **`02-mesa-single-output.ipynb`** — a real astrophysical campaign against the
   STIR+SNEC supernova light-curve dataset from
   [zenodo.org/records/6631964](https://zenodo.org/records/6631964), including
   a comparison against non-adaptive designs.
3. **`03-mesa-multi-output.ipynb`** — the same dataset with all three outputs
   fit at once, showing how the campaign changes as the weights change.

The Zenodo download is commented out by default in both MESA notebooks.

---

## Testing

```bash
pytest
```

The suite needs no data files.

---
 
## Acknowledgements

The initial version of this software was written by Trevor Gravely. Subsequent and current development 
is carried out by Brandon Barker and Marko Ristić.
