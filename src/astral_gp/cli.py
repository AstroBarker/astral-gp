"""Command line interface.

    astral-gp demo
    astral-gp place   --data data_at_step_0.json
    astral-gp inspect --data-dir runs/mine --stop 50
    astral-gp step    --data-dir runs/mine --step 0 --pool finished.dat

Data files are the ``{"X": [[...]], "y": [[...]]}`` JSON the package writes; the
older ``{"data": [[...]]}`` layout is also accepted.  Pool files are
whitespace-delimited text, with ``--input-cols`` and ``--output-cols`` selecting
which columns are which.
"""

import argparse
import json

import numpy as np

from .astral import Astral


def load_pool(path, input_cols=(0,), output_cols=(1,)):
    """Read a whitespace-delimited table of finished evaluations."""
    data = np.loadtxt(path)

    return data[:, list(input_cols)], data[:, list(output_cols)]


def _read_data(path, n_features=1):
    """Read a data JSON into (X, y)."""
    with open(path, 'r') as f:
        payload = json.load(f)

    if 'X' in payload:
        return np.array(payload['X']), np.array(payload['y'])

    matrix = np.array(payload['data'])

    return matrix[:, :n_features], matrix[:, n_features:]


def _build(args, **extra):
    return Astral(test_name=args.test_name, n_test=args.n_test,
                  n_sigma=args.n_sigma, seed=args.seed,
                  weights=args.weights, bounds=args.bounds,
                  runs_dir=args.runs_dir, **extra)


def cmd_place(args):
    """Report where the next evaluation(s) should go."""
    ast = _build(args, num_new_samples=args.num_new_samples)
    ast.set_data(*_read_data(args.data, args.n_features))
    ast.train_gp()

    print(f'  mean band width: {ast.mean_uncertainty():.12g}')
    for row in ast.place_next():
        print(f'  place next at {np.array2string(row, precision=10)}')


def cmd_inspect(args):
    """Report the uncertainty across a saved sequence."""
    ast = _build(args)
    ast.verbose = False

    print(f"{'step':>5} {'N':>6} {'uncertainty':>16}  next")
    for step in range(args.start, args.stop):
        ast.load_data(step, data_dir=args.data_dir,
                      n_features=args.n_features)
        ast.train_gp()
        uncertainty = ast.mean_uncertainty()
        placement = ast.place_next()[0]

        ast.history['iteration'].append(step)
        ast.history['n'].append(len(ast.X))
        ast.history['uncertainty'].append(uncertainty)
        ast.history['placement'].append(placement.tolist())
        ast.history['kernels'].append([str(k) for k in ast.gp.kernels_])

        print(f'{step:>5} {len(ast.X):>6} {uncertainty:>16.12g}  '
              f'{np.array2string(placement, precision=6)}')

    u = ast.history['uncertainty']
    if len(u) > 1:
        print(f'\nuncertainty {u[0]:.6g} -> {u[-1]:.6g}  '
              f'({100.0 * u[-1] / u[0]:.4f}% of initial)')

    print(f'wrote {ast.save_history()}')
    if args.plot:
        ast.make_plots()


def cmd_step(args):
    """Advance stored data by one variance-driven placement."""
    ast = _build(args, num_new_samples=args.num_new_samples)
    ast.load_data(args.step, data_dir=args.data_dir,
                  n_features=args.n_features)
    ast.set_pool(*load_pool(args.pool, args.input_cols, args.output_cols))
    ast.step()

    print(f'wrote {ast.save_data(data_dir=args.data_dir)}')


def cmd_demo(args):
    """Run the README example: refine a 1-D analytic target."""
    def target(X):
        return np.sin(X[:, 0]) + 0.3 * X[:, 0]

    ast = _build(args)
    ast.verbose = False
    ast.build_initial_design([(0.0, 10.0)], 8, simulator=target)

    print(f"{'N':>4}  {'mean band width':>18}  {'placed at':>10}")
    for _ in range(args.n_steps):
        ast.train_gp()
        print(f'{len(ast.X):>4}  {ast.mean_uncertainty():>18.6f}'
              f'  {ast.place_next()[0][0]:>10.4f}')
        ast.step(simulator=target)

    ast.train_gp()
    print(f'{len(ast.X):>4}  {ast.mean_uncertainty():>18.6f}')

    if args.plot:
        ast.make_plots(xlabel='x', ylabel='f(x)')


def _add_common(parser):
    parser.add_argument('--test-name', default='astral')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--n-test', type=int, default=1000,
                        help='number of random candidate points per step')
    parser.add_argument('--n-sigma', type=float, default=1.0,
                        help='band half-width in posterior standard deviations')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for the candidate draws')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='per-output placement weights (default: equal)')
    parser.add_argument('--bounds', type=float, nargs='+', default=None,
                        metavar='LO HI',
                        help='sampling box as LO HI pairs, one per feature')
    parser.add_argument('--n-features', type=int, default=1,
                        help='input columns when reading legacy data files')

    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog='astral-gp',
        description='Variance-driven placement of GP training data.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    place = subparsers.add_parser(
        'place', help='report where the next evaluation should go')
    place.add_argument('--data', required=True, help='data_at_step_N.json')
    place.add_argument('--num-new-samples', type=int, default=1)
    _add_common(place)
    place.set_defaults(func=cmd_place)

    inspect = subparsers.add_parser(
        'inspect', help='uncertainty across a saved sequence')
    inspect.add_argument('--data-dir', required=True)
    inspect.add_argument('--start', type=int, default=0)
    inspect.add_argument('--stop', type=int, required=True)
    inspect.add_argument('--plot', action='store_true')
    _add_common(inspect)
    inspect.set_defaults(func=cmd_inspect)

    step = subparsers.add_parser(
        'step', help='advance stored data by one placement')
    step.add_argument('--data-dir', required=True)
    step.add_argument('--step', type=int, required=True)
    step.add_argument('--pool', required=True,
                      help='text file of finished evaluations')
    step.add_argument('--input-cols', type=int, nargs='+', default=[0])
    step.add_argument('--output-cols', type=int, nargs='+', default=[1])
    step.add_argument('--num-new-samples', type=int, default=1)
    _add_common(step)
    step.set_defaults(func=cmd_step)

    demo = subparsers.add_parser('demo', help='run the README example')
    demo.add_argument('--n-steps', type=int, default=12)
    demo.add_argument('--plot', action='store_true')
    _add_common(demo)
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.bounds is not None:
        flat = args.bounds
        if len(flat) % 2:
            raise SystemExit('--bounds needs an even number of values')
        args.bounds = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    args.func(args)


if __name__ == '__main__':
    main()
