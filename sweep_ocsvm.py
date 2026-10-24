#!/usr/bin/env python
"""
Sweep OC-SVM hyperparameters on the DEV split. Same style as run_bihn.py:
every command is printed before it runs; --dry_run prints only.

The reference implementation hardcodes  OneClassSVM(nu=0.1, kernel='rbf')
with sklearn's default gamma='scale' and never sweeps, so there is no
author-recommended grid beyond that one point. What is worth sweeping:

    nu        fraction of bank treated as outliers = the fixed threshold's FPR
    gamma     RBF width. Features are L2-normalised in 1536-d, so 'scale'
              (= 1 / (n_features * var)) is ~1. Try 0.3x / 3x around it.
    n_train   subsample size. exact fit is ~quadratic: 40k ~ 3 min,
              100k ~ 20 min, 200k ~ 1.5 h on 12 cores.

Runs are skipped if their results/<run>/metrics.txt already exists, so the
sweep can be interrupted and resumed. Ends by printing summary.tsv sorted by
patch AUROC.

Examples
    python sweep_ocsvm.py --backbone gigapath --dry_run
    python sweep_ocsvm.py --backbone gigapath --nu 0.02 0.05 0.1 0.2 --gamma scale 0.3 3
    python sweep_ocsvm.py --backbone gigapath --nu 0.1 --gamma scale --n_train 100000 200000
"""
import argparse
import os
import subprocess
import sys


def pretty(cmd):
    lines, i = [os.path.basename(cmd[0]) + ' ' + cmd[1]], 2
    while i < len(cmd):
        if cmd[i].startswith('--') and i + 1 < len(cmd) and not cmd[i + 1].startswith('--'):
            lines.append(f'{cmd[i]} {cmd[i + 1]}'); i += 2
        else:
            lines.append(cmd[i]); i += 1
    return ' \\\n    '.join(lines)


def run(cmd, dry):
    print('\n$ ' + pretty(cmd), flush=True)
    if dry:
        return
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f'failed: {cmd[1]}')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--backbone', choices=['gigapath', 'plip'], required=True)
    ap.add_argument('--query', default='dev', help="tune on dev; only score 'test' once, at the end")
    ap.add_argument('--dn2_root', default='/home/user01/dn2_gigapath')
    ap.add_argument('--feat_root', default='')
    ap.add_argument('--nu', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument('--gamma', nargs='+', default=['scale'])
    ap.add_argument('--n_train', type=int, nargs='+', default=[40000])
    ap.add_argument('--method', choices=['exact', 'sgd'], default='exact')
    ap.add_argument('--standardize', action='store_true')
    ap.add_argument('--save_n_examples', type=int, default=0, help='montages off during sweeps')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    feat_root = args.feat_root or (os.path.join(args.dn2_root, 'features')
                                   if args.backbone == 'gigapath' else f'features_{args.backbone}')
    bank, query = os.path.join(feat_root, 'bank'), os.path.join(feat_root, args.query)
    py = sys.executable
    std = '_std' if args.standardize else ''

    grid = [(nu, g, n) for n in args.n_train for g in args.gamma for nu in args.nu]
    print(f'{len(grid)} configurations on {query}')
    for nu, g, n in grid:
        tag = f'{args.backbone}_{args.method}_nu{nu}_g{g}_n{n}{std}'
        model = os.path.join('models', f'ocsvm_{tag}.pkl')
        run_name = f'{args.query}_{tag}'
        if os.path.exists(os.path.join('results', run_name, 'metrics.txt')):
            print(f'\nskip (done): {run_name}')
            continue
        if not os.path.exists(model):
            cmd = [py, 'ocsvm_train.py', '--bank_dir', bank, '--out', model,
                   '--nu', str(nu), '--gamma', str(g), '--n_train', str(n), '--method', args.method]
            if args.standardize:
                cmd.append('--standardize')
            run(cmd, args.dry_run)
        run([py, 'ocsvm_score.py', '--model', model, '--query_dir', query,
             '--run', run_name, '--save_n_examples', str(args.save_n_examples)], args.dry_run)

    summ = os.path.join('results', 'summary.tsv')
    if not args.dry_run and os.path.exists(summ):
        rows = [l.rstrip('\n').split('\t') for l in open(summ)]
        head, body = rows[0], [r for r in rows[1:] if r[0].startswith(args.query + '_')]
        body.sort(key=lambda r: -float(r[1]) if r[1] != 'nan' else 0)
        print(f'\n{args.query} runs sorted by patch AUROC (%):')
        print('\t'.join(head))
        for r in body:
            print('\t'.join(r))


if __name__ == '__main__':
    main()
