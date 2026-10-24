#!/usr/bin/env python
"""
BIHN-with-foundation-model -- optional driver.

Replaces the old run_bihn.sh. Nothing is hidden: it resolves every path from
the flags below, PRINTS each command it is about to run, and with --dry_run
prints them without running anything. You can always copy a printed line and
run that stage by hand -- the driver adds no behaviour of its own.

Stages
    extract   only for a backbone whose features are not on disk yet (PLIP)
    train     fit the one-class SVM on the normal bank
    score     score the query split and write results/<run>/

Examples
    # see exactly what would happen, run nothing
    python run_bihn.py --backbone gigapath --query dev --dry_run

    # GigaPath: features already exist, so extract is skipped automatically
    python run_bihn.py --backbone gigapath --query dev --nu 0.1 --n_train 40000

    # PLIP: extracts bank + dev first
    python run_bihn.py --backbone plip --query dev --nu 0.1 --n_train 40000
"""
import argparse
import json
import os
import subprocess
import sys


def count_manifest_rows(p):
    with open(p) as f:
        return sum(1 for _ in f) - 1                     # minus header


def extraction_complete(feat_dir, manifest):
    """True iff features.npy exists with exactly one row per manifest entry.
    Does NOT rely on progress.json (your DN2 dirs predate this script)."""
    fp = os.path.join(feat_dir, 'features.npy')
    if not (os.path.exists(fp) and os.path.exists(os.path.join(feat_dir, 'ok.npy'))):
        return False
    try:
        import numpy as np
        n_rows = np.load(fp, mmap_mode='r').shape[0]
    except Exception:
        return False
    return n_rows == count_manifest_rows(manifest)


def pretty(cmd):
    """Group each --flag with its value on one line, so a printed command can be
    copied straight into the shell and run by hand."""
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
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f'\nstage failed with exit code {r.returncode}: {cmd[1]}')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--backbone', choices=['gigapath', 'plip'], required=True)
    ap.add_argument('--query', default='dev', help='split to score: dev, test, c17_test')
    ap.add_argument('--bank_split', default='bank')
    ap.add_argument('--dn2_root', default='/home/user01/dn2_gigapath',
                    help='where manifests/ and the existing GigaPath features/ live')
    ap.add_argument('--feat_root', default='',
                    help='feature dirs root; default = dn2_root/features for gigapath, '
                         'else features_<backbone>')
    ap.add_argument('--nu', type=float, default=0.1)
    ap.add_argument('--gamma', default='scale')
    ap.add_argument('--n_train', type=int, default=40000)
    ap.add_argument('--method', choices=['exact', 'sgd'], default='exact')
    ap.add_argument('--run_name', default='', help='default: <query>_<backbone>_nu<nu>_n<n_train>')
    ap.add_argument('--save_n_examples', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--force_extract', action='store_true')
    ap.add_argument('--force_train', action='store_true')
    ap.add_argument('--dry_run', action='store_true', help='print the commands, run nothing')
    args = ap.parse_args()

    feat_root = args.feat_root or (os.path.join(args.dn2_root, 'features')
                                   if args.backbone == 'gigapath' else f'features_{args.backbone}')
    man = lambda s: os.path.join(args.dn2_root, 'manifests', f'{s}.csv')
    feat = lambda s: os.path.join(feat_root, s)
    py = sys.executable

    model = os.path.join('models',
                         f'ocsvm_{args.backbone}_{args.method}_nu{args.nu}_n{args.n_train}.pkl')
    run_name = args.run_name or f'{args.query}_{args.backbone}_nu{args.nu}_n{args.n_train}'

    print(f'backbone   : {args.backbone}')
    print(f'manifests  : {args.dn2_root}/manifests/')
    print(f'features   : {feat_root}/')
    print(f'model      : {model}')
    print(f'results    : results/{run_name}/')

    # ---- stage 1: extract (skipped when features are already complete) ----------
    for split in (args.bank_split, args.query):
        m = man(split)
        if not os.path.exists(m):
            sys.exit(f'missing manifest: {m}')
        if extraction_complete(feat(split), m) and not args.force_extract:
            print(f'\nskip extract: {feat(split)} already complete '
                  f'({count_manifest_rows(m):,} rows)')
            continue
        cmd = [py, 'extract_features_fm.py', '--backbone', args.backbone,
               '--manifest', m, '--out_dir', feat(split),
               '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers)]
        if args.force_extract:
            cmd.append('--overwrite')
        run(cmd, args.dry_run)

    # ---- stage 2: fit the one-class SVM ----------------------------------------
    if os.path.exists(model) and not args.force_train:
        print(f'\nskip train: {model} exists (use --force_train to refit)')
    else:
        run([py, 'ocsvm_train.py', '--bank_dir', feat(args.bank_split), '--out', model,
             '--nu', str(args.nu), '--gamma', str(args.gamma),
             '--n_train', str(args.n_train), '--method', args.method], args.dry_run)

    # ---- stage 3: score --------------------------------------------------------
    run([py, 'ocsvm_score.py', '--model', model, '--query_dir', feat(args.query),
         '--run', run_name, '--save_n_examples', str(args.save_n_examples)], args.dry_run)

    if args.dry_run:
        print('\n(dry run: nothing was executed)')


if __name__ == '__main__':
    main()
