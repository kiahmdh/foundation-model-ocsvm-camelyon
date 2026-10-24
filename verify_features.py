#!/usr/bin/env python
"""
Check a feature directory before trusting it. Run this after every extraction.

Catches the failure modes that are silent otherwise: rows never written (a
crashed run that still left a full-size memmap), unreadable patches, features
that are not unit-norm, dead dimensions, and duplicate vectors from a decode
that returned the same blank image over and over.

    python verify_features.py --dir features_plip/bank
    python verify_features.py --dir features_plip/bank --manifest /home/user01/dn2_gigapath/manifests/bank.csv
"""
import argparse
import csv
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--manifest', default='', help='cross-check row count and labels')
    ap.add_argument('--sample', type=int, default=20000, help='rows to sample for stats')
    args = ap.parse_args()

    f = np.load(os.path.join(args.dir, 'features.npy'), mmap_mode='r')
    ok = np.load(os.path.join(args.dir, 'ok.npy'))
    N, D = f.shape
    print(f'{args.dir}\n  shape {N:,} x {D}   dtype {f.dtype}')

    prog = os.path.join(args.dir, 'progress.json')
    if os.path.exists(prog):
        # this script writes 'rows_done'; the DN2 extractor writes 'n_done'
        try:
            d = json.load(open(prog))
            r = next((d[k] for k in ('rows_done', 'n_done') if k in d), None)
        except Exception:
            r = None
        if r is None:
            print('  progress.json present but has no row count -- skipping that check')
        else:
            print(f'  progress.json rows done {r:,}'
                  + ('  <-- INCOMPLETE' if r < N else '  (complete)'))

    n_bad = int((ok == 0).sum())
    print(f'  unreadable {n_bad:,} ({100 * n_bad / N:.3f}%)')

    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(N, min(args.sample, N), replace=False))
    X = np.asarray(f[idx], dtype=np.float32)
    good = ok[idx] == 1
    Xg = X[good]

    zero_rows = int((np.abs(Xg).sum(1) == 0).sum())
    norms = np.linalg.norm(Xg, axis=1)
    print(f'  sampled {len(Xg):,} readable rows')
    print(f'  all-zero rows {zero_rows}' + ('  <-- rows never written' if zero_rows else ''))
    print(f'  L2 norm  min {norms.min():.4f}  mean {norms.mean():.4f}  max {norms.max():.4f}'
          + ('  (expect ~1.0)' if abs(norms.mean() - 1) > 0.01 else '  ok'))
    dead = int((Xg.std(0) < 1e-6).sum())
    print(f'  near-constant dims {dead} / {D}' + ('  <-- suspicious' if dead > D * 0.1 else ''))

    n_chk = min(5000, len(Xg))
    uniq = len(np.unique(np.round(Xg[:n_chk], 4), axis=0))
    print(f'  distinct vectors in first {n_chk:,} sampled: {uniq:,}'
          + ('  <-- many duplicates' if uniq < 0.9 * n_chk else '  ok'))

    if args.manifest:
        rows = list(csv.DictReader(open(args.manifest)))
        print(f'  manifest rows {len(rows):,}' +
              ('  MATCH' if len(rows) == N else f'  <-- MISMATCH vs {N:,}'))
        lab = np.array([int(r['label']) for r in rows])
        print(f'  labels: {int((lab == 0).sum()):,} normal, {int((lab == 1).sum()):,} tumour '
              f'({100 * lab.mean():.2f}% prevalence)')
        print(f'  slides: {len(set(r["slide"] for r in rows))}')


if __name__ == '__main__':
    main()