#!/usr/bin/env python
"""
BIHN stage 2 -- fit the one-class classifier on NORMAL patch features.

Mirrors anomaly_detector.py::anomaly_training() in the BI repo:
    clf = svm.OneClassSVM(nu=0.1, kernel='rbf'); clf.fit(features)
but the "CNN" is replaced by a frozen foundation model whose features are
already on disk (features.npy / ok.npy / manifest.csv per split).

Why subsample: libsvm OC-SVM is ~O(n^2). Bank A is 1.3M x 1536 -- infeasible.
BIHN's own training set was tens of thousands of patches. We take --n_train
patches with a per-slide cap so every normal slide contributes.

    --method exact   sklearn OneClassSVM (rbf). Fit time is ~O(n^2) (measured
                     exponent 1.99), so 40k ~ 3 min, 100k ~ 17 min, 200k ~ 1.1 h,
                     1.3M ~ 48 h. Practical ceiling is a few hundred thousand.
    --method sgd     Nystroem(rbf) + SGDOneClassSVM, linear in n. The bank is
                     STREAMED: Nystroem is fitted on a small subsample, then the
                     data is transformed and partial_fit in chunks, so the
                     n x n_components matrix is never materialised. Use for the
                     full bank. Note this APPROXIMATES the rbf kernel, so its
                     result is not a drop-in comparison with --method exact.

Example:
    python ocsvm_train.py \
        --bank_dir /home/user01/dn2_gigapath/features/bank \
        --out models/ocsvm_gigapath_nu0.1_n40k.pkl \
        --n_train 40000 --nu 0.1
"""
import argparse
import csv
import json
import os
import pickle
import time

import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_split(d):
    feats = np.load(os.path.join(d, 'features.npy'), mmap_mode='r')
    ok = np.load(os.path.join(d, 'ok.npy'))
    with open(os.path.join(d, 'manifest.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    assert feats.shape[0] == len(ok) == len(rows), f'{d}: shape/row mismatch'
    return feats, ok, rows


def stratified_subsample(rows, keep_mask, n_total, per_slide_cap, rng):
    """Pick <= n_total row indices, spread evenly over slides.
    If n_total >= the number available, every eligible row is returned."""
    idx = np.flatnonzero(keep_mask)
    if per_slide_cap <= 0 and n_total >= len(idx):
        n_sl = len({rows[j]['slide'] for j in idx})
        return idx, n_sl
    by = {}
    for j in idx:
        by.setdefault(rows[j]['slide'], []).append(j)
    n_slides = len(by)
    cap = per_slide_cap if per_slide_cap > 0 else max(1, n_total // n_slides)
    sel = []
    for js in by.values():
        js = np.asarray(js)
        sel.append(rng.choice(js, cap, replace=False) if len(js) > cap else js)
    sel = np.concatenate(sel)
    if len(sel) > n_total:
        sel = rng.choice(sel, n_total, replace=False)
    return np.sort(sel), n_slides



def show_config(name, args):
    """Print every resolved argument before doing any work (cf. BIHN show_configuration)."""
    d = vars(args)
    w = max(len(k) for k in d)
    print('=' * (w + 30))
    print(f'  {name}')
    print('=' * (w + 30))
    for k in sorted(d):
        print(f'  {k:<{w}} : {d[k]}')
    print('=' * (w + 30), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank_dir', nargs='+', required=True,
                    help='one or more feature dirs of NORMAL patches (concatenated)')
    ap.add_argument('--out', required=True, help='output .pkl')
    ap.add_argument('--n_train', type=int, default=40000)
    ap.add_argument('--per_slide_cap', type=int, default=0, help='0 = n_train/n_slides')
    ap.add_argument('--exclude_slides', default='', help='txt, one slide per line')
    ap.add_argument('--method', choices=['exact', 'sgd'], default='exact')
    ap.add_argument('--nu', type=float, default=0.1)            # BIHN default
    ap.add_argument('--gamma', default='scale')                 # BIHN default
    ap.add_argument('--standardize', action='store_true',
                    help='z-score features first (BIHN does not; try both)')
    ap.add_argument('--nystroem_components', type=int, default=2000,
                    help='higher = better rbf approximation, more memory per chunk')
    ap.add_argument('--chunk', type=int, default=20000,
                    help='sgd only: rows transformed and partial_fit at a time')
    ap.add_argument('--sgd_epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=500)            # BIHN seed
    args = ap.parse_args()
    show_config('ocsvm_train  (BIHN stage 2: fit one-class SVM)', args)
    rng = np.random.default_rng(args.seed)

    drop = set()
    if args.exclude_slides:
        drop = {l.strip() for l in open(args.exclude_slides) if l.strip()}

    # ---- gather training vectors -------------------------------------------------
    X_parts, slides_used = [], set()
    share = args.n_train // len(args.bank_dir)
    for d in args.bank_dir:
        feats, ok, rows = load_split(d)
        keep = ok == 1
        labels = np.array([int(r['label']) for r in rows])
        assert labels.max() == 0, f'{d} contains label!=0 rows; bank must be normal only'
        if drop:
            keep &= ~np.array([r['slide'] in drop for r in rows])
        sel, n_slides = stratified_subsample(rows, keep, share, args.per_slide_cap, rng)
        X_parts.append(np.asarray(feats[sel], dtype=np.float32))
        slides_used |= {rows[j]['slide'] for j in sel}
        print(f'  {d}: {len(sel):,} of {keep.sum():,} normal patches, {n_slides} slides')
    X = np.concatenate(X_parts)
    print(f'training matrix {X.shape}, {len(slides_used)} slides')

    # ---- fit ---------------------------------------------------------------------
    gamma = args.gamma if args.gamma in ('scale', 'auto') else float(args.gamma)
    t0 = time.time()

    if args.method == 'exact':
        est = 162.0 * (X.shape[0] / 39909.0) ** 1.99          # measured on this data
        print(f'fitting OneClassSVM on {X.shape[0]:,} x {X.shape[1]} '
              f'(single libsvm call, no progress bar)')
        print(f'  projected fit time ~{est / 60:.1f} min based on measured O(n^1.99)', flush=True)
        steps = [StandardScaler()] if args.standardize else []
        steps.append(svm.OneClassSVM(nu=args.nu, kernel='rbf', gamma=gamma, cache_size=4000))
        clf = make_pipeline(*steps)
        clf.fit(X)
    else:
        g = 1.0 / (X.shape[1] * X.var()) if gamma == 'scale' else gamma
        scaler = StandardScaler().fit(X) if args.standardize else None
        n_ny = min(args.nystroem_components, X.shape[0])
        sub = rng.choice(X.shape[0], min(len(X), max(n_ny * 5, 20000)), replace=False)
        Xs = scaler.transform(X[sub]) if scaler else X[sub]
        print(f'fitting Nystroem({n_ny} components, gamma={g:.5g}) on {len(sub):,} rows', flush=True)
        ny = Nystroem(kernel='rbf', gamma=g, n_components=n_ny,
                      random_state=args.seed).fit(Xs)
        del Xs
        est_gb = args.chunk * n_ny * 8 / 1e9
        print(f'streaming {X.shape[0]:,} rows in chunks of {args.chunk:,} '
              f'(~{est_gb:.1f} GB per chunk, {args.sgd_epochs} epochs)', flush=True)
        sgd = SGDOneClassSVM(nu=args.nu, random_state=args.seed)
        order = np.arange(X.shape[0])
        for ep in range(args.sgd_epochs):
            rng.shuffle(order)
            for a in tqdm(range(0, len(order), args.chunk), unit='chunk', dynamic_ncols=True,
                          desc=f'sgd epoch {ep + 1}/{args.sgd_epochs}'):
                b = np.sort(order[a:a + args.chunk])
                Xb = scaler.transform(X[b]) if scaler else X[b]
                sgd.partial_fit(ny.transform(Xb))
        steps = ([scaler] if scaler else []) + [ny, sgd]
        clf = make_pipeline(*steps)

    fit_s = time.time() - t0

    n_out = 0
    for a in range(0, X.shape[0], 50000):
        n_out += int((-clf.decision_function(X[a:a + 50000]) > 0).sum())
    frac_out = n_out / X.shape[0]                     # >0 = anomalous (BIHN sign)
    info = dict(backbone_dim=int(X.shape[1]), n_train=int(X.shape[0]),
                n_slides=len(slides_used), fit_seconds=round(fit_s, 1),
                train_outlier_fraction=round(frac_out, 4), args=vars(args))
    if args.method == 'exact':
        info['n_support_vectors'] = int(clf[-1].support_.shape[0])
    print(json.dumps(info, indent=2))
    print(f'  (train outlier fraction should be ~nu={args.nu})')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'clf': clf, 'info': info}, f)
    json.dump(info, open(os.path.splitext(args.out)[0] + '.json', 'w'), indent=2)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()