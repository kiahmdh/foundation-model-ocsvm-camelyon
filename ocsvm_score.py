#!/usr/bin/env python
"""
BIHN evaluation -- score a query split with the fitted OC-SVM and report.

Reproduces the "performance summary" block of anomaly_detector.py:
    score = -decision_function            (>0 anomalous, fixed threshold 0)
    balanced accuracy @0, AUROC, F1 @0, FPR/FNR @0, ROC.png
plus the extras you already report for DN2 (AP, slide-level AUROC). All rates
are reported in PERCENT (95.41, not 0.9541) to match the DN2 tables.

--save_n_examples N writes, per kind, ONE montage PNG of the N worst FP / FN
patches (10 per row, 128 px thumbnails) plus a CSV of their paths and scores.
Nothing is copied, so results/ stays small.

Writes results/<run>/scores.csv with columns  patch_name,slide,label,score,pred
(higher score = more anomalous, same convention as your DN2 scores).

Example:
    python ocsvm_score.py --model models/ocsvm_gigapath_nu0.1_n40k.pkl \
        --query_dir /home/user01/dn2_gigapath/features/test \
        --run test_gigapath_nu0.1 --save_n_examples 100
"""
import argparse
import csv
import json
import os
import pickle
import shutil

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, roc_curve)


def fast_rbf_decision(ocsvm, X):
    """decision_function for an rbf OneClassSVM via BLAS instead of libsvm.

    libsvm evaluates the kernel one query at a time on a single core. The same
    quantity is  dual_coef @ exp(-gamma * ||X - SV||^2) + intercept, and the
    squared-distance matrix is a matmul, so numpy/OpenBLAS uses every core.
    Accumulate in float64: at float32 the result matches sklearn to ~6e-06,
    which leaves AUROC unchanged but can reorder exactly-tied patches. float64
    matches to ~1e-13 and reproduces sklearn's ordering exactly, so scores.csv
    is byte-comparable with a libsvm run (--no_fast).
    """
    sv = ocsvm.support_vectors_.astype(np.float64)
    d2 = (np.einsum('ij,ij->i', X, X)[:, None]
          + np.einsum('ij,ij->i', sv, sv)[None, :]
          - 2.0 * (X @ sv.T))
    np.maximum(d2, 0, out=d2)
    K = np.exp(-ocsvm._gamma * d2)
    return K @ ocsvm.dual_coef_.ravel() + ocsvm.intercept_[0]


def decision(clf, X, fast):
    """Apply any pipeline pre-steps, then score with libsvm or the BLAS path."""
    from sklearn.svm import OneClassSVM
    est = clf[-1]
    if not (fast and isinstance(est, OneClassSVM) and est.kernel == 'rbf'):
        return clf.decision_function(X)
    for step in clf[:-1]:
        X = step.transform(X)
    return fast_rbf_decision(est, np.ascontiguousarray(X, dtype=np.float64))


def load_split(d):
    feats = np.load(os.path.join(d, 'features.npy'), mmap_mode='r')
    ok = np.load(os.path.join(d, 'ok.npy'))
    with open(os.path.join(d, 'manifest.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    assert feats.shape[0] == len(ok) == len(rows)
    return feats, ok, rows


def slide_metrics(scores, labels, slides, agg):
    s_score, s_lab = {}, {}
    for sc, lb, sl in zip(scores, labels, slides):
        s_score.setdefault(sl, []).append(sc)
        s_lab[sl] = max(s_lab.get(sl, 0), lb)          # slide is tumour if any patch is
    names = sorted(s_score)
    y = np.array([s_lab[n] for n in names])
    if agg == 'max':
        v = np.array([np.max(s_score[n]) for n in names])
    elif agg == 'mean':
        v = np.array([np.mean(s_score[n]) for n in names])
    else:                                                # BIHN figure-A statistic
        v = np.array([np.mean(np.asarray(s_score[n]) > 0) for n in names])
    auc = roc_auc_score(y, v) if 0 < y.sum() < len(y) else float('nan')
    return auc, len(names), int(y.sum())



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
    ap.add_argument('--model', required=True)
    ap.add_argument('--query_dir', required=True)
    ap.add_argument('--run', required=True, help='results/<run>/')
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--chunk', type=int, default=20000)
    ap.add_argument('--no_fast', action='store_true',
                    help='use libsvm (single-core) instead of the BLAS rbf path')
    ap.add_argument('--save_n_examples', type=int, default=100,
                    help='montage + csv of this many worst FP and FN patches (0 = off)')
    ap.add_argument('--thumb', type=int, default=128, help='montage thumbnail size in px')
    args = ap.parse_args()
    show_config('ocsvm_score  (BIHN evaluation)', args)

    out = os.path.join(args.results_root, args.run)
    os.makedirs(out, exist_ok=True)
    bundle = pickle.load(open(args.model, 'rb'))
    clf = bundle['clf']

    feats, ok, rows = load_split(args.query_dir)
    keep = np.flatnonzero(ok == 1)
    print(f'{args.query_dir}: {len(keep):,} readable of {len(rows):,}')

    # ---- score in chunks (decision_function is n_query x n_SV kernel evals) ----
    scores = np.empty(len(keep), np.float32)
    for a in tqdm(range(0, len(keep), args.chunk), unit='chunk', dynamic_ncols=True,
                  desc=f'scoring {os.path.basename(os.path.normpath(args.query_dir))}'):
        idx = keep[a:a + args.chunk]
        X = np.asarray(feats[idx], dtype=np.float32)
        scores[a:a + len(idx)] = -decision(clf, X, not args.no_fast)  # BIHN sign flip

    labels = np.array([int(rows[j]['label']) for j in keep])
    slides = [rows[j]['slide'] for j in keep]
    pred = (scores > 0).astype(int)

    with open(os.path.join(out, 'scores.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patch_name', 'slide', 'label', 'Ascore', 'pred'])  # Ascore: dn2_score.py's schema, so metrics_report.py reads it with no renaming
        for j, sc, p in zip(keep, scores, pred):
            w.writerow([rows[j]['patch_name'], rows[j]['slide'], rows[j]['label'], f'{sc:.6f}', p])

    # ---- metrics (BIHN block + your usual extras) ------------------------------
    P = 100.0                                                 # report rates in percent
    m = {}
    if 0 < labels.sum() < len(labels):
        m['patch_auroc'] = P * roc_auc_score(labels, scores)
        m['patch_ap'] = P * average_precision_score(labels, scores)
        m['balanced_acc@0'] = P * balanced_accuracy_score(labels, pred)
        m['f1@0'] = P * f1_score(labels, pred)
        tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
        m.update(tp=tp, fp=fp, fn=fn, tn=tn,
                 sensitivity=P * tp / max(tp + fn, 1), specificity=P * tn / max(tn + fp, 1),
                 fpr_normal_flagged=P * fp / max(tn + fp, 1),   # "predicted anomalies in normal data"
                 fnr_anomaly_missed=P * fn / max(tp + fn, 1))   # "predicted normals in anomaly data"
        for agg in ('max', 'mean', 'frac_pos'):
            auc, ns, nt = slide_metrics(scores, labels, slides, agg)
            m[f'slide_{agg}_auroc'] = P * auc
        m['n_slides'], m['n_tumour_slides'] = ns, nt
        fpr, tpr, _ = roc_curve(labels, scores)
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.plot(m['fpr_normal_flagged'] / P, m['sensitivity'] / P, '*', ms=12, label='threshold 0')
            ax.set_xlabel('False positive rate (1-specificity)'); ax.set_ylabel('True positive rate')
            ax.set_title(f"{args.run}  AUROC={m['patch_auroc']:.2f}%"); ax.legend()
            fig.savefig(os.path.join(out, 'ROC.png'), bbox_inches='tight'); plt.close(fig)
        except Exception as e:
            print('ROC plot skipped:', e)
    else:
        print('single-class query split: only scores written, no AUROC')
    m['n_patches'] = int(len(keep)); m['prevalence'] = P * float(labels.mean())
    m['model'] = args.model; m['query_dir'] = args.query_dir
    m['train_info'] = bundle.get('info', {})

    with open(os.path.join(out, 'metrics.txt'), 'w') as f:
        f.write('# rates in percent; threshold fixed at 0 (BIHN)\n')
        for k, v in m.items():
            if not isinstance(v, dict):
                f.write(f'{k}: {v:.2f}\n' if isinstance(v, float) else f'{k}: {v}\n')
    json.dump(m, open(os.path.join(out, 'config.json'), 'w'), indent=2, default=str)
    print(open(os.path.join(out, 'metrics.txt')).read())

    summ = os.path.join(args.results_root, 'summary.tsv')
    new = not os.path.exists(summ)
    with open(summ, 'a') as f:
        cols = ['run', 'patch_auroc', 'patch_ap', 'balanced_acc@0', 'f1@0',
                'slide_max_auroc', 'slide_mean_auroc', 'slide_frac_pos_auroc', 'n_patches']
        if new:
            f.write('\t'.join(cols) + '\n')
        f.write('\t'.join([args.run] + [f'{m.get(c, float("nan")):.2f}' if isinstance(m.get(c), float)
                                        else str(m.get(c, '')) for c in cols[1:]]) + '\n')

    # ---- BIHN-style FP / FN examples: one montage PNG + csv per kind, no copies ----
    if args.save_n_examples > 0:
        from PIL import Image, ImageDraw
        T, per_row = args.thumb, 10
        for kind, mask, order in (('FP', (labels == 0) & (pred == 1), -scores),
                                  ('FN', (labels == 1) & (pred == 0), scores)):
            cand = np.flatnonzero(mask)
            cand = cand[np.argsort(order[cand])][:args.save_n_examples]
            with open(os.path.join(out, f'{kind}_examples.csv'), 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['rank', 'score', 'slide', 'patch_name', 'path'])
                for r_, i in enumerate(cand):
                    r = rows[keep[i]]
                    w.writerow([r_, f'{scores[i]:+.4f}', r['slide'], r['patch_name'], r['path']])
            n_rows = (len(cand) + per_row - 1) // per_row
            sheet = Image.new('RGB', (per_row * T, max(n_rows, 1) * T), 'white')
            draw = ImageDraw.Draw(sheet)
            for r_, i in enumerate(cand):
                try:
                    im = Image.open(rows[keep[i]]['path']).convert('RGB').resize((T, T))
                except Exception:
                    im = Image.new('RGB', (T, T), 'grey')
                x, y = (r_ % per_row) * T, (r_ // per_row) * T
                sheet.paste(im, (x, y))
                draw.text((x + 3, y + 2), f'{scores[i]:+.2f}', fill='yellow')
            sheet.save(os.path.join(out, f'{kind}_examples.png'))
            print(f'  {kind}: {len(cand)} worst -> {kind}_examples.png + {kind}_examples.csv '
                  f'(sorted by score; row-major, 10 per row)')

if __name__ == '__main__':
    main()