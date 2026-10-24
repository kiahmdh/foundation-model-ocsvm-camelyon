#!/usr/bin/env python
"""
Build the DN2-format results table (AUROC / F1 / Recall at max, top1 and patch
level) from one or more results/<run>/scores.csv files.

ocsvm_score.py reports slide-level AUROC only. The F1 and Recall columns in the
DN2 table need a slide-level DECISION, not just a ranking, so they are computed
here:

    max aggregation    slide score = max patch score
    top1 aggregation   slide score = mean of the top --top_pct % of patch scores

--threshold youden (default) picks the decision threshold maximising Youden's J
(tpr - fpr) on the data being evaluated, matching the DN2 results table. It is
fitted on the same data it is scored on, so F1 and Recall are optimistic;
AUROC is threshold-free and unaffected.

--threshold zero uses the fixed threshold 0 from the reference implementation.
At PATCH level that is the meaningful BIHN operating point. At SLIDE level it is
degenerate: the maximum patch score of a slide with thousands of patches exceeds
0 for every slide, so every slide is predicted positive, recall is 100% and F1
collapses to a constant set by prevalence.

--exclude drops whole slides by their `class` in reference.csv (negative /
itc / micro / macro). A slide is removed ENTIRELY -- all its patches leave the
evaluation, not just its tumour patches -- which is the only coherent way to do
it, since keeping a micro slide's normal patches while discarding its tumour
patches would inflate specificity against a slide contributing no positives.
Same convention as exclude_class.py in the DN2 pipeline.

Every rate is a percentage. Writes results_table.html (styled like the DN2
table) and results_table.tsv.

    python make_table.py \
        --run results/test_gigapath_nu0.02_g3_n40000 "GigaPath OC-SVM (dev-tuned)" \
        --run results/test_gigapath_nu0.1_gscale_n40000 "GigaPath OC-SVM (BIHN default)" \
        --run results/test_plip_nu0.02_g10_n40000 "PLIP OC-SVM (dev rank 1)" \
        --run results/test_plip_nu0.05_g10_n40000 "PLIP OC-SVM (dev rank 2)" \
        --title "BIHN OC-SVM on Cam16 (Bank A)"
"""
import argparse
import csv
import html
import os

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve

COLS = ['AUROC_max', 'F1_max', 'Recall_max', 'AUROC_top1', 'F1_top1', 'Recall_top1',
        'AUROC_patch', 'F1_patch', 'Recall_patch']


def stem(s):
    """test_001.tif -> test_001 ; reference.csv and scores.csv disagree on extensions."""
    return os.path.splitext(str(s).strip())[0]


def load_excluded_slides(ref_csv, classes, class_col='class'):
    """Return the set of slide stems whose class is in `classes`.
    Camelyon16 reference.csv: class in {negative, itc, micro, macro}.
    Camelyon17 stage_labels.csv: same values under the column name 'stage'."""
    with open(ref_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'{ref_csv} is empty')
    cols = {c.lower(): c for c in rows[0]}
    if class_col.lower() not in cols:
        raise SystemExit(f'{ref_csv} has no "{class_col}" column; found {list(rows[0])}. '
                         f'Use --class_col to name the right one.')
    class_col = class_col.lower()
    id_col = next((cols[c] for c in ('image', 'slide', 'patient', 'node') if c in cols), None)
    if id_col is None:
        raise SystemExit(f'{ref_csv} has no image/slide id column; found {list(rows[0])}')
    cc, want = cols[class_col], {c.lower() for c in classes}
    ref = {stem(r[id_col]): str(r[cc]).strip().lower() for r in rows}
    drop = {k for k, v in ref.items() if v in want}
    return drop, ref


def read_scores(d, drop=None, ref=None):
    with open(os.path.join(d, 'scores.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    if drop:
        seen = {stem(r['slide']) for r in rows}
        missing = seen - set(ref)
        if missing:
            print(f'  WARNING: {len(missing)} slide(s) in scores.csv absent from reference.csv, '
                  f'kept as-is: {sorted(missing)[:5]}{"..." if len(missing) > 5 else ""}')
        before = len(seen)
        rows = [r for r in rows if stem(r['slide']) not in drop]
        after = len({stem(r['slide']) for r in rows})
        print(f'  excluded {before - after} slide(s); {after} remain')
        if not rows:
            raise SystemExit('every slide was excluded')
    key = next((k for k in ('Ascore', 'score') if k in rows[0]), None)
    if key is None:
        raise SystemExit(f"{d}/scores.csv has no Ascore/score column; found {list(rows[0])}")
    score = np.array([float(r[key]) for r in rows], np.float64)
    label = np.array([int(r['label']) for r in rows], np.int64)
    slide = np.array([r['slide'] for r in rows])
    return score, label, slide


def youden_threshold(y, v):
    """Threshold maximising tpr - fpr. Fitted on the data it scores, as in DN2."""
    fpr, tpr, thr = roc_curve(y, v)
    return thr[int(np.argmax(tpr - fpr))]


def decide(y, v, mode, fixed):
    t = youden_threshold(y, v) if mode == 'youden' else fixed
    return (v >= t).astype(int) if mode == 'youden' else (v > t).astype(int), t


def slide_level(score, label, slide, how, top_pct, thr, mode):
    names = sorted(set(slide.tolist()))
    y, v = [], []
    for n in names:
        m = slide == n
        s = score[m]
        y.append(int(label[m].max()))                       # slide is tumour if any patch is
        if how == 'max':
            v.append(s.max())
        else:
            k = max(1, int(round(len(s) * top_pct / 100.0)))
            v.append(np.sort(s)[-k:].mean())
    y, v = np.array(y), np.array(v)
    pred, t = decide(y, v, mode, thr)
    auc = 100 * roc_auc_score(y, v) if 0 < y.sum() < len(y) else float('nan')
    return auc, 100 * f1_score(y, pred), 100 * recall_score(y, pred), len(names), int(y.sum()), t


def metrics_for(d, top_pct, thr, mode, drop=None, ref=None):
    score, label, slide = read_scores(d, drop, ref)
    out = {}
    for how, tag in (('max', 'max'), ('top1', 'top1')):
        a, f, r, ns, nt, t = slide_level(score, label, slide, how, top_pct, thr, mode)
        out[f'AUROC_{tag}'], out[f'F1_{tag}'], out[f'Recall_{tag}'] = a, f, r
        out[f'_thr_{tag}'] = t
    pred, tp_ = decide(label, score, mode, thr)
    out['_thr_patch'] = tp_
    out['AUROC_patch'] = 100 * roc_auc_score(label, score)
    out['F1_patch'] = 100 * f1_score(label, pred)
    out['Recall_patch'] = 100 * recall_score(label, pred)
    out['_n_slides'], out['_n_tumour'] = ns, nt
    out['_n_patches'] = len(score)
    return out


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#fff;padding:24px;color:#222}
table{border-collapse:collapse;margin-bottom:28px;font-size:13px}
caption{background:#8e4a6f;color:#fff;font-weight:700;font-style:italic;padding:7px 12px;
        border:2px solid #6d2b4f;text-align:center;font-size:14px}
th{background:#f2d9e6;color:#8e4a6f;font-style:italic;font-weight:700;border:2px solid #8e4a6f;
   padding:6px 12px;white-space:nowrap}
td{background:#fff;border:2px solid #8e4a6f;padding:6px 12px;text-align:center}
td.name{text-align:left;font-weight:600;background:#faf3f7;white-space:nowrap}
td.best{background:#4dfd4d;font-weight:700}
.note{font-size:12px;color:#555;max-width:900px;line-height:1.55}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs=2, action='append', metavar=('DIR', 'LABEL'), required=True)
    ap.add_argument('--title', default='Results')
    ap.add_argument('--top_pct', type=float, default=1.0,
                    help='top1 aggregation = mean of the top N%% of patch scores')
    ap.add_argument('--slide_thr', type=float, default=0.0,
                    help='fixed threshold, used only with --threshold zero')
    ap.add_argument('--threshold', choices=['youden', 'zero'], default='youden',
                    help='youden matches the DN2 table; zero is the BIHN fixed threshold')
    ap.add_argument('--out', default='results_table')
    ap.add_argument('--reference_csv', default='',
                    help='reference.csv with a class column (negative/itc/micro/macro)')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='class values whose slides are dropped entirely, '
                         'e.g. --exclude micro  or  --exclude itc micro')
    ap.add_argument('--class_col', default='class',
                    help="column holding the lesion class. Camelyon16 reference.csv uses "
                         "'class'; Camelyon17 stage_labels.csv uses 'stage'.")
    ap.add_argument('--highlight', action='store_true', help='green-highlight the best per column')
    args = ap.parse_args()

    drop, ref = None, None
    if args.exclude:
        if not args.reference_csv:
            raise SystemExit('--exclude needs --reference_csv')
        drop, ref = load_excluded_slides(args.reference_csv, args.exclude, args.class_col)
        print(f'excluding class {args.exclude}: {len(drop)} slide(s) in reference.csv\n')

    rows = []
    for d, label in args.run:
        print(label)
        m = metrics_for(d, args.top_pct, args.slide_thr, args.threshold, drop, ref)
        rows.append((label, m))
        print(f'  {m["_n_patches"]:,} patches, {m["_n_slides"]} slides '
              f'({m["_n_tumour"]} tumour)')
        print(f'  thresholds  max {m["_thr_max"]:+.4f}  top1 {m["_thr_top1"]:+.4f}  '
              f'patch {m["_thr_patch"]:+.4f}\n')

    best = {c: max(r[1][c] for r in rows) for c in COLS} if args.highlight else {}

    with open(args.out + '.tsv', 'w') as f:
        f.write('\t'.join(['method'] + COLS) + '\n')
        for label, m in rows:
            f.write('\t'.join([label] + [f'{m[c]:.2f}' for c in COLS]) + '\n')

    h = [f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>',
         f'<table><caption>{html.escape(args.title)}</caption><tr><th>Method</th>'
         + ''.join(f'<th>{c}</th>' for c in COLS) + '</tr>']
    for label, m in rows:
        h.append(f'<tr><td class="name">{html.escape(label)}</td>' + ''.join(
            f'<td class="{"best" if args.highlight and abs(m[c]-best[c])<1e-9 else ""}">{m[c]:.2f}</td>'
            for c in COLS) + '</tr>')
    n = rows[0][1]
    h.append('</table>')
    h.append(f'<p class="note">All values are percentages. '
             f'{n["_n_patches"]:,} patches over {n["_n_slides"]} slides '
             f'({n["_n_tumour"]} tumour-bearing). '
             f'<b>max</b>: slide score is the maximum patch score. '
             f'<b>top1</b>: slide score is the mean of the top {args.top_pct:g}% of patch scores. '
             + (f'F1 and Recall use a Youden-optimal threshold fitted on the same data and are '
                f'therefore optimistic; AUROC is threshold-free.' if args.threshold == 'youden'
                else f'F1 and Recall use the fixed threshold {args.slide_thr:g} from the reference '
                     f'implementation; AUROC is threshold-free.')
             + (f' Slides of class {", ".join(args.exclude)} were excluded entirely.'
                if args.exclude else '')
             + '</p></body></html>')
    open(args.out + '.html', 'w').write('\n'.join(h))
    print(f'\nwrote {args.out}.html and {args.out}.tsv')


if __name__ == '__main__':
    main()