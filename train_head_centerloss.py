#!/usr/bin/env python
"""
OPTIONAL -- BIHN stage 1 analog on top of a FROZEN foundation model.

BIHN trains EfficientNet-B0 on an auxiliary tissue-type classification task
with  loss = CE + cl_weight * CenterLoss(target class)   (train_cnn.py, losses.py).
Here the encoder is frozen (GigaPath / PLIP) and only a small MLP head is
trained on the cached features. The CenterLoss is a straight port of the BI
implementation (moving-average centre update, mu=0.5, per-dim normalised).

You must supply the auxiliary labels yourself:  --aux_csv  slide,aux_class
(see the notes in the chat: Camelyon16 has no organ-level classes, the usual
choices are acquisition centre -- RUMC vs UMCU -- or k-means pseudo-classes).

After training, --export writes head embeddings for each --export_dir into
<export_root>/<split_name>/ in the SAME layout as the raw feature dirs, so
ocsvm_train.py / ocsvm_score.py run on them unchanged.

Example:
    python train_head_centerloss.py \
        --train_dir /home/user01/dn2_gigapath/features/bank \
        --aux_csv aux_labels_c16.csv --target_classes RUMC UMCU \
        --epochs 15 --cl_weight 1.0 --out models/head_gigapath.pt \
        --export_dir /home/user01/dn2_gigapath/features/bank \
                     /home/user01/dn2_gigapath/features/test \
        --export_root features_gigapath_head
"""
import argparse
import csv
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- BI CenterLoss
def load_bi_centerloss(repo_root):
    """Import the AUTHORS' CenterLoss from a clone of
    github.com/Boehringer-Ingelheim/anomaly-detection-in-histology.

    models/losses.py imports only torch and torch.nn, so it loads cleanly on a
    modern stack even though the rest of the repo does not (np.float, Qt5Agg,
    cm.get_cmap). Using it unmodified is what you want to be able to claim in
    the report. Falls back to the equivalent port below if the clone is absent.
    """
    import importlib.util
    p = os.path.join(repo_root, 'models', 'losses.py')
    spec = importlib.util.spec_from_file_location('bi_losses', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f'using authors CenterLoss from {p}')
    return mod.CenterLoss


class CenterLoss(nn.Module):
    """Line-for-line port of models/losses.py::CenterLoss (used when --bi_repo
    is not given). Same moving-average centre update, same mu, same division by
    feat_dim at the end."""
    def __init__(self, num_classes, feat_dim, device, constrained_classes=None, mu=0.5):
        super().__init__()
        self.num_classes, self.mu = num_classes, mu
        self.centers = torch.randn(num_classes, feat_dim, device=device)
        self.set = False
        self.constrained = constrained_classes

    def forward(self, x, labels):
        with torch.no_grad():
            for c in range(self.num_classes):
                idx = labels == c
                if idx.any():
                    new = x[idx].mean(0)
                    self.centers[c] = ((1 - self.mu) * self.centers[c] + self.mu * new) if self.set else new
                    self.set = True
        B = x.size(0)
        dist = (x.pow(2).sum(1, keepdim=True) + self.centers.pow(2).sum(1)[None, :]
                - 2 * x @ self.centers.t())
        mask = labels[:, None] == torch.arange(self.num_classes, device=x.device)[None, :]
        n = mask.sum(0).clamp(min=1)
        loss = (dist * mask).clamp(1e-12, 1e12).sum(0) / n
        loss = loss[self.constrained] if self.constrained is not None else loss
        return loss.mean() / x.size(1)


class Head(nn.Module):
    def __init__(self, d_in, d_emb, n_classes):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(1024, d_emb), nn.GELU())
        self.cls = nn.Linear(d_emb, n_classes)

    def forward(self, x):
        z = self.body(x)
        return {'categories': self.cls(z), 'pooled_codes': z}


# ----------------------------------------------------------------------------- data
def load_split(d):
    feats = np.load(os.path.join(d, 'features.npy'), mmap_mode='r')
    ok = np.load(os.path.join(d, 'ok.npy'))
    with open(os.path.join(d, 'manifest.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    return feats, ok, rows



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
    ap.add_argument('--train_dir', nargs='+', required=True, help='NORMAL feature dirs')
    ap.add_argument('--aux_csv', required=True, help='csv with columns slide,aux_class')
    ap.add_argument('--target_classes', nargs='*', default=None,
                    help='classes the centre loss is applied to (BIHN: the target tissue). '
                         'omit = all classes')
    ap.add_argument('--n_per_class', type=int, default=50000)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--d_emb', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ce_weight', type=float, default=1.0)
    ap.add_argument('--cl_weight', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=500)
    ap.add_argument('--bi_repo', default='..',
                    help="root of the cloned BI repo; its models/losses.py::CenterLoss "
                         "is used verbatim. Set '' to use the local port instead.")
    ap.add_argument('--out', required=True)
    ap.add_argument('--export_dir', nargs='*', default=[])
    ap.add_argument('--export_root', default='features_head')
    args = ap.parse_args()
    show_config('train_head_centerloss  (BIHN stage 1 analog)', args)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device('cuda')

    aux = {r['slide']: r['aux_class'] for r in csv.DictReader(open(args.aux_csv))}
    classes = sorted(set(aux.values())); cid = {c: i for i, c in enumerate(classes)}
    print('aux classes:', cid)

    # ---- gather balanced training set by aux class -----------------------------
    X, y = [], []
    per_class = {c: [] for c in classes}
    for d in args.train_dir:
        feats, ok, rows = load_split(d)
        for j in np.flatnonzero(ok == 1):
            s = rows[j]['slide']
            if s in aux:
                per_class[aux[s]].append((d, j))
    cache = {}
    for c, items in per_class.items():
        if len(items) > args.n_per_class:
            items = [items[i] for i in rng.choice(len(items), args.n_per_class, replace=False)]
        for d, j in items:
            if d not in cache:
                cache[d] = load_split(d)[0]
            X.append(np.asarray(cache[d][j], np.float32)); y.append(cid[c])
        print(f'  {c}: {len(items):,} patches')
    X = torch.tensor(np.stack(X)); y = torch.tensor(y)
    perm = torch.randperm(len(y)); nval = int(len(y) * args.val_frac)
    va, tr = perm[:nval], perm[nval:]

    # ---- train ----------------------------------------------------------------
    model = Head(X.shape[1], args.d_emb, len(classes)).to(dev)
    tgt = [cid[c] for c in args.target_classes] if args.target_classes else None
    CL = CenterLoss
    if args.bi_repo and os.path.isfile(os.path.join(args.bi_repo, 'models', 'losses.py')):
        CL = load_bi_centerloss(args.bi_repo)
    else:
        print('BI repo not found, using the local port of CenterLoss')
    center = CL(num_classes=len(classes), feat_dim=args.d_emb, device=dev,
                constrained_classes=tgt, mu=0.5)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)   # BIHN optimiser
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train(); idx = tr[torch.randperm(len(tr))]; tot = 0
        for a in range(0, len(idx), args.batch_size):
            b = idx[a:a + args.batch_size]
            xb, yb = X[b].to(dev), y[b].to(dev)
            out = model(xb)
            loss = args.ce_weight * F.cross_entropy(out['categories'], yb) \
                 + args.cl_weight * center(out['pooled_codes'], yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            acc = (model(X[va].to(dev))['categories'].argmax(1).cpu() == y[va]).float().mean().item()
        print(f'epoch {ep:02d}  loss {tot / len(tr):.4f}  val acc {acc:.4f}')
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'state': best_state, 'classes': classes, 'd_in': X.shape[1],
                'd_emb': args.d_emb, 'args': vars(args), 'val_acc': best}, args.out)
    print(f'best val acc {best:.4f} -> {args.out}')

    # ---- export embeddings in the standard feature-dir layout ------------------
    for d in args.export_dir:
        name = os.path.basename(os.path.normpath(d))
        od = os.path.join(args.export_root, name); os.makedirs(od, exist_ok=True)
        feats, ok, rows = load_split(d)
        outm = np.lib.format.open_memmap(os.path.join(od, 'features.npy'), 'w+',
                                         np.float16, (feats.shape[0], args.d_emb))
        with torch.no_grad():
            for a in range(0, feats.shape[0], 8192):
                xb = torch.tensor(np.asarray(feats[a:a + 8192], np.float32)).to(dev)
                z = F.normalize(model(xb)['pooled_codes'], dim=1)
                outm[a:a + len(z)] = z.cpu().numpy().astype(np.float16)
        outm.flush(); np.save(os.path.join(od, 'ok.npy'), ok)
        shutil.copy(os.path.join(d, 'manifest.csv'), os.path.join(od, 'manifest.csv'))
        json.dump({'rows_done': int(feats.shape[0])}, open(os.path.join(od, 'progress.json'), 'w'))
        print(f'exported {feats.shape[0]:,} x {args.d_emb} -> {od}')


if __name__ == '__main__':
    main()
