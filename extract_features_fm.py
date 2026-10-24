#!/usr/bin/env python
"""
BIHN-with-foundation-model -- feature extraction (frozen encoder).

Writes the SAME on-disk layout as dn2_gigapath/extract_features.py so that
ocsvm_train.py / ocsvm_score.py (and your existing metrics_report.py) work
on either backbone without changes:

    <out_dir>/features.npy   float16 (N, D), L2-normalised, row-aligned to manifest
    <out_dir>/ok.npy         uint8 (N,)  1 = decoded, 0 = unreadable -> dropped downstream
    <out_dir>/progress.json  rows done, for resume-after-crash
    <out_dir>/manifest.csv   copy of the input manifest (path,patch_name,slide,label)

Backbones
    plip      vinid/plip           CLIP ViT-B/32 image tower, projected 512-d
    gigapath  prov-gigapath tile   ViT-g/14, 1536-d  (you already have these; only
                                   re-run if you want a different manifest)

Usage (inside tmux):
    python extract_features_fm.py --backbone plip \
        --manifest /home/user01/dn2_gigapath/manifests/bank.csv \
        --out_dir features_plip/bank --batch_size 512 --num_workers 12
"""
import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

BACKBONES = {
    'plip':     dict(dim=512,  mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)),
    'gigapath': dict(dim=1536, mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
}


# ----------------------------------------------------------------------------- models
class PLIPImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import CLIPModel
        self.m = CLIPModel.from_pretrained('vinid/plip')

    def forward(self, x):
        return self.m.get_image_features(pixel_values=x)      # (B, 512)


class GigaPathTileEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.m = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)

    def forward(self, x):
        return self.m(x)                                        # (B, 1536)


def build_model(name):
    return {'plip': PLIPImageEncoder, 'gigapath': GigaPathTileEncoder}[name]().eval()


def build_transform(name):
    cfg = BACKBONES[name]
    return transforms.Compose([
        transforms.Resize(256 if name == 'gigapath' else 224,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std']),
    ])


# ----------------------------------------------------------------------------- data
def read_manifest(p):
    with open(p, newline='') as f:
        rows = list(csv.DictReader(f))
    need = {'path', 'patch_name', 'slide', 'label'}
    assert need <= set(rows[0].keys()), f'manifest needs columns {need}'
    return rows


class PatchList(Dataset):
    """Never raises: a bad JPEG yields zeros + ok=0 so a single corrupt file
    can't kill an 8-hour run."""
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            im = Image.open(self.rows[i]['path']).convert('RGB')
            return self.tf(im), 1
        except Exception:
            return torch.zeros(3, 224, 224), 0


# ----------------------------------------------------------------------------- main

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
    ap.add_argument('--backbone', choices=list(BACKBONES), default='plip')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--fp16', action='store_true', help='autocast forward pass')
    ap.add_argument('--overwrite', action='store_true',
                    help='required to touch an out_dir that already holds a complete features.npy')
    args = ap.parse_args()
    show_config('extract_features_fm  (frozen encoder -> memmap)', args)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    N, D = len(rows), BACKBONES[args.backbone]['dim']
    man_copy = os.path.join(args.out_dir, 'manifest.csv')
    if not os.path.exists(man_copy):
        shutil.copy(args.manifest, man_copy)

    feat_p, ok_p, prog_p = (os.path.join(args.out_dir, n)
                            for n in ('features.npy', 'ok.npy', 'progress.json'))
    if os.path.exists(feat_p) and not os.path.exists(prog_p) and not args.overwrite:
        raise SystemExit(f'{feat_p} exists but has no progress.json from this script -- '
                         f'it looks like a finished extraction from elsewhere (e.g. DN2). '
                         f'Refusing to overwrite. Pass --overwrite if you really mean it.')
    if os.path.exists(feat_p) and not args.overwrite:
        _n = np.load(feat_p, mmap_mode='r').shape[0]
        if _n != N:
            raise SystemExit(f'{feat_p} has {_n} rows but manifest has {N}; refusing.')
    mode = 'r+' if os.path.exists(feat_p) else 'w+'
    feats = np.lib.format.open_memmap(feat_p, mode=mode, dtype=np.float16, shape=(N, D))
    ok = (np.load(ok_p) if os.path.exists(ok_p) else np.zeros(N, np.uint8))
    done = json.load(open(prog_p))['rows_done'] if os.path.exists(prog_p) else 0
    print(f'{args.backbone}: {N:,} rows, dim {D}, resuming at row {done:,}')

    dev = torch.device('cuda')
    model = build_model(args.backbone).to(dev)
    ds = Subset(PatchList(rows, build_transform(args.backbone)), range(done, N))
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, shuffle=False)

    t0, ptr = time.time(), done
    bar = tqdm(total=N, initial=done, unit='patch', unit_scale=True, dynamic_ncols=True,
               desc=f'{args.backbone} -> {os.path.basename(args.out_dir)}')
    with torch.no_grad():
        for x, flag in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast('cuda', enabled=args.fp16):
                z = model(x).float()
            z = F.normalize(z, dim=1)                            # L2 as in DN2 memmaps
            b = z.shape[0]
            feats[ptr:ptr + b] = z.cpu().numpy().astype(np.float16)
            ok[ptr:ptr + b] = flag.numpy().astype(np.uint8)
            ptr += b
            bar.update(b)
            if (ptr // args.batch_size) % 50 == 0 or ptr == N:
                feats.flush(); np.save(ok_p, ok)
                json.dump({'rows_done': int(ptr)}, open(prog_p, 'w'))

    bar.close()
    feats.flush(); np.save(ok_p, ok)
    json.dump({'rows_done': int(N)}, open(prog_p, 'w'))
    print(f'done. unreadable: {(ok == 0).sum():,}  ->  {args.out_dir}')


if __name__ == '__main__':
    main()
