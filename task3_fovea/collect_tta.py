# -*- coding: utf-8 -*-
"""collect_tta.py -- multi-scale TTA for the pipeline family that actually won
on the leaderboard (v6: selected-best checkpoints, per-axis extraction, entropy
shrink).

The all-data snapshot family is deliberately not used here: it scored 21.27 on
the public LB even though its OOF was better, so the only trustworthy anchor is
the configuration that measured 16.787.

Writes, for final3.py compatibility:
    mem2_<tag>_tta.npz   oof_g / oof_f / te_g / te_f   (original pixels)
    mem_<tag>_tta.npz    oof_conf / te_conf
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
from hm1 import FoveaDS, FoveaNet, split_folds, CELL2CANON, CKPT, GATE
from hm2 import PatchDS, CELL2CANON2, PATCH
from common import train_items, test_items, WORK


def load(tag, k):
    ck = torch.load(os.path.join(CKPT, f'{tag}_fold{k}.pt'), map_location='cuda',
                    weights_only=False)
    hm1.DOG = (ck['state']['stem.0.weight'].shape[1] == 4)
    m = FoveaNet('resnet18').cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def entropy_of(h):
    f = h.flatten(2)
    mx = f.max(2, keepdim=True).values
    f = torch.where(f >= mx - GATE, f, torch.full_like(f, -1e30))
    p = torch.softmax(f, 2)
    ent = -(p * torch.log(p + 1e-30)).sum(2)[:, 0]
    w = h.shape[-1]
    return (ent / np.log(w * w)).cpu().numpy()


@torch.no_grad()
def views(m, ds):
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    G, F, C = [], [], []
    for x, _, _ in dl:
        g = np.zeros((x.shape[0], 2)); f = np.zeros((x.shape[0], 2))
        c = np.zeros(x.shape[0])
        for mirror in (False, True):
            xx = torch.flip(x, dims=[3]) if mirror else x
            h, _ = m(xx.cuda())
            if mirror:
                h = torch.flip(h, dims=[3])
            g += hm1.soft_argmax(h).cpu().numpy() / 2
            f += hm1.soft_argmax(h, gate=0.0).cpu().numpy() / 2
            c += entropy_of(h) / 2
        G.append(g); F.append(f); C.append(c)
    return np.concatenate(G), np.concatenate(F), np.concatenate(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--stage', type=int, default=2)
    ap.add_argument('--base', default='hm1b')
    ap.add_argument('--zooms', default='1.0,1.1')
    a = ap.parse_args()
    zooms = [float(z) for z in a.zooms.split(',')]

    items, te = train_items(), test_items()
    tr_c = te_c = None
    if a.stage == 2:
        b = np.load(os.path.join(WORK, f'mem2_{a.base}.npz'))
        # mem2 stores original px; PatchDS needs the canonical frame -> reuse the
        # mem3 record when present, else invert the geometry per image.
        p3 = os.path.join(WORK, f'mem3_{a.base}.npz')
        if os.path.exists(p3):
            d3 = np.load(p3)
            tr_c, te_c = d3['oof_canon'], d3['te_canon']
        else:
            tr_c = np.array([items[i].to_canon(*b['oof_g'][i]) for i in range(len(items))])
            te_c = np.array([te[i].to_canon(*b['te_g'][i]) for i in range(len(te))])

    acc = {}
    for z in zooms:
        hm1.ZOOM = z
        og = of = None
        oc = np.zeros(len(items)); tc = np.zeros(len(te))
        tg = np.zeros((len(te), 2)); tf = np.zeros((len(te), 2))
        for k in range(5):
            m = load(a.tag, k)
            if a.stage == 1:
                c1 = CELL2CANON / z
                g, f, c = views(m, FoveaDS([items[i] for i in split_folds()[1][k]],
                                           'train', train=False))
                g, f = g * c1, f * c1
            else:
                c2 = CELL2CANON2 / z
                def run(geos, cen, split):
                    gg, ff, cc = views(m, PatchDS(geos, cen, split, train=False))
                    base = np.clip(np.asarray(cen, float), PATCH / 2,
                                   hm1.CANON - PATCH / 2) - PATCH / 2
                    return base + gg * c2, base + ff * c2, cc
                g, f, c = run([items[i] for i in split_folds()[1][k]],
                              tr_c[split_folds()[1][k]], 'train')
            folds = split_folds()[1]
            if og is None:
                og = np.zeros((len(items), 2)); of = np.zeros((len(items), 2))
            og[folds[k]] = g; of[folds[k]] = f; oc[folds[k]] = c
            if a.stage == 1:
                g2, f2, c2_ = views(m, FoveaDS(te, 'test', train=False))
                g2, f2 = g2 * (CELL2CANON / z), f2 * (CELL2CANON / z)
            else:
                g2, f2, c2_ = run(te, te_c, 'test')
            tg += g2 / 5; tf += f2 / 5; tc += c2_ / 5
            del m
            torch.cuda.empty_cache()
        acc[z] = (og, of, oc, tg, tf, tc)
        print('zoom %.2f done' % z, flush=True)

    hm1.ZOOM = 1.0
    og = np.mean([acc[z][0] for z in zooms], axis=0)
    of = np.mean([acc[z][1] for z in zooms], axis=0)
    oc = np.mean([acc[z][2] for z in zooms], axis=0)
    tg = np.mean([acc[z][3] for z in zooms], axis=0)
    tf = np.mean([acc[z][4] for z in zooms], axis=0)
    tc = np.mean([acc[z][5] for z in zooms], axis=0)

    og = np.array([items[i].to_orig(*og[i]) for i in range(len(items))])
    of = np.array([items[i].to_orig(*of[i]) for i in range(len(items))])
    tg = np.array([te[i].to_orig(*tg[i]) for i in range(len(te))])
    tf = np.array([te[i].to_orig(*tf[i]) for i in range(len(te))])

    np.savez(os.path.join(WORK, f'mem2_{a.tag}_tta.npz'),
             oof_g=og, oof_f=of, te_g=tg, te_f=tf)
    np.savez(os.path.join(WORK, f'mem_{a.tag}_tta.npz'),
             oof_conf=oc, te_conf=tc)
    print('wrote mem2_%s_tta.npz / mem_%s_tta.npz' % (a.tag, a.tag))


if __name__ == '__main__':
    main()
