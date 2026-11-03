"""Image-generic LN modes: per-image Jacobian Gram over COCO, filter part projected out.

For each image J_i = d out_i / d theta (theta = LN gamma||beta, 160). The filter part of a
column is its least-squares fit, per output channel, on [photo_R, photo_G, photo_B, 1].
G_full = sum J_i^T J_i, G_struct = sum (P J_i)^T (P J_i), P = I - filter projection.
"""
import sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from net import *
import coco

torch.set_grad_enabled(True)
SRC = 'experiments/tiny_dw_whitebox_sky_480p/trials/packets/'


def batch(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().permute(0, 3, 1, 2) / 255


def ln_jacobian(p, style, x):
    gn, bn, _ = norm_names(style)
    th0 = torch.cat([p[gn], p[bn]])

    def f(th):
        q = dict(p); q[gn] = th[:80]; q[bn] = th[80:]
        return forward(q, x, style)[0]           # (3,H,W)
    J = torch.vmap(lambda e: torch.func.jvp(f, (th0,), (e,))[1], chunk_size=32)(torch.eye(160))
    return J.detach().double(), th0              # (160,3,H,W)


def strip_filter(J, x):
    """Remove per-channel affine-in-photo component from each Jacobian column."""
    A = torch.cat([x[0].reshape(3, -1).T.double(), torch.ones(x[0, 0].numel(), 1, dtype=torch.double)], 1)
    Q, _ = torch.linalg.qr(A)                     # (HW,4)
    Jf = J.reshape(160, 3, -1)
    return (Jf - (Jf @ Q) @ Q.T).reshape(J.shape)


if __name__ == '__main__':
    style = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    p = load_params(Checkpoint.open(SRC))
    imgs = coco.load('coco_train')[::12][:64]
    Gf = torch.zeros(160, 160, dtype=torch.double); Gs = torch.zeros_like(Gf)
    for i, a in enumerate(imgs):
        x = F.interpolate(batch(a[None]), scale_factor=0.5, mode='area')
        J, th0 = ln_jacobian(p, style, x)
        Js = strip_filter(J, x)
        Jm, Jsm = J.reshape(160, -1), Js.reshape(160, -1)
        Gf += Jm @ Jm.T; Gs += Jsm @ Jsm.T
    ef, Vf = torch.linalg.eigh(Gf); es, Vs = torch.linalg.eigh(Gs)
    ef, Vf, es, Vs = ef.flip(0), Vf.flip(1), es.flip(0), Vs.flip(1)
    filt = 1 - torch.einsum('ik,ij,jk->k', Vf, Gs, Vf) / ef
    print('full   sqrt(eig)/filter-share', [f'{s:.1f}/{f:.2f}' for s, f in zip(ef[:10].sqrt().tolist(), filt[:10].tolist())])
    print('struct sqrt(eig)', [round(s, 1) for s in es[:12].sqrt().tolist()])
    print('struct energy share of total', float(es.sum() / ef.sum()))
    np.savez(f'runs/astra2/lnjac/coco_svd_s{style}.npz', Sf=ef.sqrt().numpy(), Vf=Vf.T.numpy(), filt=filt.numpy(),
             Ss=es.clamp(min=0).sqrt().numpy(), Vs=Vs.T.numpy(), theta0=th0.detach().numpy())
