"""Differentiable simplification objectives and a packet optimiser over COCO.

Every loss is computed against the photo x and the frozen base style-0 render b.
No external target image: the objective names a look, the network finds how to draw it.
"""
import sys, time, math
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from net import *
import coco

SRC = 'experiments/tiny_dw_whitebox_sky_480p/trials/packets/'
W = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def luma(y):
    return (y * W).sum(1, keepdim=True)


def gauss(x, s):
    r = max(1, int(3 * s))
    k = torch.exp(-torch.arange(-r, r + 1).float() ** 2 / (2 * s * s)); k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def grads(x):
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def gmag(x, eps=1e-4):
    dx, dy = grads(x)
    return torch.sqrt(dx * dx + dy * dy + eps)


def edge_map(x):
    """Soft photo edges in [0,1]: blurred-luma gradient over its per-image 90th percentile."""
    g = gmag(gauss(luma(x), 1.0))
    q = torch.quantile(g.flatten(1), 0.9, dim=1).view(-1, 1, 1, 1)
    return (g / (q + 1e-6)).clamp(0, 1)


def chroma(y):
    return y - luma(y)


def tone_stats(y):
    l = luma(y)
    return l.mean((1, 2, 3)), l.std((1, 2, 3))


def hinge(v):
    return F.relu(v)


def losses(y, x, b, E, cfg):
    """y output, x photo, b base render, E photo edges. cfg = weights/targets of one style."""
    L, Lb = luma(y), luma(b)
    out = {}
    nonedge = (1 - gauss(E, 1.5)).clamp(0, 1)
    # flatness: L1 gradient (TV) of the colour image away from photo edges
    tv = gmag(y).mean(1, keepdim=True)
    out['flat'] = (tv * nonedge).mean() / nonedge.mean()
    # edges kept: luma contrast across photo edges at least `edge` x photo's
    ge, gp = gmag(gauss(L, 0.7)), gmag(gauss(luma(x), 0.7))
    out['edge'] = hinge(cfg.get('edge', 1.0) * gp - ge)[E > 0.5].mean() if (E > 0.5).any() else y.sum() * 0
    # dark lines on edges: luma below its local mean where the photo has an edge
    dark = (gauss(L, 2.5) - L)
    out['line'] = -(dark * E).mean() / E.mean()
    # low-frequency tone: follow a target curve of the base render (contrast c, lift l)
    m = Lb.mean((2, 3), keepdim=True)
    if cfg.get('levels'):
        # auto levels: 2..98th percentile -> 0.03..0.97, stretch capped, then contrast about 0.5
        flat_l = Lb.flatten(1)
        lo = torch.quantile(flat_l, 0.02, dim=1).view(-1, 1, 1, 1)
        hi = torch.quantile(flat_l, 0.98, dim=1).view(-1, 1, 1, 1)
        k = (0.94 / (hi - lo).clamp(min=1e-3)).clamp(max=cfg['levels'])
        mid = (lo + hi) / 2
        Ls = 0.5 + (Lb - mid) * k
        Lt = (0.5 + (Ls - 0.5) * cfg.get('contrast', 1.0) + cfg.get('lift', 0.0)).clamp(0.02, 0.98)
    else:
        Lt = (m + (Lb - m) * cfg.get('contrast', 1.0) + cfg.get('lift', 0.0)).clamp(0.02, 0.98)
    if 'paper' in cfg:  # line art: non-edge goes to paper white
        Lt = Lt * (1 - cfg['paper_mix']) + cfg['paper'] * cfg['paper_mix']
    out['tone'] = (gauss(L, 4) - gauss(Lt, 4)).abs().mean()
    # chroma: same hue as base, scaled by `sat`
    out['chroma'] = (gauss(chroma(y), 2) - cfg.get('sat', 1.0) * gauss(chroma(b), 2)).abs().mean()
    # clipping / dead image guards
    out['clip'] = (hinge(y - 1) + hinge(-y)).mean()
    _, sd = tone_stats(y)
    out['spread'] = hinge(cfg.get('min_std', 0.12) - sd).mean()
    return out


def total(parts, cfg):
    return sum(cfg['w'].get(k, 0) * v for k, v in parts.items())


def batch(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().permute(0, 3, 1, 2) / 255


def packet_param_names(p, style):
    gn, bn, fn = norm_names(style)
    names = [gn, bn] + [n for n in p if n.startswith(fn + '.')]
    names += [n for n in guide_affine_names(style) if n in p]
    return names


def bias_penalty(p, style, margin=0.9):
    """Hinge on the folded int16 guide bias code, `margin` of full scale."""
    lg, _ = guide_affine_names(style)
    if lg not in p:
        return torch.zeros(())
    code = guide_bias_codes(p, style) / 32767
    return (F.relu(code - margin) ** 2).sum() + 1e-3 * (p[lg] ** 2).sum()


def optimise(p, cfg, names, steps=200, lr=0.03, bs=6, style=0, seed=0, log=50):
    torch.set_grad_enabled(True)
    pool = coco.load('coco_train')
    rng = np.random.default_rng(seed)
    q = dict(p)
    for n in names:
        q[n] = p[n].clone().requires_grad_(True)
    opt = torch.optim.Adam([q[n] for n in names], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr * 0.1)
    t = time.time()
    for it in range(steps):
        x = batch(pool[rng.choice(len(pool), bs, replace=False)])
        with torch.no_grad():
            b = forward(p, x, 0) if 'ref' not in cfg else cfg['ref'](x)
            E = edge_map(x)
        y = forward(q, x, style)
        parts = losses(y, x, b, E, cfg)
        parts['bias'] = bias_penalty(q, style)
        loss = total(parts, cfg) + 100 * parts['bias']
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % log == 0 or it == steps - 1:
            print(f'  {it:4d} {loss.item():.4f} ' + ' '.join(f'{k}={v.item():.4f}' for k, v in parts.items()), f'{time.time()-t:.0f}s', flush=True)
    torch.set_grad_enabled(False)
    return {n: (q[n].detach() if n in names else v) for n, v in q.items()}


STYLES = {
    # 风格 0: whitebox with colour pop, detail kept
    'pop': dict(levels=1.8, contrast=1.15, lift=0.02, sat=1.5, edge=1.0, min_std=0.18,
                w=dict(flat=2.0, edge=2.0, line=0.0, tone=5.0, chroma=5.0, clip=10, spread=5)),
    # 动漫平涂: flat interiors, edges kept, brighter and more saturated than base
    'anime': dict(contrast=1.25, lift=0.03, sat=1.5, edge=1.0, min_std=0.16,
                  w=dict(flat=10.0, edge=1.0, line=0.0, tone=4.0, chroma=4.0, clip=10, spread=5)),
    # 线稿: dark lines on photo edges, paper-light interiors, faint colour
    'lineart': dict(contrast=0.6, lift=0.0, paper=0.9, paper_mix=0.7, sat=0.35, edge=0.6, min_std=0.08,
                    w=dict(flat=10.0, edge=0.0, line=30.0, tone=4.0, chroma=3.0, clip=10, spread=2)),
    # 漫画 (slot 3, was poster): flat vivid colour plus ink outlines
    'poster': dict(levels=1.6, contrast=1.3, lift=0.0, sat=1.9, edge=1.0, min_std=0.2,
                   w=dict(flat=14.0, edge=1.0, line=20.0, tone=3.0, chroma=3.0, clip=10, spread=5)),
    # three-style set (2026-11-03): bright flat / line art / comic
    # 明亮平涂: anime flatness with auto levels and more colour
    'bright': dict(levels=1.8, contrast=1.15, lift=0.03, sat=1.6, edge=1.0, min_std=0.18,
                   w=dict(flat=10.0, edge=1.0, line=0.0, tone=5.0, chroma=4.0, clip=10, spread=5)),
    # 线稿 v2: whiter paper, stronger flat term against the blotchy ground
    'lineart2': dict(contrast=0.5, lift=0.0, paper=0.93, paper_mix=0.85, sat=0.35, edge=0.6, min_std=0.08,
                     w=dict(flat=16.0, edge=0.0, line=30.0, tone=5.0, chroma=3.0, clip=10, spread=2)),
}
STYLES['comic'] = STYLES['poster']
# 水粉 gouache: replaces line art in slot 1. Pastel tone lifted toward paper, moderate
# colour, flat strokes, faint outlines only.
STYLES['gouache'] = dict(levels=1.4, contrast=0.85, lift=0.08, paper=0.85, paper_mix=0.3, sat=1.0,
                         edge=0.7, min_std=0.12,
                         w=dict(flat=14.0, edge=1.0, line=4.0, tone=5.0, chroma=4.0, clip=10, spread=4))
