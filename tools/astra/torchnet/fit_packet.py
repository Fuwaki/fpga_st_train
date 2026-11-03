"""Fit one style packet (LN gamma/beta + H/2 FiLM) to target(x); shared convs frozen."""
import sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from net import *

torch.set_grad_enabled(True)
SRC = 'experiments/tiny_dw_whitebox_sky_480p/trials/packets/'


def packet_names(p, style):
    gn, bn, fn = norm_names(style)
    return [gn, bn] + [n for n in p if n.startswith(fn + '.')]


def enhance(y, contrast=1.35, sat=1.4, lift=0.04):
    """Luma S-curve around the image mid-grey, then chroma gain. y: (N,3,H,W)."""
    w = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    L = (y * w).sum(1, keepdim=True)
    m = L.mean((2, 3), keepdim=True)
    L2 = (m + (L - m) * contrast + lift).clamp(0, 1)
    s = torch.sigmoid((L2 - 0.5) * 6)  # gentle S on top
    s = (s - torch.sigmoid(torch.tensor(-3.))) / (torch.sigmoid(torch.tensor(3.)) - torch.sigmoid(torch.tensor(-3.)))
    L3 = 0.5 * L2 + 0.5 * s
    c = (y - L) * sat
    return (L3 + c).clamp(0, 1)


def fit(p, style, xs, target_fn, steps=300, lr=0.01, names=None):
    names = names or packet_names(p, style)
    q = dict(p)
    for n in names:
        q[n] = p[n].clone().requires_grad_(True)
    with torch.no_grad():
        ts = [target_fn(x) for x in xs]
    opt = torch.optim.Adam([q[n] for n in names], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        loss = sum(F.l1_loss(forward(q, x, style), t) for x, t in zip(xs, ts)) / len(xs)
        loss.backward(); opt.step()
        if it % 50 == 0 or it == steps - 1:
            print(f'  step {it:4d} l1 {loss.item():.4f}', flush=True)
    return {n: (q[n].detach() if n in names else v) for n, v in q.items()}
