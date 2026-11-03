"""SVD of d(output image)/d(LN gamma, beta), split into filter vs structure parts.

Filter subspace for an image: per output channel c, span{photo_c, 1} (the a*x+b test),
extended to span{photo_R, photo_G, photo_B, 1} per channel (3x3 colour matrix test).
"""
import sys, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from net import *

torch.set_grad_enabled(True)
SRC = 'experiments/tiny_dw_whitebox_sky_480p/trials/packets/'
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/astra2/lnjac')
OUT.mkdir(parents=True, exist_ok=True)
style = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ims = ['cat', 'landscape', 'water', 'portrait']
p = load_params(Checkpoint.open(SRC))
gn, bn, _ = norm_names(style)
xs = [F.interpolate(read(SRC + f'content_{im}.png'), scale_factor=0.5, mode='area') for im in ims]

def out_all(theta):
    q = dict(p); q[gn] = theta[:80]; q[bn] = theta[80:]
    return torch.cat([forward(q, x, style).flatten() for x in xs])

theta0 = torch.cat([p[gn], p[bn]])
cols = []
for i in range(160):
    e = torch.zeros(160); e[i] = 1
    _, j = torch.func.jvp(out_all, (theta0,), (e,))
    cols.append(j.detach())
J = torch.stack(cols, 1).double()  # (P, 160)

# Filter basis: per image, per channel c: [photo_R, photo_G, photo_B, 1] at that channel's rows.
basis = []
off = 0
for x in xs:
    n = x[0, 0].numel()
    ph = x[0].reshape(3, -1).double()
    for c in range(3):
        for v in [ph[0], ph[1], ph[2], torch.ones(n, dtype=torch.double)]:
            col = torch.zeros(J.shape[0], dtype=torch.double)
            col[off + c * n: off + (c + 1) * n] = v
            basis.append(col)
    off += 3 * n
Q, _ = torch.linalg.qr(torch.stack(basis, 1))

U, S, Vt = torch.linalg.svd(J, full_matrices=False)
filt = ((Q.T @ U) ** 2).sum(0)  # share of each output mode inside filter subspace
energy = S ** 2 / (S ** 2).sum()
print('style', style, 'top singular values / energy / filter-share')
for k in range(16):
    print(f'{k:2d}  s={S[k]:9.2f}  energy={energy[k]:.3f}  filter={filt[k]:.3f}  gamma|beta share={float((Vt[k,:80]**2).sum()):.2f}/{float((Vt[k,80:]**2).sum()):.2f}')
# Same split for gamma-only and beta-only blocks
for name, sl in [('gamma', slice(0, 80)), ('beta', slice(80, 160))]:
    Jb = J[:, sl]
    tot = (Jb ** 2).sum(); inside = ((Q.T @ Jb) ** 2).sum()
    print(f'{name}: total |J|^2={float(tot):.1f}  filter share={float(inside/tot):.3f}')
# structure-only SVD: remove filter projection, then SVD -> knobs that change the drawing
Js = J - Q @ (Q.T @ J)
Us, Ss, Vts = torch.linalg.svd(Js, full_matrices=False)
print('structure-only singular values', [round(float(s), 2) for s in Ss[:12]])
np.savez(OUT / f'svd_style{style}.npz', S=S.numpy(), Vt=Vt.numpy(), filt=filt.numpy(), Ss=Ss.numpy(), Vts=Vts.numpy(), theta0=theta0.detach().numpy())
