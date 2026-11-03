import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from net import *; from metrics import simplify; import coco
SRC = 'experiments/tiny_dw_whitebox_sky_480p/trials/packets/'
p = load_params(Checkpoint.open(SRC))
S = np.load('runs/astra2/lnjac/coco_svd_s0.npz')
th0 = torch.from_numpy(S['theta0'])
val = coco.load('coco_val')[:24]
X = torch.from_numpy(val).float().permute(0, 3, 1, 2) / 255
photos = val / 255.0


def render(th, x=X):
    q = dict(p); q['trunk_norm.norm.affine.gamma'] = th[:80].float(); q['trunk_norm.norm.affine.beta'] = th[80:].float()
    return forward(q, x, 0).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def score(th):
    ys = render(th)
    ms = [simplify(y, ph) for y, ph in zip(ys, photos)]
    out = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    out.update({k + '_sd': float(np.std([m[k] for m in ms])) for k in ['line', 'flat', 'luma']})
    return out


if __name__ == '__main__':
    rows = [dict(name='base', **score(th0))]
    for kind, V, K in [('s', S['Vs'], 8), ('f', S['Vf'], 6)]:
        for k in range(K):
            for a in (-6, -4, -2, 2, 4, 6):
                rows.append(dict(name=f'{kind}{k}{a:+d}', **score(th0 + a * torch.from_numpy(V[k]).float())))
    json.dump(rows, open('runs/astra2/lnjac/scan_s0.json', 'w'), indent=1)
    keys = ['line', 'flat', 'edge_keep', 'block', 'clip', 'luma', 'lstd', 'sat', 'c3', 'luma_sd']
    print('name    ' + ''.join(f'{k:>9}' for k in keys))
    for r in rows:
        print(f"{r['name']:8}" + ''.join(f'{r[k]:9.3f}' for k in keys))
