import sys, json
sys.path.insert(0, 'tools/astra/torchnet')
from styles import *
from sheet import contact
from metrics import simplify
tag = sys.argv[1]; which = sys.argv[2].split(','); scope = sys.argv[3]  # ln | packet
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
init = sys.argv[5] if len(sys.argv) > 5 else None
p = load_params(Checkpoint.open(SRC))
LN = ['trunk_norm.norm.affine.gamma', 'trunk_norm.norm.affine.beta']
FILM = [n for n in p if n.startswith('film_h2.')]
out = Path('runs/astra2/mine') / tag; out.mkdir(parents=True, exist_ok=True)
val = coco.load('coco_val')
Xv = batch(val[[3, 7, 11, 20, 30, 41]])
demo = torch.cat([F.interpolate(read(SRC + f'content_{n}.png'), size=(240, 320), mode='area') for n in ['cat', 'landscape']])
tiles = [(f'base {i}', to_np(forward(p, x[None], 0))) for i, x in enumerate(torch.cat([demo, Xv]))]
report = {}
for s in which:
    start = p
    if init:
        start = dict(p); start.update(torch.load(Path('runs/astra2/mine') / init / f'{s}.pt'))
    q = optimise(start, STYLES[s], LN if scope == 'ln' else LN + FILM, steps=steps, lr=0.03 if scope == 'ln' else 0.01)
    torch.save({n: q[n] for n in LN + FILM}, out / f'{s}.pt')
    ys = forward(q, batch(val[:24]), 0).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    ms = [simplify(y, ph) for y, ph in zip(ys, val[:24] / 255.0)]
    report[s] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    report[s]['d_gamma'] = float((q[LN[0]] - p[LN[0]]).norm() / p[LN[0]].norm())
    report[s]['d_beta'] = float((q[LN[1]] - p[LN[1]]).norm())
    print(s, {k: round(v, 3) for k, v in report[s].items()}, flush=True)
    tiles += [(f'{s} {i}', to_np(forward(q, x[None], 0))) for i, x in enumerate(torch.cat([demo, Xv]))]
json.dump(report, open(out / 'report.json', 'w'), indent=1)
contact(tiles, 8, out / 'sheet.png', 0.6)
