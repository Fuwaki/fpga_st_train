"""Style slots on the shared trunk, each one a packet: LN + H/2 FiLM + guide affine.

Slot layout mirrors a `pack-styles --styles N --packet-guide` checkpoint:
style 0 = top-level trunk_norm, film_h2, guide_affine; style k = style_packets.{k-1}.

    train_styles.py <tag> <steps> [only,names] [slots] [init]

`slots` is a comma list of STYLES names (default the four-slot t1 layout). `init` is a
comma list of `name=run/styleK.pt` packets to start from (remapped onto the new slot).
Writes params to runs/astra2/train/<tag>/style{k}.pt, a sheet and metrics.
"""
import sys, json
sys.path.insert(0, 'tools/astra/torchnet')
from styles import *
from sheet import contact
from metrics import simplify

tag = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
only = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] not in ('', 'all') else None
SLOTS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['pop', 'anime', 'lineart', 'poster']
INIT = dict(kv.split('=') for kv in sys.argv[5].split(',')) if len(sys.argv) > 5 and sys.argv[5] else {}
# Per-style affines and their widths. Default is guide only, the one Rust has;
# ASTRA_AFFINES=guide,head,out adds the torch-only head/out experiments (section 11).
import os
AFFINES = {w: c for w, c in {'guide_affine': 8, 'head_affine': 8, 'out_affine': 3}.items()
           if w.split('_')[0] in os.environ.get('ASTRA_AFFINES', 'guide').split(',')}


def packet_suffixes(style):
    """Map full param names of one slot to slot-free suffixes."""
    pre = '' if style == 0 else f'style_packets.{style-1}.'
    return pre


def load_packet(path):
    """A saved styleK.pt -> {slot-free suffix: tensor}."""
    d = torch.load(path)
    k = int(Path(path).stem.replace('style', ''))
    pre = packet_suffixes(k)
    return {n[len(pre):]: v for n, v in d.items() if n.startswith(pre) and (pre or not n.startswith('style_packets.'))}
out = Path('runs/astra2/train') / tag; out.mkdir(parents=True, exist_ok=True)

base = load_params(Checkpoint.open(SRC))
p = {n: v for n, v in base.items() if not n.startswith('style_packets.')}
for k, name in enumerate(SLOTS):
    src = {}
    init = Path('runs/astra2/mine/pk1') / f'{name}.pt'
    if name in INIT:
        src = load_packet(Path('runs/astra2/train') / INIT[name])
    elif init.exists():
        src = torch.load(init)
    gn, bn, fn = norm_names(k)
    for a in ['trunk_norm.norm.affine.gamma', 'trunk_norm.norm.affine.beta'] + [n for n in base if n.startswith('film_h2.')]:
        dest = a if k == 0 else f'style_packets.{k-1}.{a}'
        p[dest] = src.get(a, base[a]).clone()
    for what, c in AFFINES.items():
        lg, gb = affine_names(k, what)
        p[lg] = src.get(f'{what}.log_gain', torch.zeros(c)).clone()
        p[gb] = src.get(f'{what}.bias', torch.zeros(c)).clone()
    if os.environ.get('ASTRA_MIX'):
        p[('' if k == 0 else f'style_packets.{k-1}.') + 'out_mix'] = src.get('out_mix', torch.eye(3)).clone()

for k, name in enumerate(SLOTS):
    if only and name not in only:
        prev = out / f'style{k}.pt'
        if prev.exists():
            p.update(torch.load(prev))
        continue
    print(f'== style {k} {name}', flush=True)
    names = packet_param_names(p, k)
    # The base render is the source checkpoint's style 0, not slot 0 of `p`: before
    # 2026-11-03 it was `p`, so every slot chased the already-boosted slot 0 (yellow drift).
    q = optimise(p, STYLES[name], names, steps=steps, lr=0.01, style=k, seed=k, ref=base)
    for n in names:
        p[n] = q[n]
    torch.save({n: p[n] for n in names}, out / f'style{k}.pt')

val = coco.load('coco_val')
report = {}
Xv = batch(val[[3, 7, 11, 20, 30, 41]])
demo = torch.cat([F.interpolate(read(SRC + f'content_{n}.png'), size=(240, 320), mode='area') for n in ['cat', 'landscape']])
tiles = [(f'photo {i}', to_np(x[None])) for i, x in enumerate(torch.cat([demo, Xv]))]
for k, name in enumerate(SLOTS):
    ys = forward(p, batch(val[:24]), k).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    ms = [simplify(y, ph) for y, ph in zip(ys, val[:24] / 255.0)]
    report[name] = {m: float(np.mean([r[m] for r in ms])) for m in ms[0]}
    # colour without the cast: mean |chroma| (photo 0.048), and (R+G)/2-B over the photo
    yt, xt = torch.from_numpy(ys).permute(0, 3, 1, 2), batch(val[:24])
    report[name]['chroma_abs'] = chroma(yt).abs().mean().item()
    yel = lambda t: ((t[:, 0] + t[:, 1]) / 2 - t[:, 2]).mean().item()
    report[name]['yellow'] = yel(yt) - yel(xt)
    for what in AFFINES:
        lg, gb = affine_names(k, what)
        report[name][what + '_gain'] = torch.exp(p[lg]).numpy().round(3).tolist()
        report[name][what + '_bias'] = p[gb].numpy().round(3).tolist()
        report[name][what + '_code_max'] = float(affine_bias_codes(p, k, what).max())
    print(name, {m: (round(v, 3) if isinstance(v, float) else v) for m, v in report[name].items()}, flush=True)
    tiles += [(f'{k}:{name} {i}', to_np(forward(p, x[None], k))) for i, x in enumerate(torch.cat([demo, Xv]))]
json.dump(report, open(out / 'report.json', 'w'), indent=1)
contact(tiles, 8, out / 'sheet.png', 0.6)
torch.save(p, out / 'params.pt')
