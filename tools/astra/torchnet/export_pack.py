"""Write trained packet params into a pack-styles checkpoint and check Rust parity.

    python export_pack.py runs/astra2/train/t1/params.pt runs/astra2/pack4 runs/astra2/final_t1
"""
import sys, os, json, subprocess, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from net import *

params, pack, out = sys.argv[1:4]
binary = os.environ.get('FPGA_ST_TRAIN', str(Path.home() / '.cache/wdiag/target-astra/release/fpga_st_train'))
SRC = 'experiments/tiny_dw_whitebox_sky_480p/trials/packets/'
p = torch.load(params)
ck = Checkpoint.open(pack)
base = load_params(ck)
changed = {}
for n in ck.names():
    if n not in p:
        raise KeyError(f'{n} missing from trained params')
    a = p[n].detach().numpy().astype('<f4')
    if not np.array_equal(a, base[n].numpy()):
        ck.set(n, a)
        d = a - base[n].numpy()
        changed[n] = dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                          rel_l2=float(np.linalg.norm(d) / (np.linalg.norm(base[n].numpy()) + 1e-12)))
shared = [n for n in changed if not (n.startswith(('trunk_norm', 'film_h2', 'guide_affine', 'style_packets')))]
assert not shared, f'shared weights changed: {shared}'
ck.save(out)
q = load_params(Checkpoint.open(out))
report = dict(changed=changed, parity={}, bias_codes={})
for style in range(len([n for n in q if n.endswith("guide_affine.bias")])):
    report['bias_codes'][style] = guide_bias_codes(q, style).numpy().round(0).tolist()
    for im in ['cat', 'landscape']:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([binary, '--device', 'cpu', 'infer', '--model', out, '--content', SRC + f'content_{im}.png',
                            '--out', tmp, '--style', str(style), '--mode', 'fp32'], check=True, capture_output=True)
            rust = read(Path(tmp) / f'content_{im}.png')
        dest = Path(out) / f'style{style}_{im}.png'
        Image.fromarray((rust[0].permute(1, 2, 0).numpy() * 255 + .5).astype(np.uint8)).save(dest)
        mine = forward(q, read(SRC + f'content_{im}.png'), style).clamp(0, 1)
        d = (mine - rust).abs()[..., :-2, :]  # bottom rows differ by reflect-pad convention
        report['parity'][f'{style}_{im}'] = dict(mae=float(d.mean()), p999=float(torch.quantile(d.flatten()[::7], .999)))
        print(style, im, report['parity'][f'{style}_{im}'], flush=True)
json.dump(report, open(Path(out) / 'export_report.json', 'w'), indent=1)
print('bias codes', report['bias_codes'])
