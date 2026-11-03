"""PyTorch CPU port of tiny_dw_guided (trunk_at=h4, ln, film_h2, h2_guide, full=0).

Mirrors src/student/guided.rs render_tapped. Parameters are loaded from the
Checkpoint as torch tensors so edits (or gradients) are cheap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weights import Checkpoint  # noqa: E402

torch.set_grad_enabled(False)


def load_params(ck: Checkpoint) -> dict[str, torch.Tensor]:
    return {n: torch.from_numpy(ck.get(n)) for n in ck.names()}


def write_params(ck: Checkpoint, p: dict[str, torch.Tensor], only_changed_from=None):
    for n, v in p.items():
        a = v.detach().cpu().numpy().astype("<f4")
        if only_changed_from is not None and np.array_equal(a, only_changed_from[n].numpy()):
            continue
        ck.set(n, a)


def rpad(x, k):
    pad = k // 2
    return F.pad(x, (pad, pad, pad, pad), mode="reflect") if pad else x


def conv(p, name, x, stride=1):
    w = p[f"{name}.conv.weight"]
    b = p.get(f"{name}.conv.bias")
    return F.conv2d(rpad(x, w.shape[-1]), w, b, stride=stride)


def dw(p, name, x, stride=1):
    w = p[f"{name}.conv.weight"]
    return F.conv2d(rpad(x, w.shape[-1]), w, p[f"{name}.conv.bias"], stride=stride, groups=w.shape[0])


def pw(p, name, x):
    return F.conv2d(x, p[f"{name}.conv.weight"], p[f"{name}.conv.bias"])


def sep(p, name, x, stride=1, out_relu=True):
    y = pw(p, f"{name}.pw", F.relu(dw(p, f"{name}.dw", x, stride)))
    return F.relu(y) if out_relu else y


def norm_names(style):
    pre = "" if style == 0 else f"style_packets.{style-1}."
    return pre + "trunk_norm.norm.affine.gamma", pre + "trunk_norm.norm.affine.beta", pre + "film_h2"


def guide_affine_names(style):
    """`packet_guide`: x * exp(log_gain) + bias on the guide conv, before its ReLU."""
    pre = "" if style == 0 else f"style_packets.{style-1}."
    return pre + "guide_affine.log_gain", pre + "guide_affine.bias"


# Requant ROM: folded guide bias code |b_conv*g + b| / (s_in * s_w * g) must fit int16.
# s_in from the exporter's ch5 code 16242 on the packets checkpoint (1/s_in ~ 147.7),
# s_w = max|W_c| / 127. Limit on |b_conv + b/g| is BIAS_LIMIT_RATIO * max|W_c|.
BIAS_LIMIT_RATIO = 32767 * 0.006769571918994188 / 127


def guide_bias_codes(p, style):
    """Folded int16 bias code per guide channel for one style (float, unsigned)."""
    w, b = p["guide.conv.weight"], p["guide.conv.bias"]
    mx = w.abs().flatten(1).max(1).values
    eff = b.clone()
    lg, gb = guide_affine_names(style)
    if lg in p:
        eff = b + p[gb] / torch.exp(p[lg])
    return eff.abs() / (mx * BIAS_LIMIT_RATIO) * 32767


def layer_norm(x, g, b, eps=1e-5):
    c = x - x.mean(1, keepdim=True)
    v = (c * c).mean(1, keepdim=True)
    return c / torch.sqrt(v + eps) * g.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)


def film(p, stem, x, tap=None):
    z = F.relu(pw(p, f"{stem}.reduce", x.mean((2, 3), keepdim=True)))
    gamma = pw(p, f"{stem}.gamma", z) + 1.0
    beta = pw(p, f"{stem}.beta", z)
    if tap is not None:
        tap("film_z", z); tap("film_gamma", gamma); tap("film_beta", beta)
    return x * gamma + beta


def up2(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def forward(p, x, style=0, tap=None, norm_fn=None, film_fn=None):
    """x: (N,3,H,W) in [0,1]. Returns unclamped RGB."""
    t = tap or (lambda *_: None)
    n, _, h0, w0 = x.shape
    hp, wp = -(-h0 // 8) * 8, -(-w0 // 8) * 8
    if (hp, wp) != (h0, w0):
        x = F.pad(x, (0, wp - w0, 0, hp - h0), mode="reflect")
    low = F.avg_pool2d(x, 2, 2)
    f = F.relu(conv(p, "stem", low)); t("stem", f)
    f = sep(p, "enc0", f, 2); t("enc0", f)
    f = sep(p, "enc1", f, 1); t("enc1", f)
    f = f + sep(p, "res.0.s2", sep(p, "res.0.s1", f), out_relu=False); t("res", f)
    gn, bn, fn = norm_names(style)
    f = (norm_fn or layer_norm)(f, p[gn], p[bn]) if norm_fn is None else norm_fn(f, p[gn], p[bn], style)
    t("trunk_norm", f)
    f = sep(p, "dec_h4", f); t("dec_h4", f)
    f = sep(p, "dec_h2", up2(f)); t("dec_h2", f)
    e = F.relu(conv(p, "h2_guide", low))
    e = pw(p, "h2_guide_proj", e) if "h2_guide_proj.conv.weight" in p else e
    t("h2_guide", e)
    f = f + e
    f = film(p, fn, f, tap) if film_fn is None else film_fn(p, fn, f)
    t("film_h2", f)
    f = F.relu(pw(p, "to_guide", f)); t("paint", f)
    f = up2(f)
    g = conv(p, "guide", x)
    lg, gb = guide_affine_names(style)
    if lg in p:
        g = g * torch.exp(p[lg]).view(1, -1, 1, 1) + p[gb].view(1, -1, 1, 1)
    g = F.relu(g); t("guide", g)
    f = f + g; t("mix", f)
    y = pw(p, "head_sep.pw", F.relu(dw(p, "head_sep.dw", f)))
    y = y[:, :, :h0, :w0]
    t("rgb", y)
    return y


def read(path) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].contiguous()


def save(y: torch.Tensor, path):
    a = (y[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def to_np(y):
    return y[0].clamp(0, 1).permute(1, 2, 0).numpy()
