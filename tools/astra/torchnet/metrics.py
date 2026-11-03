"""Image measures for 'is it a painting or a filter'. All inputs HxWx3 float in [0,1]."""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, sobel

LUMA = np.array([0.2126, 0.7152, 0.0722])


def affine_fit(src, dst):
    """Per-channel least squares dst ~ a*src+b. Returns residual MAE and explained share."""
    res = np.empty_like(dst)
    for c in range(3):
        A = np.stack([src[..., c].ravel(), np.ones(src[..., c].size)], 1)
        coef, *_ = np.linalg.lstsq(A, dst[..., c].ravel(), rcond=None)
        res[..., c] = dst[..., c] - (A @ coef).reshape(dst.shape[:2])
    raw = np.abs(dst - src).mean()
    r = np.abs(res).mean()
    return dict(raw=float(raw), resid=float(r), explained=float(1 - r / max(raw, 1e-9)))


def full_affine_fit(src, dst):
    """3x3 colour matrix + bias. Stronger filter test (cross-channel mixing allowed)."""
    A = np.concatenate([src.reshape(-1, 3), np.ones((src.shape[0] * src.shape[1], 1))], 1)
    coef, *_ = np.linalg.lstsq(A, dst.reshape(-1, 3), rcond=None)
    r = np.abs(dst.reshape(-1, 3) - A @ coef).mean()
    return float(r)


def grad(l):
    return np.hypot(sobel(l, 0), sobel(l, 1)) / 8


def masks(photo):
    l = gaussian_filter(photo @ LUMA, 1)
    g = grad(l)
    strong = g > np.quantile(g, 0.9)
    flat = maximum_filter(g, 5) < np.quantile(g, 0.35)
    b = np.zeros_like(g, bool); b[5:-5, 5:-5] = True
    return strong & b, flat & b


def measure(y, photo):
    strong, flat = masks(photo)
    l = y @ LUMA
    hf = np.abs(l - gaussian_filter(l, 1))
    g = grad(l)
    hsv_s = (y.max(2) - y.min(2)) / (y.max(2) + 1e-6)
    return dict(
        luma=float(l.mean()), luma_std=float(l.std()),
        p5=float(np.quantile(l, .05)), p95=float(np.quantile(l, .95)),
        sat=float(hsv_s.mean()), chroma=float((y.max(2) - y.min(2)).mean()),
        rb=float((y[..., 0] - y[..., 2]).mean()),
        flat_hf=float(hf[flat].mean()), edge=float(g[strong].mean()),
        affine_vs_photo=affine_fit(photo, y), color3x3_resid=full_affine_fit(photo, y),
    )


def kmeans_resid(y, k=6, iters=8, n=4000, seed=0):
    """RGB error after quantising to k colours, relative to image colour std."""
    rng = np.random.default_rng(seed)
    pts = y.reshape(-1, 3)
    sub = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
    c = sub[rng.choice(len(sub), k, replace=False)]
    for _ in range(iters):
        a = ((sub[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([sub[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    d = np.sqrt(((sub[:, None] - c[None]) ** 2).sum(-1).min(1)).mean()
    return float(d / (sub.std(0).mean() + 1e-6))


def simplify(y, photo):
    """Scores for line art / flat paint / colour blocks. Higher line, flat, block = more simplified."""
    strong, flat = masks(photo)
    l = y @ LUMA
    pl = photo @ LUMA
    local = gaussian_filter(l, 3)
    line = float((local - l)[strong].mean())                       # dark strokes on edges
    hf = np.abs(l - gaussian_filter(l, 1.5))
    phf = np.abs(pl - gaussian_filter(pl, 1.5))
    nonedge = ~(maximum_filter(strong, 3))
    flatr = float(hf[nonedge].mean() / (phf[nonedge].mean() + 1e-6))  # <1 = flatter than photo
    g = grad(l)
    edge_keep = float(g[strong].mean() / (grad(pl)[strong].mean() + 1e-6))
    clip = float(((y > 0.98) | (y < 0.02)).all(2).mean())
    return dict(line=line, flat=1 - flatr, edge_keep=edge_keep, block=1 - kmeans_resid(y),
                clip=clip, luma=float(l.mean()), lstd=float(l.std()),
                sat=float(((y.max(2) - y.min(2)) / (y.max(2) + 1e-6)).mean()),
                c3=full_affine_fit(photo, y))
