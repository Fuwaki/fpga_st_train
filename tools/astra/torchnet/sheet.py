from PIL import Image, ImageDraw
import numpy as np


def contact(tiles, cols, path, scale=0.5):
    """tiles: list of (label, HxWx3 float array)."""
    h, w = tiles[0][1].shape[:2]
    h2, w2 = int(h * scale), int(w * scale)
    rows = -(-len(tiles) // cols)
    s = Image.new('RGB', (w2 * cols, (h2 + 16) * rows), 'white')
    d = ImageDraw.Draw(s)
    for i, (lab, a) in enumerate(tiles):
        x, y = i % cols * w2, i // cols * (h2 + 16)
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + .5).astype(np.uint8)).resize((w2, h2), Image.LANCZOS)
        s.paste(im, (x, y + 16)); d.text((x + 3, y + 2), lab, fill='black')
    s.save(path)
