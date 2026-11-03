"""Cache COCO images as a uint8 array (N,H,W,3) for CPU packet fitting.

Images are resized (short side) and centre-cropped to W x H. Two shards give a
train pool; shard 9 gives a held-out pool.
"""
import io, sys, glob
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CACHE = Path.home() / '.cache/astra'
SNAP = glob.glob(str(Path.home() / '.cache/huggingface/hub/datasets--detection-datasets--coco/snapshots/*/data'))[0]


def build(name, shards, n, w=320, h=240):
    out = []
    for s in shards:
        f = pq.ParquetFile(glob.glob(f'{SNAP}/train-{s:05d}-*.parquet')[0])
        for b in f.iter_batches(batch_size=64, columns=['image']):
            for img in b.column('image').to_pylist():
                im = Image.open(io.BytesIO(img['bytes'])).convert('RGB')
                sc = max(w / im.width, h / im.height)
                im = im.resize((round(im.width * sc), round(im.height * sc)), Image.BICUBIC)
                l, t = (im.width - w) // 2, (im.height - h) // 2
                out.append(np.asarray(im.crop((l, t, l + w, t + h))))
                if len(out) >= n * (shards.index(s) + 1) // len(shards):
                    break
            if len(out) >= n * (shards.index(s) + 1) // len(shards):
                break
    a = np.stack(out)
    np.save(CACHE / f'{name}.npy', a)
    print(name, a.shape)


def load(name):
    return np.load(CACHE / f'{name}.npy')


if __name__ == '__main__':
    build('coco_train', [0, 1], 800)
    build('coco_val', [9], 96)
