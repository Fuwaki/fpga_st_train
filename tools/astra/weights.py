#!/usr/bin/env python3
"""Read and write a guided-student checkpoint for weight-space programming.

The file is Burn's named msgpack (`model.mpk`). Each parameter keeps its id.
Only the f32 bytes change, so the Rust loader accepts the result.

Open a model directory (it contains `model.json` and `model.mpk`) or the mpk
itself. Names match the module path with the `item.` prefix removed.

    from weights import Checkpoint

    model = Checkpoint.open("experiments/tiny_dw_whitebox_sky_480p/init")
    model.catalog()
    gamma = model.packet_gamma(0)          # style 0, shape (80,)
    gamma = gamma.copy()
    gamma[3] *= 0.5
    model.set_packet_gamma(0, gamma)
    model.save("runs/astra_edit")          # writes model.mpk and copies model.json
    model.forward("content.png", "out", style=0)

Style 0 lives in `trunk_norm` and `film_h2`. Style k>0 lives in
`style_packets.{k-1}`. Convolution weights stay on the shared trunk.

`forward` shells out to `fpga_st_train infer --style N`. Set `FPGA_ST_TRAIN`
to the binary and `FPGA_ST_DEVICE` to its device (a CPU build uses `cpu`).

Requires `numpy` and `msgpack`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import msgpack
except ImportError as err:  # pragma: no cover
    raise SystemExit("pip install msgpack") from err


_DTYPE = "F32"


class Checkpoint:
    """One named-mpk checkpoint. Edits stay in memory until `save`."""

    def __init__(self, root: dict, path: Path, card: Path | None):
        self._root = root
        item = root.get("item")
        if not isinstance(item, dict):
            raise ValueError(f"{path} has no item map")
        self._item = item
        self.path = path
        self.card = card
        self._params = _index(item)
        self.edits: list[str] = []

    @classmethod
    def open(cls, path: str | Path) -> Checkpoint:
        path = Path(path)
        if path.is_dir():
            mpk = path / "model.mpk"
            card = path / "model.json"
            if not card.is_file():
                card = None
        else:
            mpk = path
            sibling = path.with_name("model.json")
            card = sibling if sibling.is_file() else None
        if not mpk.is_file():
            raise FileNotFoundError(mpk)
        root = msgpack.unpackb(mpk.read_bytes(), raw=False, strict_map_key=False)
        if not isinstance(root, dict):
            raise ValueError(f"{mpk} is not a msgpack map")
        return cls(root, mpk, card)

    def names(self) -> list[str]:
        return sorted(self._params)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(v) for v in self._params[_resolve(self._params, name)]["param"]["shape"])

    def get(self, name: str) -> np.ndarray:
        node = self._params[_resolve(self._params, name)]["param"]
        dtype = node.get("dtype")
        if dtype != _DTYPE:
            raise TypeError(f"{name} is {dtype}, expected {_DTYPE}")
        array = np.frombuffer(node["bytes"], dtype="<f4").copy()
        return array.reshape(self.shape(name))

    def set(self, name: str, value: np.ndarray) -> None:
        key = _resolve(self._params, name)
        node = self._params[key]["param"]
        value = np.asarray(value, dtype="<f4")
        shape = self.shape(key)
        if tuple(value.shape) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {tuple(value.shape)}")
        node["bytes"] = value.tobytes()
        node["dtype"] = _DTYPE
        self.edits.append(key)

    def catalog(self) -> str:
        lines = [f"{'name':<62} {'shape':<18} {'rms':>10}  group"]
        for name in self.names():
            array = self.get(name)
            lines.append(
                f"{name:<62} {str(tuple(array.shape)):<18} {_rms(array):>10.4f}  {_group(name)}"
            )
        return "\n".join(lines)

    def kernels(self, name: str) -> str:
        """Print a conv weight as one 2-D slice per input/output pair."""
        array = self.get(name)
        if array.ndim == 1:
            return np.array2string(array, precision=4, suppress_small=True)
        if array.ndim == 4 and array.shape[-1] <= 7 and array.shape[-2] <= 7:
            lines = [f"{name} {tuple(array.shape)}"]
            out_c, in_c, kh, kw = array.shape
            for oc in range(out_c):
                for ic in range(in_c):
                    lines.append(f"out {oc} in {ic}")
                    lines.append(np.array2string(array[oc, ic], precision=3, suppress_small=True))
            return "\n".join(lines)
        return f"{name} {tuple(array.shape)}\n" + np.array2string(
            array.reshape(array.shape[0], -1)[:12], precision=4, suppress_small=True
        )

    def packet_gamma(self, style: int) -> np.ndarray:
        return self.get(_packet_name(style, "trunk_norm.norm.affine.gamma"))

    def packet_beta(self, style: int) -> np.ndarray:
        return self.get(_packet_name(style, "trunk_norm.norm.affine.beta"))

    def set_packet_gamma(self, style: int, value: np.ndarray) -> None:
        self.set(_packet_name(style, "trunk_norm.norm.affine.gamma"), value)

    def set_packet_beta(self, style: int, value: np.ndarray) -> None:
        self.set(_packet_name(style, "trunk_norm.norm.affine.beta"), value)

    def film(self, style: int) -> dict[str, np.ndarray]:
        stem = _packet_name(style, "film_h2")
        return {
            "reduce_weight": self.get(f"{stem}.reduce.conv.weight"),
            "reduce_bias": self.get(f"{stem}.reduce.conv.bias"),
            "gamma_weight": self.get(f"{stem}.gamma.conv.weight"),
            "gamma_bias": self.get(f"{stem}.gamma.conv.bias"),
            "beta_weight": self.get(f"{stem}.beta.conv.weight"),
            "beta_bias": self.get(f"{stem}.beta.conv.bias"),
        }

    def set_film(self, style: int, parts: dict[str, np.ndarray]) -> None:
        stem = _packet_name(style, "film_h2")
        mapping = {
            "reduce_weight": f"{stem}.reduce.conv.weight",
            "reduce_bias": f"{stem}.reduce.conv.bias",
            "gamma_weight": f"{stem}.gamma.conv.weight",
            "gamma_bias": f"{stem}.gamma.conv.bias",
            "beta_weight": f"{stem}.beta.conv.weight",
            "beta_bias": f"{stem}.beta.conv.bias",
        }
        unknown = set(parts) - set(mapping)
        if unknown:
            raise KeyError(f"unknown FiLM parts {sorted(unknown)}")
        for key, name in mapping.items():
            if key in parts:
                self.set(name, parts[key])

    def styles(self) -> int:
        packets = self._item.get("style_packets")
        extra = len(packets) if isinstance(packets, list) else 0
        return extra + 1

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix == ".mpk":
            mpk = path
            card_dir = None
        else:
            path.mkdir(parents=True, exist_ok=True)
            mpk = path / "model.mpk"
            card_dir = path
        mpk.parent.mkdir(parents=True, exist_ok=True)
        mpk.write_bytes(msgpack.packb(self._root, use_bin_type=True))
        if card_dir is not None and self.card is not None:
            shutil.copyfile(self.card, card_dir / "model.json")
        return mpk

    def forward(
        self,
        content: str | Path,
        out: str | Path,
        style: int = 0,
        binary: str | Path | None = None,
        device: str | None = None,
    ) -> str:
        """Run fp32 infer. Saves a temporary directory when this object has unsaved edits."""
        binary = Path(binary or os.environ.get("FPGA_ST_TRAIN") or _default_binary())
        device = device or os.environ.get("FPGA_ST_DEVICE") or "cpu"
        if not binary.is_file():
            raise FileNotFoundError(
                f"fpga_st_train binary not found at {binary}. Set FPGA_ST_TRAIN."
            )
        with tempfile.TemporaryDirectory(prefix="astra-fwd-") as tmp:
            folder = Path(tmp)
            if self.edits or self.card is None:
                if self.card is None:
                    raise FileNotFoundError("forward needs model.json next to the checkpoint")
                self.save(folder)
            else:
                folder = self.path.parent
            cmd = [
                str(binary),
                "--device",
                device,
                "infer",
                "--model",
                str(folder),
                "--content",
                str(content),
                "--out",
                str(out),
                "--style",
                str(style),
                "--mode",
                "fp32",
            ]
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "infer failed")
        return proc.stdout


def _index(node: object, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    if isinstance(node, dict) and _is_param(node):
        if prefix:
            out[prefix] = node
        return out
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _index(value, child, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _index(value, f"{prefix}.{index}", out)
    return out


def _is_param(node: dict) -> bool:
    param = node.get("param")
    return isinstance(node.get("id"), str) and isinstance(param, dict) and "bytes" in param


def _resolve(params: dict, name: str) -> str:
    if name in params:
        return name
    alias = _alias(name)
    if alias in params:
        return alias
    raise KeyError(name)


def _alias(name: str) -> str:
    """style0.gamma -> trunk_norm... ; style1.gamma -> style_packets.0..."""
    if not name.startswith("style") or "." not in name:
        return name
    head, _, rest = name.partition(".")
    if not head[5:].isdigit():
        return name
    style = int(head[5:])
    return _packet_name(style, _short_rest(rest))


def _short_rest(rest: str) -> str:
    shorts = {
        "gamma": "trunk_norm.norm.affine.gamma",
        "beta": "trunk_norm.norm.affine.beta",
    }
    return shorts.get(rest, rest)


def _packet_name(style: int, suffix: str) -> str:
    if style < 0:
        raise ValueError(f"style {style}")
    if style == 0:
        return suffix
    return f"style_packets.{style - 1}.{suffix}"


def _group(name: str) -> str:
    if name.startswith("style_packets") or name.startswith("trunk_norm") or name.startswith("film_h2"):
        return "packet"
    return "shared"


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def _default_binary() -> Path:
    root = Path(__file__).resolve().parents[2]
    release = root / "target" / "release" / "fpga_st_train"
    debug = root / "target" / "debug" / "fpga_st_train"
    return release if release.is_file() else debug


def _cmd_list(args: argparse.Namespace) -> int:
    model = Checkpoint.open(args.checkpoint)
    if args.group == "packet":
        names = [name for name in model.names() if _group(name) == "packet"]
    elif args.group == "shared":
        names = [name for name in model.names() if _group(name) == "shared"]
    else:
        names = model.names()
    print(f"styles {model.styles()}  tensors {len(names)}")
    for name in names:
        array = model.get(name)
        print(f"{name:<62} {str(tuple(array.shape)):<18} rms {_rms(array):.4f}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    model = Checkpoint.open(args.checkpoint)
    print(model.kernels(args.name))
    return 0


def _cmd_dump(args: argparse.Namespace) -> int:
    model = Checkpoint.open(args.checkpoint)
    np.save(args.out, model.get(args.name))
    print(args.out)
    return 0


def _cmd_set(args: argparse.Namespace) -> int:
    model = Checkpoint.open(args.checkpoint)
    model.set(args.name, np.load(args.npy))
    written = model.save(args.out)
    print(written)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read and write a guided-student model.mpk")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_cmd = sub.add_parser("list", help="print tensor names, shapes, and rms")
    list_cmd.add_argument("checkpoint")
    list_cmd.add_argument("--group", choices=("all", "packet", "shared"), default="all")
    list_cmd.set_defaults(func=_cmd_list)

    show = sub.add_parser("show", help="print one tensor, kernels as 2-D slices")
    show.add_argument("checkpoint")
    show.add_argument("name")
    show.set_defaults(func=_cmd_show)

    dump = sub.add_parser("dump", help="write one tensor to a .npy file")
    dump.add_argument("checkpoint")
    dump.add_argument("name")
    dump.add_argument("out")
    dump.set_defaults(func=_cmd_dump)

    put = sub.add_parser("set", help="replace one tensor from a .npy file and save")
    put.add_argument("checkpoint")
    put.add_argument("name")
    put.add_argument("npy")
    put.add_argument("out", help="output .mpk file or model directory")
    put.set_defaults(func=_cmd_set)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
