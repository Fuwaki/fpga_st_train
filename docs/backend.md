# Compute backend

## The rule

Default training backend is burn `cuda` (`default = ["cuda"]` in `Cargo.toml`).
`src/backend.rs` wires that to `InferBackend` / `TrainBackend`. On this machine
the Tesla V100-SXM2-16GB is CUDA ordinal 0, so pass `--device d0`.

Vulkan stays available for an AMD GPU, and wgpu stays available as the WGSL path.
Both are opt-in and are not compiled into the default binary:

```bash
cargo run --release --no-default-features --features cuda -- --device d0 train --config ...
cargo run --release --no-default-features --features vulkan -- --device d0 train --config ...
cargo run --release --no-default-features --features wgpu -- --device d0 train --config ...
cargo run --release --no-default-features --features ndarray -- ...
```

`dN` on a CUDA build is the `nvidia-smi` index. On a vulkan/wgpu build it is the
N-th discrete adapter. When a Vega and the V100 are both present, wgpu calls the
Vega `d0` and the V100 `d1`. CUDA only sees the V100, as `d0`.

## Why CUDA is the default now

Training runs are on the NVIDIA card. CUDA is the CubeCL path that can use the
NVIDIA driver directly, including cooperative-matrix kernels where the GPU has
them. The AMD Vega is no longer the machine these runs are planned around.
Vulkan remains the explicit fallback when a run does have to go there.

## Driver 580, compiler pinned to CUDA 12.8

`nvidia-smi` reports `CUDA Version: 13.0` (driver 580.178.04) on a Tesla
V100-SXM2-16GB, compute 7.0. `/opt/cuda` is toolkit 13.4. Two constraints
meet here:

- 13.4 PTX fails at launch with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The
  driver only accepts PTX up to the 13.0 line.
- CUDA 13 nvrtc's lowest `--gpu-architecture` is `sm_75`. `sm_70` is rejected,
  so that compiler cannot emit a kernel for this V100.

`scripts/cuda_env.sh` therefore loads the CUDA 12.8 nvrtc redistributable at
`/data/competition/2026fpga/cuda-12.8/prefix` and sets `CUDA_PATH` to the
matching headers. `.cargo/config.toml` sets `CUDARC_CUDA_VERSION=12080`.
12.8 still warns that anything before `sm_75` is deprecated, then emits PTX
the driver loads. Launch training through `scripts/train.sh` or
`scripts/study_run.sh`, which source `cuda_env.sh`.

## What the other features are for

ROCm is still not a backend here. The AMD path that was measured correct is
`wgpu` (WGSL). `vulkan` (SPIR-V) is faster on NVIDIA and is the named fallback,
but it is numerically wrong on RADV / the RX 580 (see below). Use `wgpu` if a
run has to land on that card and the pictures have to match the CPU oracle.

`wgpu` needs no CUDA toolkit. The default `cuda` build compiles kernels with
the CUDA 12.8 prefix from `scripts/cuda_env.sh`, not the distro tree at
`/opt/cuda`.

## Measured: burn `vulkan` (SPIR-V/CubeCL) is broken on this machine

burn has a second GPU path, feature `vulkan`, which also goes through the
`wgpu` crate but has CubeCL emit SPIR-V directly (via `cubecl-spirv`) instead
of WGSL, so it can use cooperative-matrix (tensor core) instructions that
WGSL/naga cannot express. It was tried first, since it is the path with a
faster ceiling on hardware with tensor cores. On this machine it silently
produces wrong output on the AMD adapter:

| backend / device | PSNR vs PyTorch reference | cosine similarity | verdict |
|---|---|---|---|
| ndarray (CPU) | 112.8 dB | 1.000000 | PASS |
| wgpu (WGSL), Tesla P4 | 109.2-109.6 dB | 1.000000 | PASS |
| wgpu (WGSL), RX 580 | 109.8 dB | 1.000000 | PASS |
| **vulkan (SPIR-V), RX 580 (default adapter)** | **13.3 dB** | **0.934** | **FAIL** |
| vulkan (SPIR-V), Tesla P4 | 109.2 dB | 1.000000 | PASS (NVIDIA driver is fine) |

Reproduce with the `parity` subcommand: it dumps the Rust forward pass of a
fixed input as a flat `.f32` file. Compare that against a reference dump made
from the original PyTorch MicroAST (`teacher-eval/MicroAST`); PSNR should read
>100 dB and cosine ~1.0.

> **Note (v5 cleanup).** The two helpers that used to automate this
> (`scripts/parity_reference.py` to produce the PyTorch dump and
> `scripts/parity_compare.py` to score it) were deleted, along with the
> checked-in reference dump, when the project was reduced to the three delivery
> paths. The numbers in the table above still hold -- neither the backend nor
> the model has changed since -- but re-validating on a new machine now means
> writing that PyTorch reference dump again from `teacher-eval/`, which is
> where the PyTorch model still lives.

The failure is specific to the RADV/Mesa SPIR-V path
on this AMD card -- almost certainly a CubeCL-SPIR-V-or-driver bug, not
something fixable from this project. `wgpu` (WGSL) does not have this bug on
either GPU in this machine. That is why an AMD run should use `wgpu`, not
`vulkan`, when correctness matters.

## Tensor cores exist only on the cuda and vulkan builds

WGSL has no cooperative-matrix instruction, and naga (the compiler `wgpu`
uses for WGSL) cannot lower to one; `cubecl-wgpu`'s WGSL compiler panics if a
kernel ever tries
(`compiler/wgsl/compiler.rs`: `"Cooperative matrix-multiply and accumulate
isn't supported on wgpu."`). Only the `vulkan` and `cuda` burn features can
reach CubeCL's cmma-based fast kernels. The bullets below are the `wgpu` feature
only; the default `cuda` build can select those cmma kernels. Concretely, on wgpu:

- every convolution -- forward and the weight-gradient backward pass --
  runs through CubeCL naive `DirectConv2dKernel` fp32 path, on any GPU, on
  any machine, as long as the `wgpu` feature is what is compiled in. Moving
  this project to an RTX 4090 later will make it faster (more raw fp32
  throughput, more parallelism) but will not unlock a different, faster
  kernel selection -- there is no autotune candidate for this backend that
  uses tensor cores.
- measured on this machine, `DirectConv2dKernel` was 67% of GPU time in a
  training step (`CUBECL_DEBUG_OPTION=profile-medium`, see below), so
  headroom exists in principle, but recovering it would require bypassing
  burn convolution op (e.g. reformulating as matmul) -- out of scope, and
  not needed.

This is an accepted trade-off, not an oversight: the student network here is
small (16/32/64 channel, a handful of 3x3/1x1 convs), fp32 correctness is
worth more than half-precision tensor-core throughput at this scale, and even
on the weakest GPU in this machine (RX 580, no tensor cores, fp32 only) a
training step is already sub-second. A faster GPU later helps through raw
throughput scaling.

## Before training on a new machine: re-validate, do not assume

`wgpu` being portable means it *compiles and runs* everywhere; it does not
by itself prove correctness on a GPU that has not been measured yet. Before
trusting a long training run on a new machine:

```
cargo run --release -- parity \
    --teacher <dir with the four MicroAST .pth.tar files> \
    --content-raw runs/parity/content.f32 \
    --style-raw runs/parity/style.f32 \
    --out-raw /tmp/parity_out.f32 \
    --device default   # or d0, d1, ... -- try every GPU the machine has

python3 scripts/parity_compare.py \
    --a runs/parity/pytorch_out.f32 \
    --b /tmp/parity_out.f32
```

Expect `PARITY: PASS`, PSNR comfortably above 100 dB, cosine similarity
1.000000. If it fails, stop and investigate before starting any training --
do not assume a passing build means a correct run.

## Known sharp edges

- **`--device cpu` is not the CPU build.** With the `wgpu` feature, `cpu` in
  the device spec asks `wgpu` for a software Vulkan adapter (lavapipe /
  llvmpipe). If the host has none installed (true on this machine), it
  panics (`"No CPU device found"`). For an actual CPU run, build with
  `--no-default-features --features ndarray` instead; `select_device` then
  ignores the spec entirely.
- **First run at a new (resolution, batch size) shape is slow.** CubeCL
  autotunes each distinct tensor shape it sees (compiles candidate kernels,
  benchmarks them), typically adding 1-5 seconds the first time; results are
  cached under `target/autotune/`. Do not read the first iteration timing
  as steady-state, and fix `--content-size` / `data.crop` before timing runs.
- **`stylize` timing must end after the GPU readback, not after
  `model.stylize()` returns.** `wgpu` records tensor ops on an async command
  stream; `stylize()` returning does not mean the GPU has run anything yet.
  `data::tensor_to_rgb()`'s `into_data()` call is what actually blocks on the
  result, so the `Instant` measurement in `main.rs::stylize` was moved to end
  after that call (previously it ended right after `model.stylize()`, so
  every printed per-image time was near-zero and the only real cost showed
  up, misleadingly, as one-time shader-compile stalls on the first image at
  each new resolution).

## If it feels slow again: profile before guessing

```
CUBECL_DEBUG_LOG=/tmp/cubecl.log CUBECL_DEBUG_OPTION=profile-medium \
    RUST_LOG=warn cargo run --release -- train --config <config.json>
```

The end of `/tmp/cubecl.log` has a per-kernel-type table (name, total GPU
time, invocation count, % of total) for the run. That is how the
`DirectConv2dKernel` / 67% figure above was obtained; use the same method
rather than trusting wall-clock prints, which include CPU-side data loading
and (for `stylize`) image encode/decode that have nothing to do with the GPU.

## PCIe / T10 host↔device traffic (2026-10-10)

Tesla T10 on this box often trains at **PCIe Gen1–3 x4** (Vega56 is Gen3 x16). Prefer
`--device d0` with a **cuda** build (`cargo build --release --no-default-features --features cuda`).
That ordinal is `nvidia-smi` index 0, the T10 on this machine. A vulkan/wgpu build
still uses discrete-adapter order: `d0` = Vega, `d1` = T10, when both are installed.
`CUDA_VISIBLE_DEVICES` applies to the CUDA build and does not apply to vulkan/wgpu.

### What hurt on x4

1. **`into_data()` teacher↔student copies** every step (full activation tensors over PCIe).
2. **Per-step `into_scalar()`** logging (GPU sync / tiny readbacks that stall the queue).
3. **Re-decoding PNG/JPEG every crop** on the CPU before each upload.

### Mitigations (in tree)

- `Tensor::from_inner` / `.inner()` for teacher outputs (stay on GPU).
- Defer `into_scalar` to `log_every` boundaries.
- Process-wide decoded-image cache in `data::dataset`.
- Faster CHW pack in `rgb_to_tensor` (still one H2D upload per image).

### Bench (vulkan, d1=T10, tiny_dw, batch4, crop128, 40 steps, no teacher)

| build | wall | cumulative step/s @40 | notes |
|------:|-----:|----------------------:|-------|
| before | 38.6 s | 1.04 | per-step scalar syncs, no image cache |
| after (log_every=10) | 22.1 s | 1.81 | ~1.75× wall |
| after (log_every=40) | 21.9 s | 1.82 | |

Steady-state steps 20→40 stay ~2.3–2.4 step/s once autotune is warm (compute-bound);
the big win is cold start + removing sync/decode overhead. With **online teacher**, avoiding
`into_data` roundtrips is the dominant PCIe saving (full NCHW maps twice per step).

### Roundtrip microbench (`examples/pcie_roundtrip.rs`, 16 MiB f32)

| adapter | into_data↔from_data | on-device add | ratio |
|---------|--------------------:|--------------:|------:|
| d1 T10 (x4) | 29.3 ms (~1.15 GB/s 2-way) | 0.44 ms | 66× |
| d0 Vega56 | 14.3 ms (~2.34 GB/s 2-way) | 0.27 ms | 53× |

Keeping teacher/student tensors on-GPU (`from_inner`) avoids that 29 ms/activation bounce on T10.


## V100 cohort throughput (2026-10-25)

Setup: `b_h4`, batch 32, crop 192, WhiteBox teacher on the Vega, COCO from the HF parquet
cache. Wall time per training step:

| | before | after |
|---|---:|---:|
| 1 student, no teacher (pseudo-GT on disk) | 1.11 s | 0.15 s |
| 6 students, no teacher | 6.75 s | 0.94 s |
| 12 students, no teacher | — | 1.76 s |
| 12 students, online WhiteBox, COCO | 8.7 s | 1.9 s |

Three causes, three fixes:

1. **Data, about 7 s per step.** Each COCO shard is a single 480 MB row group with no page
   index, and every sample re-decoded the image column from row 0. `src/data/hf.rs` now
   packs each shard's JPEG bytes once into `<hf cache>/../fpga_st_train/jpeg-packs/`
   (4.5 GB for 10 shards, about 40 s the first time) and reads one positioned slice per
   sample: 26–69 ms per batch. The bytes match the parquet exactly.
2. **Student backward, 0.92 s of each student's 1.1 s.** CubeCL's `conv2d` backward is
   30–280× its forward for depthwise, 1×1 and few-channel dense convolutions.
   `src/student/conv.rs` expresses those as shifted slices and batched matmuls (same sums,
   about 4e-7 apart). Per-op table in that file.
3. **Teacher, per-image calls.** WhiteBox ran as 32 `session.run`s of batch 1: 1017 ms per
   batch on the Vega, about 25% busy. The batch dimension is now rewritten to be dynamic and
   the batch runs in one call: 381 ms, bit-identical.

After this the step is the students' own GPU time, about 0.15 s each. Twelve students per
cohort fit in 3.7 GB.

Checked and not worth keeping: `NoCheckpointing` instead of `BalancedCheckpointing` (same
time), skipping zero-weight `flat`/`edge`/`contrast` (4%), and moving the teacher to the V100
via ONNX Runtime CUDA (its 33 ms per batch would be under 2% of a step; on sm_70 it also
needs onnxruntime-gpu built against CUDA 12 and cuDNN ≤ 9.10).
