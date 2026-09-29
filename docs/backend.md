# Compute backend: wgpu, and why

## The rule

This project uses exactly one GPU backend: burn `wgpu` feature (WGSL compute
shaders, compiled by naga, dispatched through whatever native API `wgpu` picks
on the host -- Vulkan on Linux, Metal on macOS, DX12 on Windows). This is fixed
project policy, independent of which machine it is built on. `default = ["wgpu"]`
in `Cargo.toml`; `src/backend.rs` wires it to `InferBackend` /
`TrainBackend`.

Do not add or switch to burn other GPU features (`vulkan`, `cuda`, `rocm`,
`metal`) without re-reading this document and re-running the validation below
on the new machine first.

## Why wgpu and nothing else

The project deliberately avoids ROCm (no official/stable AMD GPU support in
most ML frameworks, and this is being developed on an AMD RX 580) and avoids
being locked to a specific vendor SDK in general, since the trained student
model is meant to move to whatever machine is available for long training
runs. `wgpu` is the one backend that:

- needs no vendor SDK to build or run (no ROCm, no CUDA toolkit, just a
  Vulkan/Metal/DX12 driver that is normally already present),
- is, by measurement on this machine, the only GPU path that produces
  numerically correct results (see below),
- is portable across vendors by construction: the same compiled feature
  works on the AMD RX 580 here, on the NVIDIA Tesla P4 also in this machine,
  and should work unmodified on whatever GPU a future training machine has.

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
either GPU in this machine, which is the other reason it is the fixed choice,
not just the portable one.

## No tensor cores, on any GPU, ever -- and that is fine

WGSL has no cooperative-matrix instruction, and naga (the compiler `wgpu`
uses for WGSL) cannot lower to one; `cubecl-wgpu`'s WGSL compiler panics if a
kernel ever tries
(`compiler/wgsl/compiler.rs`: `"Cooperative matrix-multiply and accumulate
isn't supported on wgpu."`). Only the `vulkan` and `cuda` burn features can
reach CubeCL's cmma-based fast kernels. Concretely, this means:

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
throughput scaling, which is the only kind of speedup this backend choice
was ever going to give.

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
