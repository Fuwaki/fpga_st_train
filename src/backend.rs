//! Backend and device selection.
//!
//! Default build: burn `cuda` (CubeCL on the NVIDIA driver). `vulkan` is the
//! SPIR-V path on the same CubeCL runtime, kept for an AMD GPU:
//! `--no-default-features --features vulkan`. `wgpu` is the WGSL path.
//! `ndarray` is the CPU oracle: `--no-default-features --features ndarray`.

use burn::tensor::Device;

#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
use burn::backend::Cuda;
#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
pub type InferBackend = Cuda;

#[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "ndarray")))]
use burn::backend::Vulkan;
#[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "ndarray")))]
pub type InferBackend = Vulkan;

#[cfg(all(
    feature = "wgpu",
    not(feature = "cuda"),
    not(feature = "vulkan"),
    not(feature = "ndarray")
))]
use burn::backend::Wgpu;
#[cfg(all(
    feature = "wgpu",
    not(feature = "cuda"),
    not(feature = "vulkan"),
    not(feature = "ndarray")
))]
pub type InferBackend = Wgpu;

#[cfg(all(
    feature = "ndarray",
    not(feature = "cuda"),
    not(feature = "wgpu"),
    not(feature = "vulkan")
))]
use burn::backend::NdArray;
#[cfg(all(
    feature = "ndarray",
    not(feature = "cuda"),
    not(feature = "wgpu"),
    not(feature = "vulkan")
))]
pub type InferBackend = NdArray;

/// Compile-time name of the selected inference backend.
#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
pub const BACKEND_NAME: &str = "cuda (CubeCL)";
#[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "ndarray")))]
pub const BACKEND_NAME: &str = "vulkan (SPIR-V/CubeCL)";
#[cfg(all(
    feature = "wgpu",
    not(feature = "cuda"),
    not(feature = "vulkan"),
    not(feature = "ndarray")
))]
pub const BACKEND_NAME: &str = "wgpu (WGSL/CubeCL)";
#[cfg(all(
    feature = "ndarray",
    not(feature = "cuda"),
    not(feature = "wgpu"),
    not(feature = "vulkan")
))]
pub const BACKEND_NAME: &str = "ndarray (CPU)";

/// Device of [InferBackend].
pub type InferDevice = Device<InferBackend>;

/// Backend used for training (adds reverse-mode autodiff).
///
/// Balanced checkpointing recomputes memory-bound ops instead of storing them.
/// Burn's default stores every one, and a dense LUT-Conv does that for each
/// input tap: at batch 32 and a 192 crop the H/2 map alone is about 22 GB.
pub type TrainBackend = burn::backend::Autodiff<
    InferBackend,
    burn_autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

/// Device of [TrainBackend].
pub type TrainDevice = Device<TrainBackend>;

/// Resolve a device spec into an inference device.
///
/// On the CUDA build, `dN` and `N` are the `nvidia-smi` ordinal. `default` is
/// `d0`. This machine's Tesla V100-SXM2-16GB is `d0`. There is no integrated-GPU or CPU
/// adapter; `iN` and `cpu` are rejected.
///
/// On the vulkan/wgpu build the spec is one of:
///
/// | spec        | meaning                                                     |
/// |-------------|-------------------------------------------------------------|
/// | default     | let wgpu choose the highest-power adapter                   |
/// | d0, d1, ... | the N-th discrete GPU                                        |
/// | i0, i1, ... | the N-th integrated GPU                                      |
/// | cpu         | a software Vulkan adapter, if the driver exposes one        |
/// | 0, 1, ...   | shorthand for d0, d1, ...                                    |
///
/// When both a Vega and the V100 are present, wgpu enumerates the Vega as `d0`
/// and the V100 as `d1`. That is not the CUDA numbering.
///
/// `cpu` on vulkan/wgpu is not the ndarray backend. For a real CPU run, build
/// with `--no-default-features --features ndarray`.
#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
pub fn select_device(spec: &str) -> InferDevice {
    use burn::backend::cuda::CudaDevice;

    let spec = spec.trim();
    let index = match spec {
        "" | "default" => 0,
        "cpu" => panic!(
            "cuda backend has no cpu device; rebuild with --no-default-features --features ndarray"
        ),
        s if s.starts_with('i') => panic!(
            "cuda backend has no integrated adapter {s:?}; use d0, d1, ... (nvidia-smi index)"
        ),
        s => {
            let rest = s.strip_prefix('d').unwrap_or(s);
            rest.parse::<usize>()
                .unwrap_or_else(|_| panic!("cuda device spec {s:?} is not dN or N"))
        }
    };
    let device = CudaDevice::new(index);
    log::info!("backend {BACKEND_NAME}, device spec {spec:?} -> {device:?}");
    device
}

#[cfg(all(any(feature = "wgpu", feature = "vulkan"), not(feature = "cuda")))]
pub fn select_device(spec: &str) -> InferDevice {
    use burn::backend::wgpu::WgpuDevice;

    let spec = spec.trim();
    let device = match spec {
        "" | "default" => WgpuDevice::DefaultDevice,
        "cpu" => WgpuDevice::Cpu,
        s => {
            let (kind, rest) = s.split_at(1);
            let index = rest.parse::<usize>().unwrap_or(0);
            match kind {
                "d" => WgpuDevice::DiscreteGpu(index),
                "i" => WgpuDevice::IntegratedGpu(index),
                _ => WgpuDevice::DiscreteGpu(s.parse::<usize>().unwrap_or(0)),
            }
        }
    };
    log::info!("backend {BACKEND_NAME}, device spec {spec:?} -> {device:?}");
    device
}

/// With a CPU-only build the spec is irrelevant.
#[cfg(all(
    feature = "ndarray",
    not(feature = "cuda"),
    not(feature = "wgpu"),
    not(feature = "vulkan")
))]
pub fn select_device(_spec: &str) -> InferDevice {
    log::info!("backend {BACKEND_NAME}");
    Default::default()
}
