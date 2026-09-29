//! Backend and device selection.
//!
//! Default build: `wgpu` (WGSL). `vulkan` is the SPIR-V compiler on the same
//! CubeCL runtime (`--no-default-features --features vulkan`), which can use
//! tensor cores on NVIDIA. `ndarray` is the CPU oracle.

use burn::tensor::Device;

#[cfg(all(feature = "vulkan", not(feature = "ndarray")))]
use burn::backend::Vulkan;
#[cfg(all(feature = "vulkan", not(feature = "ndarray")))]
pub type InferBackend = Vulkan;

#[cfg(all(feature = "wgpu", not(feature = "vulkan"), not(feature = "ndarray")))]
use burn::backend::Wgpu;
#[cfg(all(feature = "wgpu", not(feature = "vulkan"), not(feature = "ndarray")))]
pub type InferBackend = Wgpu;

#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "vulkan")))]
use burn::backend::NdArray;
#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "vulkan")))]
pub type InferBackend = NdArray;

/// Compile-time name of the selected inference backend.
#[cfg(feature = "vulkan")]
pub const BACKEND_NAME: &str = "vulkan (SPIR-V/CubeCL)";
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
pub const BACKEND_NAME: &str = "wgpu (WGSL/CubeCL)";
#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "vulkan")))]
pub const BACKEND_NAME: &str = "ndarray (CPU)";

/// Device of [InferBackend].
pub type InferDevice = Device<InferBackend>;

/// Backend used for training (adds reverse-mode autodiff).
pub type TrainBackend = burn::backend::Autodiff<InferBackend>;

/// Device of [TrainBackend].
pub type TrainDevice = Device<TrainBackend>;

/// Resolve a device spec into an inference device.
///
/// With the wgpu backend the spec is one of:
///
/// | spec        | meaning                                                     |
/// |-------------|-------------------------------------------------------------|
/// | default     | let wgpu choose the highest-power adapter                   |
/// | d0, d1, ... | the N-th discrete GPU (d0 = RX 580 here)                     |
/// | i0, i1, ... | the N-th integrated GPU                                      |
/// | cpu         | the CPU adapter, if wgpu exposes one (see note below)        |
/// | 0, 1, ...   | shorthand for d0, d1, ...                                    |
///
/// `cpu` here is wgpu falling back to a software Vulkan implementation (e.g. lavapipe);
/// it is not installed on this machine and the spec panics. For an actual CPU run, build
/// with `--no-default-features --features ndarray` instead (see [crate::backend] docs).
///
/// With the CPU (ndarray) backend the spec is ignored.
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
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
#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "vulkan")))]
pub fn select_device(_spec: &str) -> InferDevice {
    log::info!("backend {BACKEND_NAME}");
    Default::default()
}
