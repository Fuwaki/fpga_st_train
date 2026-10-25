//! ONNX distillation teachers on the AMD GPU.
//!
//! Training stays on CUDA. These sessions use ONNX Runtime's WebGPU execution
//! provider, and Dawn is pointed at Vulkan. The process's Vulkan loader is
//! restricted to the RADV ICD before the first session, so the high-performance
//! adapter Dawn requests is the Vega and not the Tesla V100.
//!
//! White-box runs the whole batch in one call. Its released graph pins the batch
//! to 1, so the batch dimension is rewritten to be dynamic first.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail};
use ash::vk;
use burn::tensor::{Tensor, TensorData};
use ndarray::Array4;
use ort::ep::webgpu::{DawnBackendType, WebGPU};
use ort::session::Session;
use ort::value::TensorRef;

use crate::backend::InferBackend;
use crate::config::TeacherKind;
use crate::teacher::DistillTeacher;

const AMD_VENDOR: u32 = 0x1002;

struct OnnxJob {
    nchw: Vec<f32>,
    n: usize,
    h: usize,
    w: usize,
    reply: Sender<Result<Vec<f32>>>,
}

/// Handle for an ONNX teacher whose session lives on one thread.
///
/// Dawn's Vulkan device is created on the first `session.run`. Keeping every
/// run on the thread that opened the session avoids moving that device onto
/// the training thread. The training loop can submit the next batch and keep
/// using CUDA; the reply arrives when Vega finishes.
pub struct OnnxTeacher {
    jobs: Option<Sender<OnnxJob>>,
    model: PathBuf,
    device_label: String,
    worker: Option<JoinHandle<()>>,
}

/// Session state owned by the teacher thread.
struct OnnxSession {
    kind: TeacherKind,
    device_label: String,
    session: Mutex<Session>,
    input_name: String,
    /// The graph accepts any batch size, so a batch runs as one call.
    batched: bool,
}

impl OnnxTeacher {
    pub fn open(kind: TeacherKind, model: &Path) -> Result<Self> {
        let (job_tx, job_rx) = mpsc::channel::<OnnxJob>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let model_buf = model.to_path_buf();
        let worker = thread::Builder::new()
            .name("onnx-teacher".into())
            .spawn(move || match OnnxSession::open(kind, &model_buf) {
                Ok(session) => {
                    let label = session.device_label.clone();
                    let _ = ready_tx.send(Ok(label));
                    for job in job_rx {
                        let result = session.run_nchw(&job.nchw, job.n, job.h, job.w);
                        let _ = job.reply.send(result);
                    }
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                }
            })
            .context("spawn the ONNX teacher thread")?;
        let device_label = ready_rx
            .recv()
            .context("ONNX teacher thread stopped before the session was ready")??;
        Ok(Self {
            jobs: Some(job_tx),
            model: model.to_path_buf(),
            device_label,
            worker: Some(worker),
        })
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    pub fn model(&self) -> &Path {
        &self.model
    }

    /// Block until Vega returns `[N, 3, H, W]` RGB in `[0, 1]`.
    pub fn run_nchw(&self, nchw01: &[f32], n: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        self.submit_nchw(nchw01.to_vec(), n, h, w)?
            .recv()
            .context("ONNX teacher thread stopped")?
    }

    /// Queue one batch and return the reply channel. The caller can do other
    /// work before `recv`. A later `run_nchw` waits behind this job.
    pub fn submit_nchw(
        &self,
        nchw: Vec<f32>,
        n: usize,
        h: usize,
        w: usize,
    ) -> Result<Receiver<Result<Vec<f32>>>> {
        let (reply, rx) = mpsc::channel();
        self.jobs
            .as_ref()
            .context("ONNX teacher is closed")?
            .send(OnnxJob {
                nchw,
                n,
                h,
                w,
                reply,
            })
            .context("ONNX teacher thread stopped")?;
        Ok(rx)
    }
}

impl Drop for OnnxTeacher {
    fn drop(&mut self) {
        // Close the job channel first so the worker leaves `recv`, then join it.
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl OnnxSession {
    fn open(kind: TeacherKind, model: &Path) -> Result<Self> {
        let (device_id, device_label) = amd_vulkan_device()?;
        println!("WebGPU device: {device_label} (vendor {AMD_VENDOR:#06x}, deviceId {device_id})");
        let (session_model, batched) = match kind {
            // Opset-9 Upsample scales are a Shape/Floor subgraph. WebGPU mis-sizes
            // that subgraph (192 -> 190). Constant 2x scales match the CPU graph exactly.
            TeacherKind::RainPrincess => (rain_constant_scales(model)?, false),
            // One call per batch: 381 ms for 32×192² on the Vega against 1017 ms as
            // 32 single-image runs, with bit-identical output.
            TeacherKind::Whitebox => (whitebox_dynamic_batch(model)?, true),
            TeacherKind::MicroAst => (model.to_path_buf(), false),
        };
        let env = ort_env()?;
        let session = Session::builder(env)
            .map_err(ort_anyhow)?
            .with_execution_providers([WebGPU::default()
                .with_dawn_backend_type(DawnBackendType::Vulkan)
                .with_device_id(device_id)
                .build()])
            .map_err(ort_anyhow)?
            .commit_from_file(&session_model)
            .map_err(ort_anyhow)
            .with_context(|| format!("failed to load ONNX teacher {}", model.display()))?;
        let input_name = session.inputs()[0].name().to_owned();
        log::info!(
            "onnx teacher {} loaded from {} (input {input_name})",
            kind.name(),
            model.display()
        );
        Ok(Self {
            kind,
            device_label,
            session: Mutex::new(session),
            input_name,
            batched,
        })
    }

    /// `nchw01` is `[N, 3, H, W]` RGB in `[0, 1]`. The returned buffer has the same shape.
    fn run_nchw(&self, nchw01: &[f32], n: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let plane = h * w;
        let expect = n * 3 * plane;
        if nchw01.len() != expect || n == 0 || h == 0 || w == 0 {
            bail!(
                "onnx input length {} does not match NCHW {n}x3x{h}x{w}",
                nchw01.len()
            );
        }
        match self.kind {
            TeacherKind::RainPrincess => self.run_rain(nchw01, n, h, w),
            TeacherKind::Whitebox => self.run_whitebox(nchw01, n, h, w),
            TeacherKind::MicroAst => bail!("microast is not an ONNX teacher"),
        }
    }

    fn run_rain(&self, nchw01: &[f32], n: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let scaled: Vec<f32> = nchw01.iter().map(|v| v * 255.0).collect();
        let input = Array4::from_shape_vec((n, 3, h, w), scaled)
            .context("rain-princess input is not contiguous NCHW")?;
        let (shape, data) = self.run_session(input.view())?;
        let (on, oc, oh, ow) = match shape.as_slice() {
            [on, oc, oh, ow] => (*on, *oc, *oh, *ow),
            other => bail!("rain-princess output rank/shape {other:?}, expected NCHW"),
        };
        if on != n || oc != 3 || oh < h || ow < w {
            bail!("rain-princess output {on}x{oc}x{oh}x{ow} cannot crop to {n}x3x{h}x{w}");
        }
        let cropped = crop_nchw(&data, n, oh, ow, h, w);
        Ok(cropped
            .into_iter()
            .map(|v| (v / 255.0).clamp(0.0, 1.0))
            .collect())
    }

    fn run_whitebox(&self, nchw01: &[f32], n: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        if !h.is_multiple_of(4) || !w.is_multiple_of(4) {
            bail!("white-box needs H and W to be multiples of 4, got {h}x{w}");
        }
        if self.batched {
            return self.run_whitebox_batch(nchw01, n, h, w);
        }
        let plane = h * w;
        let mut out = vec![0f32; n * 3 * plane];
        for ni in 0..n {
            let src = &nchw01[ni * 3 * plane..(ni + 1) * 3 * plane];
            let nhwc = rgb01_to_bgr_m11(src, h, w);
            let input = Array4::from_shape_vec((1, h, w, 3), nhwc)
                .context("white-box input is not contiguous NHWC")?;
            let (shape, data) = self.run_session(input.view())?;
            let (on, oh, ow, oc) = match shape.as_slice() {
                [on, oh, ow, oc] => (*on, *oh, *ow, *oc),
                other => bail!("white-box output shape {other:?}, expected NHWC"),
            };
            if on != 1 || oc != 3 || oh < h || ow < w {
                bail!("white-box output {on}x{oh}x{ow}x{oc} cannot crop to 1x{h}x{w}x3");
            }
            let rgb = bgr_m11_to_rgb01(&data, oh, ow, h, w);
            let dst = ni * 3 * plane;
            out[dst..dst + rgb.len()].copy_from_slice(&rgb);
        }
        Ok(out)
    }

    /// One `[N, H, W, 3]` call. Same per-image conversions as the loop above.
    fn run_whitebox_batch(&self, nchw01: &[f32], n: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let plane = h * w;
        let mut nhwc = Vec::with_capacity(n * 3 * plane);
        for ni in 0..n {
            nhwc.extend(rgb01_to_bgr_m11(
                &nchw01[ni * 3 * plane..(ni + 1) * 3 * plane],
                h,
                w,
            ));
        }
        let input = Array4::from_shape_vec((n, h, w, 3), nhwc)
            .context("white-box input is not contiguous NHWC")?;
        let (shape, data) = self.run_session(input.view())?;
        let (on, oh, ow, oc) = match shape.as_slice() {
            [on, oh, ow, oc] => (*on, *oh, *ow, *oc),
            other => bail!("white-box output shape {other:?}, expected NHWC"),
        };
        if on != n || oc != 3 || oh < h || ow < w {
            bail!("white-box output {on}x{oh}x{ow}x{oc} cannot crop to {n}x{h}x{w}x3");
        }
        let per = oh * ow * 3;
        let mut out = Vec::with_capacity(n * 3 * plane);
        for ni in 0..n {
            out.extend(bgr_m11_to_rgb01(
                &data[ni * per..(ni + 1) * per],
                oh,
                ow,
                h,
                w,
            ));
        }
        Ok(out)
    }

    fn run_session(&self, input: ndarray::ArrayView4<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let tensor = TensorRef::from_array_view(input).map_err(ort_anyhow)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(ort_anyhow)
            .context("onnx session run")?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(ort_anyhow)
            .context("onnx output is not f32")?;
        let shape = shape.iter().map(|d| *d as usize).collect();
        Ok((shape, data.to_vec()))
    }
}

impl DistillTeacher for OnnxTeacher {
    fn stylize(&self, content: Tensor<InferBackend, 4>) -> Result<Tensor<InferBackend, 4>> {
        let device = content.device();
        let [n, c, h, w] = content.dims();
        if c != 3 {
            bail!("onnx teacher expects 3 channels, got {c}");
        }
        let data = content
            .into_data()
            .to_vec::<f32>()
            .context("read teacher content back from the training device")?;
        let out = self.run_nchw(&data, n, h, w)?;
        Ok(Tensor::from_data(
            TensorData::new(out, [n, 3, h, w]),
            &device,
        ))
    }
}

fn ort_anyhow(err: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{err}")
}

/// Rewrite rain-princess so both nearest upsamples use scales `[1, 1, 2, 2]`.
fn rain_constant_scales(model: &Path) -> Result<PathBuf> {
    let script = r#"
import sys
import numpy as np
import onnx
from onnx import numpy_helper
src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
name = "const_scales_2x"
scales = numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), name=name)
model.graph.initializer.append(scales)
rewritten = 0
for node in model.graph.node:
    if node.op_type == "Upsample":
        node.input[1] = name
        rewritten += 1
if rewritten != 2:
    raise SystemExit(f"expected 2 Upsample nodes, rewrote {rewritten}")
onnx.save(model, dst)
"#;
    let out = rewrite_onnx(model, script, "fpga_st_train_rain_scales.onnx")
        .context("rain-princess scale rewrite")?;
    log::info!(
        "rain-princess upsamples pinned to constant 2x scales ({})",
        out.display()
    );
    Ok(out)
}

/// Rewrite white-box so the batch dimension of its input and output is `N`.
///
/// The released graph is `[1, H, W, 3]`. Every other op already reads its sizes
/// from `Shape`, so a batch of 32 gives the same images as 32 single runs.
fn whitebox_dynamic_batch(model: &Path) -> Result<PathBuf> {
    let script = r#"
import sys
import onnx
src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
for value in list(model.graph.input) + list(model.graph.output):
    dim = value.type.tensor_type.shape.dim[0]
    if dim.dim_value not in (0, 1):
        raise SystemExit(f"{value.name}: batch dim is {dim.dim_value}, expected 1")
    dim.ClearField("dim_value")
    dim.dim_param = "N"
onnx.checker.check_model(model)
onnx.save(model, dst)
"#;
    let out = rewrite_onnx(model, script, "fpga_st_train_whitebox_batch.onnx")
        .context("white-box batch rewrite")?;
    log::info!("white-box batch dimension made dynamic ({})", out.display());
    Ok(out)
}

/// Run `script src dst` with the teachers' Python and return `dst` (in the temp dir).
fn rewrite_onnx(model: &Path, script: &str, out_name: &str) -> Result<PathBuf> {
    let python = model
        .parent()
        .and_then(|dir| dir.parent())
        .map(|root| root.join(".venv/bin/python"))
        .filter(|path| path.is_file())
        .context("rewriting the ONNX graph needs the teachers .venv next to the model directory")?;
    let python = python.canonicalize().unwrap_or(python);
    let model = model.canonicalize().unwrap_or_else(|_| model.to_path_buf());
    let out = std::env::temp_dir().join(out_name);
    let status = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(model)
        .arg(&out)
        .status()
        .context("failed to run the teachers' Python")?;
    if !status.success() {
        bail!("ONNX rewrite exited with {status}");
    }
    Ok(out)
}

fn ort_env() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .with_name("fpga_st_train")
        .build()
        .map_err(ort_anyhow)
        .context("failed to initialise ONNX Runtime")?;
    let _ = ENV.set(env);
    ENV.get().context("ONNX Runtime environment was not stored")
}

/// Restrict Vulkan to RADV, then return Dawn's device index and the adapter name.
///
/// Dawn's WebGPU path asks for the high-performance adapter. With both the Vega
/// and the V100 visible that can be the NVIDIA card, which is the training GPU.
/// Setting the ICD list before the loader's first call leaves only vendor `0x1002`.
fn amd_vulkan_device() -> Result<(i32, String)> {
    static PINNED: OnceLock<(i32, String)> = OnceLock::new();
    if let Some(pinned) = PINNED.get() {
        return Ok(pinned.clone());
    }
    let icd = radv_icd()?;
    // The loader reads this on its first call. CUDA does not. Edition 2024 makes
    // set_var unsafe because it races other threads; teacher init runs before the
    // prefetch thread starts.
    unsafe {
        std::env::set_var("VK_DRIVER_FILES", &icd);
        std::env::set_var("VK_ICD_FILENAMES", &icd);
    }
    log::info!("Vulkan ICD pinned to {icd} for the ONNX teacher");

    let entry = unsafe { ash::Entry::load() }.context("failed to load libvulkan")?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&info, None) }
        .context("failed to create a Vulkan instance on the RADV ICD")?;
    let devices = unsafe { instance.enumerate_physical_devices() }
        .context("failed to enumerate Vulkan devices")?;
    let mut chosen: Option<(i32, String)> = None;
    for (index, device) in devices.iter().copied().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        log::info!(
            "vulkan device {index}: vendor {:#06x} {name}",
            props.vendor_id
        );
        let lower = name.to_ascii_lowercase();
        if lower.contains("nvidia") || lower.contains("tesla") {
            unsafe { instance.destroy_instance(None) };
            bail!(
                "Vulkan enumeration still sees {name} after pinning RADV; refusing to run the teacher on the training GPU"
            );
        }
        if props.vendor_id == AMD_VENDOR && chosen.is_none() {
            chosen = Some((index as i32, name));
        }
    }
    unsafe { instance.destroy_instance(None) };
    let chosen = chosen.context("no Vulkan device with vendor 0x1002 after pinning RADV")?;
    let lower = chosen.1.to_ascii_lowercase();
    if !lower.contains("vega") && !lower.contains("radeon") {
        bail!(
            "AMD device {:?} is not the Vega this run is supposed to use",
            chosen.1
        );
    }
    let _ = PINNED.set(chosen.clone());
    Ok(chosen)
}

fn radv_icd() -> Result<String> {
    const CANDIDATES: &[&str] = &[
        "/usr/share/vulkan/icd.d/radeon_icd.json",
        "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    ];
    CANDIDATES
        .iter()
        .find(|path| Path::new(path).is_file())
        .map(|path| (*path).to_owned())
        .with_context(|| format!("RADV Vulkan ICD not found in {CANDIDATES:?}"))
}

fn crop_nchw(data: &[f32], n: usize, oh: usize, ow: usize, h: usize, w: usize) -> Vec<f32> {
    if oh == h && ow == w {
        return data.to_vec();
    }
    let mut out = vec![0f32; n * 3 * h * w];
    for ni in 0..n {
        for c in 0..3 {
            for y in 0..h {
                let src = ((ni * 3 + c) * oh + y) * ow;
                let dst = ((ni * 3 + c) * h + y) * w;
                out[dst..dst + w].copy_from_slice(&data[src..src + w]);
            }
        }
    }
    out
}

fn rgb01_to_bgr_m11(rgb: &[f32], h: usize, w: usize) -> Vec<f32> {
    let plane = h * w;
    let mut out = vec![0f32; plane * 3];
    for i in 0..plane {
        let r = rgb[i];
        let g = rgb[plane + i];
        let b = rgb[2 * plane + i];
        let o = i * 3;
        out[o] = b * 2.0 - 1.0;
        out[o + 1] = g * 2.0 - 1.0;
        out[o + 2] = r * 2.0 - 1.0;
    }
    out
}

fn bgr_m11_to_rgb01(bgr: &[f32], _oh: usize, ow: usize, h: usize, w: usize) -> Vec<f32> {
    let plane = h * w;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            let src = (y * ow + x) * 3;
            let i = y * w + x;
            let b = (bgr[src] + 1.0) * 0.5;
            let g = (bgr[src + 1] + 1.0) * 0.5;
            let r = (bgr[src + 2] + 1.0) * 0.5;
            out[i] = r.clamp(0.0, 1.0);
            out[plane + i] = g.clamp(0.0, 1.0);
            out[2 * plane + i] = b.clamp(0.0, 1.0);
        }
    }
    out
}
