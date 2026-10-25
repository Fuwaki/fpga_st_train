//! Measure f32 / f16 throughput on one adapter.
//!
//! Build and run (SPIR-V, so f16 matmul can use tensor cores where the hardware has them):
//!
//! ```text
//! cargo run --release --example flops_bench --no-default-features --features cuda -- d0
//! cargo run --release --example flops_bench --no-default-features --features vulkan -- d0
//! ```
//!
//! GEMM reports dense square matmul. Conv shapes are the ones this training stack actually
//! launches (depthwise / pointwise / 64-ch 3x3) plus one large 3x3 that is big enough for
//! a tensor-core implicit GEMM. Bandwidth is a streamed add, three tensors per iteration.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::FloatDType;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use fpga_st_train::backend::{BACKEND_NAME, select_device};
use fpga_st_train::{InferBackend, InferDevice};

fn main() {
    let spec = std::env::args().nth(1).unwrap_or_else(|| "d0".into());
    let device = select_device(&spec);
    println!("backend {BACKEND_NAME}");
    println!("device_spec {spec}");
    println!("snapshot {}", snapshot());

    for (n, iters) in [(256usize, 40usize), (1024, 15), (4096, 6)] {
        for dtype in [FloatDType::F32, FloatDType::F16] {
            run_case(&format!("gemm n={n} {dtype:?}"), || {
                bench_gemm(&device, n, dtype, iters)
            });
        }
    }

    // crop 192 -> H/2 = 96, H/4 = 48. Channels match tiny_dw / MicroAST.
    let convs = [
        ("dw 12x64 k3 96", 12, 64, 64, 3, 96, 64, 40),
        ("pw 12x64 k1 96", 12, 64, 64, 1, 96, 1, 40),
        ("conv 12x64 k3 48", 12, 64, 64, 3, 48, 1, 20),
        ("conv 4x256 k3 64", 4, 256, 256, 3, 64, 1, 4),
    ];
    for (name, n, cin, cout, k, hw, groups, iters) in convs {
        for dtype in [FloatDType::F32, FloatDType::F16] {
            run_case(&format!("{name} {dtype:?}"), || {
                bench_conv(&device, n, cin, cout, k, hw, groups, dtype, iters)
            });
        }
    }

    for dtype in [FloatDType::F32, FloatDType::F16] {
        run_case(&format!("bandwidth {dtype:?}"), || {
            bench_add(&device, 64 * 1024 * 1024, dtype, 20)
        });
    }

    println!("snapshot {}", snapshot());
}

fn run_case(label: &str, f: impl FnOnce() -> String) {
    let started = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(f));
    match result {
        Ok(line) => println!(
            "{label}  {line}  wall {:.1}s",
            started.elapsed().as_secs_f64()
        ),
        Err(_) => println!("{label}  FAILED"),
    }
}

fn sync2(t: &Tensor<InferBackend, 2>) {
    let v = t.clone().mean().into_scalar().to_f64();
    if !v.is_finite() {
        panic!("non-finite");
    }
}

fn sync4(t: &Tensor<InferBackend, 4>) {
    let v = t.clone().mean().into_scalar().to_f64();
    if !v.is_finite() {
        panic!("non-finite");
    }
}

fn bench_gemm(device: &InferDevice, n: usize, dtype: FloatDType, iters: usize) -> String {
    let a = Tensor::<InferBackend, 2>::random([n, n], Distribution::Default, device).cast(dtype);
    let b = Tensor::<InferBackend, 2>::random([n, n], Distribution::Default, device).cast(dtype);
    let warm = a.clone().matmul(b.clone());
    sync2(&warm);
    let t0 = Instant::now();
    let mut c = warm;
    for _ in 0..iters {
        c = a.clone().matmul(b.clone());
    }
    sync2(&c);
    let sec = t0.elapsed().as_secs_f64();
    let flops = 2.0 * (n as f64).powi(3) * iters as f64;
    format!(
        "{:.2} TFLOP/s  {:.2} ms/iter",
        flops / sec / 1e12,
        sec * 1e3 / iters as f64
    )
}

fn bench_conv(
    device: &InferDevice,
    n: usize,
    cin: usize,
    cout: usize,
    k: usize,
    hw: usize,
    groups: usize,
    dtype: FloatDType,
    iters: usize,
) -> String {
    let x = Tensor::<InferBackend, 4>::random([n, cin, hw, hw], Distribution::Default, device)
        .cast(dtype);
    let w = Tensor::<InferBackend, 4>::random(
        [cout, cin / groups, k, k],
        Distribution::Default,
        device,
    )
    .cast(dtype);
    let opts = ConvOptions::new([1, 1], [k / 2, k / 2], [1, 1], groups);
    let warm = conv2d(x.clone(), w.clone(), None, opts.clone());
    sync4(&warm);
    let t0 = Instant::now();
    let mut y = warm;
    for _ in 0..iters {
        y = conv2d(x.clone(), w.clone(), None, opts.clone());
    }
    sync4(&y);
    let sec = t0.elapsed().as_secs_f64();
    let flops = (n * hw * hw * cout * (cin / groups) * k * k * 2 * iters) as f64;
    format!(
        "{:.2} TFLOP/s  {:.2} ms/iter",
        flops / sec / 1e12,
        sec * 1e3 / iters as f64
    )
}

fn bench_add(device: &InferDevice, elems: usize, dtype: FloatDType, iters: usize) -> String {
    let bytes = match dtype {
        FloatDType::F16 | FloatDType::BF16 => 2.0,
        FloatDType::F64 => 8.0,
        _ => 4.0,
    };
    let a =
        Tensor::<InferBackend, 2>::random([1, elems], Distribution::Default, device).cast(dtype);
    let b =
        Tensor::<InferBackend, 2>::random([1, elems], Distribution::Default, device).cast(dtype);
    let warm = a.clone() + b.clone();
    sync2(&warm);
    let t0 = Instant::now();
    let mut c = warm;
    for _ in 0..iters {
        c = a.clone() + b.clone();
    }
    sync2(&c);
    let sec = t0.elapsed().as_secs_f64();
    let traffic = 3.0 * elems as f64 * bytes * iters as f64;
    format!(
        "{:.0} GB/s  {:.2} ms/iter",
        traffic / sec / 1e9,
        sec * 1e3 / iters as f64
    )
}

fn snapshot() -> String {
    let amd = std::fs::read_to_string("/sys/class/drm/card1/device/pp_dpm_sclk")
        .unwrap_or_default()
        .lines()
        .find(|l| l.contains('*'))
        .unwrap_or("?")
        .trim()
        .to_string();
    let nv = Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm,clocks.mem,power.draw,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "nvidia-smi?".into());
    let vega_link = format!(
        "{} {}",
        std::fs::read_to_string("/sys/bus/pci/devices/0000:0a:00.0/current_link_speed")
            .unwrap_or_default()
            .trim(),
        std::fs::read_to_string("/sys/bus/pci/devices/0000:0a:00.0/current_link_width")
            .unwrap_or_default()
            .trim(),
    );
    let t10_link = format!(
        "{} {}",
        std::fs::read_to_string("/sys/bus/pci/devices/0000:04:00.0/current_link_speed")
            .unwrap_or_default()
            .trim(),
        std::fs::read_to_string("/sys/bus/pci/devices/0000:04:00.0/current_link_width")
            .unwrap_or_default()
            .trim(),
    );
    format!("vega_sclk={amd} vega_pcie={vega_link} | nvidia {nv} nvidia_pcie={t10_link}")
}
