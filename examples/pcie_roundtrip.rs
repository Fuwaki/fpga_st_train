//! Measure host↔device bounce cost on the selected adapter.
use burn::prelude::*;
use burn::tensor::Distribution;
use fpga_st_train::backend::{InferBackend, select_device};
use std::time::Instant;

fn sync(t: &Tensor<InferBackend, 4>) {
    let _ = t.clone().mean().into_scalar();
}

fn main() {
    let spec = std::env::args().nth(1).unwrap_or_else(|| "d1".into());
    let device = select_device(&spec);
    let shape = [4, 64, 128, 128]; // ~4 MiB f32
    let n = 30usize;

    let mut x = Tensor::<InferBackend, 4>::random(shape, Distribution::Default, &device);
    sync(&x);

    let t0 = Instant::now();
    for _ in 0..n {
        let data = x.clone().into_data();
        x = Tensor::from_data(data, &device);
    }
    sync(&x);
    let roundtrip_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;

    let t1 = Instant::now();
    for _ in 0..n {
        x = x.clone().add_scalar(0.0);
    }
    sync(&x);
    let ondevice_ms = t1.elapsed().as_secs_f64() * 1e3 / n as f64;

    let bytes = (shape.iter().product::<usize>() * 4) as f64;
    println!(
        "device {spec}  tensor {shape:?}  ({:.2} MiB f32)",
        bytes / (1024.0 * 1024.0)
    );
    println!(
        "into_data+from_data roundtrip: {roundtrip_ms:.3} ms/iter  (~{:.2} GB/s effective two-way)",
        (2.0 * bytes) / (roundtrip_ms * 1e-3) / 1e9
    );
    println!("on-device add_scalar chain:   {ondevice_ms:.3} ms/iter");
    println!(
        "roundtrip / on-device ratio:   {:.0}×",
        roundtrip_ms / ondevice_ms.max(1e-9)
    );
}
