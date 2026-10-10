use anyhow::Result;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};

use fpga_st_train::backend::select_device;
use fpga_st_train::InferBackend;

use crate::cli::ProbeArgs;

pub fn run(args: ProbeArgs, device_spec: &str) -> Result<()> {
    let device = select_device(device_spec);
    let mut report = String::new();
    report.push_str(&format!(
        "backend {}\n",
        fpga_st_train::backend::BACKEND_NAME
    ));

    let x = Tensor::<InferBackend, 4>::from_data(
        TensorData::new(probe_input([1, 3, 32, 32]), [1, 3, 32, 32]),
        &device,
    );
    stats("00 input", x.clone(), &mut report);

    let padded = x.clone().pad([(1, 1), (1, 1)], PadMode::Reflect);
    stats("01 reflect pad 1", padded.clone(), &mut report);

    let mk = |cin: usize, cout: usize, k: usize, s: usize, g: usize, bias: bool| {
        let mut conv = Conv2dConfig::new([cin, cout], [k, k])
            .with_stride([s, s])
            .with_groups(g)
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(bias)
            .init::<InferBackend>(&device);
        let w_shape = [cout, cin / g, k, k];
        let n: usize = w_shape.iter().product();
        let w: Vec<f32> = (0..n)
            .map(|i| ((i * 29 % 97) as f32 / 97.0 - 0.5) * 0.2)
            .collect();
        conv.weight = burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(w, w_shape),
            &device,
        ));
        if bias {
            let b: Vec<f32> = (0..cout)
                .map(|i| ((i * 13 % 41) as f32 / 41.0 - 0.5) * 0.1)
                .collect();
            conv.bias = Some(burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::new(b, [cout]),
                &device,
            )));
        }
        conv
    };

    let c1 = mk(3, 16, 9, 1, 1, true);
    let p4 = x.clone().pad([(4, 4), (4, 4)], PadMode::Reflect);
    let h1 = relu(c1.forward(p4));
    stats("02 relu(conv 3->16 k9 g1)", h1.clone(), &mut report);

    let c2 = mk(16, 32, 3, 2, 16, true);
    let h2 = relu(c2.forward(h1.pad([(1, 1), (1, 1)], PadMode::Reflect)));
    stats("03 relu(conv 16->32 k3 s2 g16)", h2.clone(), &mut report);

    let c3 = mk(32, 32, 1, 1, 1, true);
    let h3 = relu(c3.forward(h2));
    stats("04 relu(conv 32->32 k1 g1)", h3.clone(), &mut report);

    let c4 = mk(32, 64, 3, 2, 32, true);
    let h4 = relu(c4.forward(h3.pad([(1, 1), (1, 1)], PadMode::Reflect)));
    stats("05 relu(conv 32->64 k3 s2 g32)", h4.clone(), &mut report);

    let c5 = mk(64, 64, 1, 1, 1, true);
    let h5 = relu(c5.forward(h4));
    stats("06 relu(conv 64->64 k1 g1)", h5.clone(), &mut report);

    let c6 = mk(64, 64, 3, 1, 64, true);
    let h6 = c6.forward(h5.pad([(1, 1), (1, 1)], PadMode::Reflect));
    stats("07 conv 64->64 k3 g64", h6.clone(), &mut report);

    let c7 = mk(64, 64, 3, 1, 1, true);
    let r1 = relu(c6.forward(h6.pad([(1, 1), (1, 1)], PadMode::Reflect)));
    let r2 = c7.forward(r1.clone().pad([(1, 1), (1, 1)], PadMode::Reflect));
    stats("08 residual out (64ch)", r2, &mut report);

    let y = Tensor::<InferBackend, 4>::from_data(
        TensorData::new(probe_input([1, 4, 8, 8]), [1, 4, 8, 8]),
        &device,
    );
    let up = burn::tensor::module::interpolate(
        y.clone(),
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    stats("08 bilinear 2x align=false", up, &mut report);

    let pool = y.clone().mean_dim(2).mean_dim(3);
    stats("09 mean_dim(2).mean_dim(3)", pool, &mut report);

    let (m, s) = fpga_st_train::teacher::microast::calc_mean_std(y.clone());
    stats("10 adain mean", m, &mut report);
    stats("11 adain std", s, &mut report);

    match &args.out {
        Some(path) => std::fs::write(path, &report)?,
        None => print!("{report}"),
    }
    Ok(())
}

fn probe_input(shape: [usize; 4]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
        .collect()
}

fn stats<B: Backend>(label: &str, t: Tensor<B, 4>, out: &mut String) {
    let shape = t.dims();
    let sum = t.clone().sum().into_scalar().to_f64();
    let mean = t.clone().mean().into_scalar().to_f64();
    let flat = t.into_data().to_vec::<f32>().unwrap_or_default();
    let mx = flat.iter().cloned().fold(f32::MIN, f32::max);
    let mn = flat.iter().cloned().fold(f32::MAX, f32::min);
    let sample: Vec<String> = [0usize, 1, flat.len() / 2, flat.len().saturating_sub(1)]
        .iter()
        .map(|&i| format!("{:.6}", flat.get(i).copied().unwrap_or(f32::NAN)))
        .collect();
    out.push_str(&format!(
        "{label:<34} shape {shape:?} sum {sum:.6} mean {mean:.6} min {mn:.6} max {mx:.6} sample [{}]\n",
        sample.join(", ")
    ));
}
