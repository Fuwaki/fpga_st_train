//! Dump decoder activations for a trained tiny_dw checkpoint.
//!
//! CPU only, so it can run beside a CUDA training job:
//!
//! ```text
//! cargo run --release --example decoder_trace --no-default-features --features ndarray \
//!   --target-dir target/ndarray -- \
//!   experiments/tiny_dw_vangogh_plugin/trials/in_h4 \
//!   experiments/tiny_dw_vangogh_plugin/analysis/in_h4_decoder
//! ```

use std::fs;
use std::path::Path;

use burn::backend::NdArray;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use fpga_st_train::config::ModelCard;
use fpga_st_train::data::{load_rgb, rgb_to_tensor};
use fpga_st_train::student::TinyDwStudent;

fn main() {
    let trial = std::env::args().nth(1).expect("trial dir");
    let out_dir = std::env::args().nth(2).expect("output dir");
    let device = Default::default();
    let card = ModelCard::load(&trial).expect("model.json");
    let options = &card.options;
    let model = TinyDwStudent::<NdArray>::from_named_plugin(
        json_usize(options, "base"),
        json_usize(options, "res_blocks"),
        json_usize(options, "mid"),
        json_usize(options, "fine"),
        options["plugin"].as_str().unwrap_or("none"),
        options["plugin_site"].as_str().unwrap_or("h4"),
        &device,
    )
    .expect("plugin")
    .load_file(
        Path::new(&trial).join("model.mpk"),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        &device,
    )
    .expect("model.mpk");

    fs::create_dir_all(&out_dir).expect("output dir");
    for name in ["animal", "cat", "city"] {
        let image = load_rgb(Path::new(&trial).join(format!("content_{name}.png"))).expect(name);
        let trace = model.decoder_trace(rgb_to_tensor::<NdArray>(&image, &device));
        write_map(&out_dir, name, "up_trunk", trace.up_trunk);
        write_map(&out_dir, name, "skip", trace.skip);
        write_map(&out_dir, name, "fused", trace.fused);
        write_map(&out_dir, name, "half_dw", trace.half_dw);
        write_map(&out_dir, name, "half_sep", trace.half_sep);
        write_map(&out_dir, name, "half_pw", trace.half_pw);
        write_map(&out_dir, name, "full_up", trace.full_up);
        write_map(&out_dir, name, "full_dw", trace.full_dw);
        write_map(&out_dir, name, "full_sep", trace.full_sep);
        write_map(&out_dir, name, "full_pw", trace.full_pw);
        write_map(&out_dir, name, "rgb", trace.rgb);
        println!("dumped {name}");
    }
}

fn json_usize(options: &serde_json::Value, key: &str) -> usize {
    options[key]
        .as_u64()
        .unwrap_or_else(|| panic!("missing {key}")) as usize
}

fn write_map<B: Backend>(dir: &str, sample: &str, stage: &str, tensor: Tensor<B, 4>) {
    let dims = tensor.dims();
    let values = tensor.into_data().to_vec::<f32>().expect("tensor to host");
    let stem = format!("{dir}/{sample}_{stage}");
    fs::write(
        format!("{stem}.json"),
        format!(
            "{{\"n\":{},\"c\":{},\"h\":{},\"w\":{}}}",
            dims[0], dims[1], dims[2], dims[3]
        ),
    )
    .expect("shape");
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(format!("{stem}.f32"), bytes).expect("raw");
    println!("  {stage} {:?}", dims);
}
