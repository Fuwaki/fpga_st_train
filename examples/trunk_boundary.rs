//! Dump the H/4 maps around a trained tiny_dw residual trunk.
//!
//! ```text
//! cargo run --release --example trunk_boundary --no-default-features --features cuda -- \
//!   experiments/tiny_dw_vangogh_plugin/trials/in_h4 \
//!   experiments/tiny_dw_vangogh_plugin/analysis/in_h4_trunk
//! ```
//!
//! Each sample writes three raw f32 NCHW files: `encoder` (trunk input), `trunk`
//! (after the residual blocks), `decoder_in` (after the H/4 plugin).

use std::fs;
use std::path::Path;

use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use fpga_st_train::InferBackend;
use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::data::{load_rgb, rgb_to_tensor};
use fpga_st_train::student::TinyDwStudent;

fn main() {
    let trial = std::env::args().nth(1).expect("trial dir");
    let out_dir = std::env::args().nth(2).expect("output dir");
    let device = select_device("d0");
    let card = ModelCard::load(&trial).expect("model.json");
    let options = &card.options;
    let model = TinyDwStudent::<InferBackend>::from_named_plugin(
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
        let boundary = model.trunk_boundary(rgb_to_tensor::<InferBackend>(&image, &device));
        write_map(&out_dir, name, "encoder", boundary.encoder);
        write_map(&out_dir, name, "trunk", boundary.trunk);
        write_map(&out_dir, name, "decoder_in", boundary.decoder_in);
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
    let bytes = f32_le_bytes(&values);
    fs::write(format!("{stem}.f32"), bytes).expect("raw");
    println!("  {stage} {:?}", dims);
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
