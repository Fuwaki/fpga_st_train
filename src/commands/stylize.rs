use std::time::Instant;

use anyhow::{Context, Result};
use image::imageops::FilterType;

use fpga_st_train::InferBackend;
use fpga_st_train::backend::select_device;
use fpga_st_train::data;
use fpga_st_train::eval;
use fpga_st_train::teacher::Teacher;
use fpga_st_train::teacher::microast::MicroAst;
use fpga_st_train::teacher::weights;

use crate::cli::StylizeArgs;

pub fn run(args: StylizeArgs, device_spec: &str) -> Result<()> {
    let device = select_device(device_spec);

    let started = Instant::now();
    let model: MicroAst<InferBackend> = weights::load(&args.teacher, &device)?;
    log::info!("teacher loaded in {:.2} s", started.elapsed().as_secs_f32());

    let contents = data::image_paths(&args.content)?;
    let styles = data::image_paths(&args.style)?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("cannot create {}", args.out.display()))?;
    log::info!(
        "{} content image(s) x {} style image(s)",
        contents.len(),
        styles.len()
    );

    let mut total = 0.0f32;
    for style_path in &styles {
        let style_img = data::resize_short_edge(
            &data::load_rgb(style_path)?,
            args.style_size,
            FilterType::Triangle,
        );
        let style = data::rgb_to_tensor::<InferBackend>(&style_img, &device);

        for content_path in &contents {
            let mut img = data::load_rgb(content_path)?;
            if args.content_size > 0 {
                img = data::resize_long_edge(&img, args.content_size, FilterType::Lanczos3);
            }
            if args.crop > 0 {
                img = data::center_crop(&img, args.crop, args.crop);
            }
            let content = data::rgb_to_tensor::<InferBackend>(&img, &device);

            let step = Instant::now();
            let output = model.stylize(content, style.clone(), args.alpha);
            let output = data::tensor_to_rgb(output)?;
            let millis = step.elapsed().as_secs_f32() * 1e3;
            total += millis;
            let name = format!(
                "{}__{}.{}",
                data::path_stem(content_path),
                data::path_stem(style_path),
                args.ext
            );
            data::save_rgb(&output, args.out.join(&name))?;
            println!(
                "{name}  {}x{}  {millis:.0} ms  {}",
                output.width(),
                output.height(),
                eval::describe(&output)
            );
        }
    }
    println!("total {total:.0} ms");
    Ok(())
}
