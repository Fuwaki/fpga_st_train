use anyhow::Result;
use image::imageops::FilterType;

use fpga_st_train::data;
use fpga_st_train::eval;

use crate::cli::MetricsArgs;

pub fn run(args: MetricsArgs) -> Result<()> {
    let paths = data::image_paths(&args.images)?;
    let reference = match &args.content {
        Some(p) => Some(data::load_rgb(p)?),
        None => None,
    };
    for path in &paths {
        let img = data::load_rgb(path)?;
        let mut line = format!("{:<44} {}", data::path_stem(path), eval::describe(&img));
        if let Some(reference) = &reference {
            let resized =
                data::resize_exact(reference, img.width(), img.height(), FilterType::Triangle);
            line.push_str(&format!(
                "  psnr_vs_content {:.2} dB",
                eval::psnr(&img, &resized)?
            ));
        }
        println!("{line}");
    }

    if let Some(sheet_path) = &args.sheet {
        let mut panes: Vec<(String, image::RgbImage)> = Vec::new();
        if let Some(reference) = &reference {
            let first = &paths[0];
            let target = data::load_rgb(first)?.dimensions();
            panes.push((
                "content".into(),
                data::resize_exact(reference, target.0, target.1, FilterType::Triangle),
            ));
        }
        for path in &paths {
            panes.push((data::path_stem(path), data::load_rgb(path)?));
        }
        let out = eval::sheet(&panes, 6);
        data::save_rgb(&out, sheet_path)?;
        println!("sheet -> {}", sheet_path.display());
    }
    Ok(())
}
