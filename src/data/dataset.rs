//! Datasets pairing content images with teacher-generated targets.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::RgbImage;
use image::imageops::FilterType;
use rand::Rng;

use super::{image_paths, load_rgb, resize_short_edge};

/// One training pair.
#[derive(Debug, Clone)]
pub struct Pair {
    /// Content image.
    pub content: PathBuf,
    /// Teacher output. None means the teacher has to be run online for this sample.
    pub target: Option<PathBuf>,
}

/// Content images paired with pseudo ground truth by matching file stems.
pub struct PairDataset {
    pairs: Vec<Pair>,
}

impl PairDataset {
    /// Build the dataset.
    ///
    /// target_dir may be None, in which case every pair has target = None and the caller
    /// is expected to run the teacher online.
    pub fn build(
        content_dir: impl AsRef<Path>,
        target_dir: Option<&Path>,
        limit: Option<usize>,
    ) -> Result<Self> {
        let content_files = image_paths(content_dir.as_ref())?;
        let targets = match target_dir {
            Some(dir) => {
                let mut map = HashMap::new();
                for path in image_paths(dir)? {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        map.insert(stem.to_owned(), path);
                    }
                }
                Some(map)
            }
            None => None,
        };

        let mut pairs = Vec::with_capacity(content_files.len());
        let mut unmatched = 0usize;
        for content in content_files {
            let stem = content
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let target = match &targets {
                Some(map) => {
                    // Teacher output is named "<content>__<style>.png" by the stylize
                    // command, so fall back to a prefix match when the exact stem misses.
                    if let Some(p) = map.get(stem) {
                        Some(p.clone())
                    } else {
                        let mut hit = None;
                        for (k, v) in map.iter() {
                            if k.starts_with(stem) {
                                hit = Some(v.clone());
                                break;
                            }
                        }
                        if hit.is_none() {
                            unmatched += 1;
                        }
                        hit
                    }
                }
                None => None,
            };
            pairs.push(Pair { content, target });
        }

        if let Some(n) = limit {
            pairs.truncate(n);
        }
        if pairs.is_empty() {
            bail!(
                "no training pairs found in {}",
                content_dir.as_ref().display()
            );
        }
        if unmatched > 0 {
            log::warn!("{unmatched} content image(s) had no matching pseudo ground truth");
        }
        log::info!("dataset: {} pairs", pairs.len());
        Ok(Self { pairs })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn get(&self, index: usize) -> &Pair {
        &self.pairs[index]
    }

    /// Pick a random pair.
    pub fn sample(&self, rng: &mut impl Rng) -> &Pair {
        &self.pairs[rng.random_range(0..self.pairs.len())]
    }
}

/// Resize if needed so both edges are at least size, then take a random crop.
pub fn random_crop(img: &RgbImage, size: u32, rng: &mut impl Rng) -> RgbImage {
    let img = if img.width() < size || img.height() < size {
        let short = img.width().min(img.height());
        resize_short_edge(img, size.max(short), FilterType::Triangle)
    } else {
        img.clone()
    };
    let max_x = img.width().saturating_sub(size);
    let max_y = img.height().saturating_sub(size);
    let x = if max_x > 0 {
        rng.random_range(0..=max_x)
    } else {
        0
    };
    let y = if max_y > 0 {
        rng.random_range(0..=max_y)
    } else {
        0
    };
    image::imageops::crop_imm(&img, x, y, size, size).to_image()
}

/// Load a pair and crop both members with the *same* window.
pub fn load_pair_cropped(
    pair: &Pair,
    size: u32,
    rng: &mut impl Rng,
) -> Result<(RgbImage, Option<RgbImage>)> {
    let content = load_rgb(&pair.content)
        .with_context(|| format!("cannot load content {}", pair.content.display()))?;
    let target = match &pair.target {
        Some(path) => {
            Some(load_rgb(path).with_context(|| format!("cannot load target {}", path.display()))?)
        }
        None => None,
    };

    // One crop window for both, so the target stays pixel-aligned with the content.
    let base = match &target {
        Some(t) => resize_pair_to_common(&content, t, size),
        None => (content, None),
    };
    let (content, target) = base;

    let (w, h) = (content.width(), content.height());
    if w < size || h < size {
        let content = resize_short_edge(&content, size, FilterType::Triangle);
        let target = target
            .as_ref()
            .map(|t| resize_short_edge(t, size, FilterType::Triangle));
        return Ok(crop_same(content, target, size, rng));
    }
    Ok(crop_same(content, target, size, rng))
}

/// Make content and target share a size (the target may differ by a pixel).
fn resize_pair_to_common(
    content: &RgbImage,
    target: &RgbImage,
    min_size: u32,
) -> (RgbImage, Option<RgbImage>) {
    let _ = min_size;
    if content.dimensions() == target.dimensions() {
        return (content.clone(), Some(target.clone()));
    }
    let resized = image::imageops::resize(
        target,
        content.width(),
        content.height(),
        FilterType::Triangle,
    );
    (content.clone(), Some(resized))
}

fn crop_same(
    content: RgbImage,
    target: Option<RgbImage>,
    size: u32,
    rng: &mut impl Rng,
) -> (RgbImage, Option<RgbImage>) {
    let max_x = content.width().saturating_sub(size);
    let max_y = content.height().saturating_sub(size);
    let x = if max_x > 0 {
        rng.random_range(0..=max_x)
    } else {
        0
    };
    let y = if max_y > 0 {
        rng.random_range(0..=max_y)
    } else {
        0
    };
    let c = image::imageops::crop_imm(&content, x, y, size, size).to_image();
    let t = target.map(|t| {
        image::imageops::crop_imm(
            &t,
            x.min(t.width().saturating_sub(1)),
            y.min(t.height().saturating_sub(1)),
            size.min(t.width()),
            size.min(t.height()),
        )
        .to_image()
    });
    (c, t)
}
