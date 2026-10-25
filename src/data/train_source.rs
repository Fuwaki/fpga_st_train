//! Disk folder or Hugging Face — one interface for training batches.

use anyhow::{Result, bail};
use image::RgbImage;
use rand::Rng;

use crate::config::DataConfig;
use crate::data::dataset::{PairDataset, load_pair_cropped};
use crate::data::hf::HfContentDataset;
use crate::data::{image_paths, load_rgb, overfit_frame};

pub enum TrainContentSource {
    Disk(PairDataset),
    Hf(HfContentDataset),
    /// One frame, returned on every sample. The teacher target is optional.
    Overfit {
        content: RgbImage,
        target: Option<RgbImage>,
    },
}

impl TrainContentSource {
    pub fn build(data: &DataConfig) -> Result<Self> {
        if let Some(edge) = data.overfit_long_edge {
            let paths = image_paths(&data.content_dir)?;
            if paths.len() != 1 {
                bail!(
                    "overfit expects one content image, found {} in {}",
                    paths.len(),
                    data.content_dir.display()
                );
            }
            let content = overfit_frame(&load_rgb(&paths[0])?, edge as u32);
            log::info!(
                "overfit frame {}x{} (long edge {edge}) from {}",
                content.width(),
                content.height(),
                paths[0].display()
            );
            return Ok(Self::Overfit {
                content,
                target: None,
            });
        }
        if let Some(hf) = &data.hf {
            let ds = HfContentDataset::open(hf, data.limit)?;
            return Ok(Self::Hf(ds));
        }
        let disk =
            PairDataset::build(&data.content_dir, data.pseudo_gt_dir.as_deref(), data.limit)?;
        Ok(Self::Disk(disk))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Disk(d) => d.len(),
            Self::Hf(h) => h.len(),
            Self::Overfit { .. } => 1,
        }
    }

    /// Content crop and optional on-disk pseudo-GT crop (HF uses online teacher only).
    pub fn sample_crops(
        &self,
        crop: u32,
        rng: &mut impl Rng,
    ) -> Result<(RgbImage, Option<RgbImage>)> {
        match self {
            Self::Disk(d) => load_pair_cropped(d.sample(rng), crop, rng),
            Self::Hf(h) => Ok((h.sample_cropped(crop, rng)?, None)),
            Self::Overfit { content, target } => Ok((content.clone(), target.clone())),
        }
    }

    /// First content path for fallback sample image when `sample_content` unset.
    pub fn first_content_path(&self) -> Option<std::path::PathBuf> {
        match self {
            Self::Disk(d) if !d.is_empty() => Some(d.get(0).content.clone()),
            Self::Disk(_) | Self::Hf(_) | Self::Overfit { .. } => None,
        }
    }
}
