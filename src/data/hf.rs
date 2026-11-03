//! Training images from a Hugging Face dataset repo (parquet + `hf-hub`, no Python).
//!
//! The COCO shards are one row group each (about 480 MB, 2,900 images) with no page
//! index, so reaching row `r` means decoding the image column from row 0. Reading a
//! sample that way cost about 7 s per batch of 32 on 13 cores, and was the whole
//! training step. The first time a shard is used its JPEG bytes are copied, in order,
//! into a flat pack file with an offset table; a sample is then one positioned read.
//! The packs live in `<hf cache>/../fpga_st_train/jpeg-packs/` and are reused across runs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use arrow::array::{Array, BinaryArray, StructArray};
use arrow::record_batch::RecordBatch;
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiBuilder},
};
use image::RgbImage;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::FileReader;
use rand::Rng;

use crate::config::HfDataConfig;
use crate::data::dataset::random_crop;

/// Random-access content from HF dataset parquet shards (e.g. `detection-datasets/coco`).
pub struct HfContentDataset {
    api: Api,
    repo_id: String,
    revision: String,
    shard_paths: Vec<String>,
    len: usize,
    index: OnceLock<ShardIndex>,
    /// `hf-hub` takes a non-blocking file lock, so only one thread may build the index.
    index_gate: Mutex<()>,
}

struct ShardIndex {
    shards: Vec<IndexedShard>,
}

impl ShardIndex {
    fn total_rows(&self) -> usize {
        self.shards.last().map(|s| s.start + s.rows).unwrap_or(0)
    }
}

struct IndexedShard {
    rows: usize,
    start: usize,
    pack: File,
    /// `rows + 1` byte offsets into `pack`; row `r` is `offsets[r]..offsets[r + 1]`.
    offsets: Vec<u64>,
}

impl HfContentDataset {
    pub fn open(cfg: &HfDataConfig, limit: Option<usize>) -> Result<Self> {
        // `from_env` reads `HF_ENDPOINT`, so a mirror can serve the dataset listing.
        let api = ApiBuilder::from_env()
            .build()
            .context("hf-hub ApiBuilder::from_env")?;
        let repo_id = cfg.repo.clone();
        let revision = cfg.revision.clone();
        let prefix = format!("data/{}-", cfg.split);
        let repo = Repo::with_revision(repo_id.clone(), RepoType::Dataset, revision.clone());
        // The hub listing is optional once the snapshot is on disk. A dead
        // huggingface.co between trials should not drop the rest of a matrix.
        let (mut shard_paths, shard_count_all) =
            if let Some(paths) = cached_shard_paths(&repo_id, &revision, &prefix) {
                let declared = declared_shard_total(&paths).unwrap_or(paths.len());
                let all = declared.max(paths.len());
                if paths.len() < all {
                    log::info!(
                        "hub cache has {} of {all} shards for {repo_id}; using that prefix",
                        paths.len()
                    );
                }
                (paths, all)
            } else {
                let info = api.repo(repo).info().with_context(|| {
                    format!("repo info for dataset {repo_id} (no local snapshot either)")
                })?;
                let mut paths: Vec<String> = info
                    .siblings
                    .into_iter()
                    .map(|s| s.rfilename)
                    .filter(|p| p.starts_with(&prefix) && p.ends_with(".parquet"))
                    .collect();
                paths.sort();
                let all = paths.len();
                (paths, all)
            };
        if shard_paths.is_empty() {
            bail!(
                "no parquet shards under data/{}-*.parquet in dataset {}",
                cfg.split,
                repo_id
            );
        }

        if let Some(max) = cfg.max_shards {
            shard_paths.truncate(max.min(shard_paths.len()));
        }
        let shard_count = shard_paths.len();

        let subset_rows = if let Some(rows) = cfg.num_rows {
            rows
        } else {
            match fetch_split_num_rows(&repo_id, &cfg.split) {
                Ok(full) => {
                    if shard_count < shard_count_all {
                        ((full as u128 * shard_count as u128) / shard_count_all as u128) as usize
                    } else {
                        full
                    }
                }
                Err(err) => match count_local_rows(&repo_id, &revision, &shard_paths) {
                    Ok(rows) => {
                        log::warn!(
                            "datasets-server /size failed ({err:#}); counted {rows} rows in the cached shards"
                        );
                        rows
                    }
                    Err(count_err) => {
                        log::warn!(
                            "datasets-server /size failed ({err:#}); cached row count unavailable ({count_err:#}); the index will use every downloaded row"
                        );
                        usize::MAX
                    }
                },
            }
        };
        let len = limit.map(|n| n.min(subset_rows)).unwrap_or(subset_rows);
        if subset_rows == usize::MAX {
            log::info!(
                "hf-hub dataset {repo_id} split={}: row count unknown until the first sample indexes {shard_count}/{shard_count_all} shards",
                cfg.split,
            );
        } else {
            log::info!(
                "hf-hub dataset {repo_id} split={}: {len} images (~{subset_rows} from {shard_count}/{shard_count_all} shards; index {shard_count} shards on first sample)",
                cfg.split,
            );
        }

        Ok(Self {
            api,
            repo_id,
            revision,
            shard_paths,
            len,
            index: OnceLock::new(),
            index_gate: Mutex::new(()),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn sample_cropped(&self, width: u32, height: u32, rng: &mut impl Rng) -> Result<RgbImage> {
        let index = self.shared_index()?;
        let effective = index.total_rows().min(self.len);
        if effective == 0 {
            bail!("hf dataset has no rows after indexing");
        }
        let global = rng.random_range(0..effective);
        let (shard, local_row) = locate(index, global)?;
        let begin = shard.offsets[local_row];
        let mut bytes = vec![0u8; (shard.offsets[local_row + 1] - begin) as usize];
        shard
            .pack
            .read_exact_at(&mut bytes, begin)
            .context("read jpeg from shard pack")?;
        let img = image::load_from_memory(&bytes)
            .context("decode jpeg from hf parquet")?
            .to_rgb8();
        Ok(if width == height {
            random_crop(&img, width, rng)
        } else {
            super::dataset::random_window(&img, width, height, rng)
        })
    }

    fn shared_index(&self) -> Result<&ShardIndex> {
        if let Some(index) = self.index.get() {
            return Ok(index);
        }
        let _gate = self
            .index_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(index) = self.index.get() {
            return Ok(index);
        }
        let built = self.build_index()?;
        let _ = self.index.set(built);
        self.index.get().context("hf index was just stored")
    }

    fn build_index(&self) -> Result<ShardIndex> {
        let repo = self.api.repo(Repo::with_revision(
            self.repo_id.clone(),
            RepoType::Dataset,
            self.revision.clone(),
        ));
        let mut shards = Vec::new();
        let mut start = 0usize;
        for hub_path in &self.shard_paths {
            log::info!("indexing shard {} (download if needed)", hub_path);
            let local = repo.get(hub_path)?;
            let rows = parquet_row_count(&local)?;
            let (pack, offsets) = open_pack(&local, hub_path, rows)?;
            shards.push(IndexedShard {
                rows,
                start,
                pack,
                offsets,
            });
            start += rows;
        }
        log::info!(
            "hf index ready: {} rows across {} shards",
            start,
            shards.len()
        );
        Ok(ShardIndex { shards })
    }
}

fn locate(index: &ShardIndex, global: usize) -> Result<(&IndexedShard, usize)> {
    for shard in &index.shards {
        if global < shard.start + shard.rows {
            return Ok((shard, global - shard.start));
        }
    }
    bail!("global index {global} out of range (indexed {} rows)", {
        let last = index.shards.last();
        last.map(|s| s.start + s.rows).unwrap_or(0)
    });
}

fn hub_cache_dir() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("HF_HUB_CACHE") {
        return std::path::PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(path).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".cache/huggingface/hub")
}

/// `data/train-00000-of-00040-*.parquet` in the snapshot `refs/<revision>` points at.
fn cached_shard_paths(repo_id: &str, revision: &str, prefix: &str) -> Option<Vec<String>> {
    let folder = format!("datasets--{}", repo_id.replace('/', "--"));
    let root = hub_cache_dir().join(folder);
    let rev_file = root.join("refs").join(revision);
    let rev = std::fs::read_to_string(&rev_file).ok()?;
    let data = root.join("snapshots").join(rev.trim()).join("data");
    let mut paths = Vec::new();
    let entries = std::fs::read_dir(&data).ok()?;
    for entry in entries {
        let name = entry.ok()?.file_name();
        let name = name.to_string_lossy();
        let hub_path = format!("data/{name}");
        if hub_path.starts_with(prefix) && hub_path.ends_with(".parquet") {
            paths.push(hub_path);
        }
    }
    paths.sort();
    if paths.is_empty() { None } else { Some(paths) }
}

/// The `of-00040` field is the full shard count, even when only a prefix is cached.
fn declared_shard_total(paths: &[String]) -> Option<usize> {
    let name = paths.first()?.rsplit('/').next()?;
    let rest = name.split("-of-").nth(1)?;
    let digits = rest.split('-').next()?;
    digits.parse().ok()
}

fn count_local_rows(repo_id: &str, revision: &str, hub_paths: &[String]) -> Result<usize> {
    let folder = format!("datasets--{}", repo_id.replace('/', "--"));
    let root = hub_cache_dir().join(folder);
    let rev = std::fs::read_to_string(root.join("refs").join(revision))
        .with_context(|| format!("no cached revision {revision} for {repo_id}"))?;
    let data = root.join("snapshots").join(rev.trim()).join("data");
    let mut total = 0usize;
    for hub in hub_paths {
        let name = hub
            .rsplit('/')
            .next()
            .context("shard path has no file name")?;
        total += parquet_row_count(&data.join(name))?;
    }
    Ok(total)
}

fn fetch_split_num_rows(repo: &str, split: &str) -> Result<usize> {
    let url = format!(
        "https://datasets-server.huggingface.co/size?dataset={}",
        repo.replace('/', "%2F")
    );
    let resp: serde_json::Value = ureq::get(&url)
        .call()
        .context("datasets-server /size request")?
        .into_json()
        .context("datasets-server json")?;
    let splits = resp
        .get("size")
        .and_then(|s| s.get("splits"))
        .and_then(|s| s.as_array())
        .context("missing size.splits")?;
    for entry in splits {
        if entry.get("split").and_then(|v| v.as_str()) == Some(split) {
            let rows = entry
                .get("num_rows")
                .and_then(|v| v.as_u64())
                .context("num_rows")?;
            return Ok(rows as usize);
        }
    }
    bail!("split {split} not found in datasets-server response for {repo}")
}

fn parquet_row_count(path: &Path) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let rows = parquet::file::reader::SerializedFileReader::new(file)?
        .metadata()
        .file_metadata()
        .num_rows();
    Ok(rows as usize)
}

fn pack_dir() -> PathBuf {
    let hub = hub_cache_dir();
    hub.parent()
        .unwrap_or(&hub)
        .join("fpga_st_train")
        .join("jpeg-packs")
}

/// Open (building it first if needed) the JPEG pack for one shard.
///
/// The pack is keyed by the shard's file name, which carries the hub content hash.
/// It is written to temporary names and renamed into place, the offset table last,
/// so a pack with an offset table is always complete.
fn open_pack(parquet: &Path, hub_path: &str, rows: usize) -> Result<(File, Vec<u64>)> {
    let name = hub_path.rsplit('/').next().unwrap_or(hub_path);
    let dir = pack_dir();
    let bin = dir.join(format!("{name}.jpegs"));
    let idx = dir.join(format!("{name}.offsets"));
    if let Some(found) = load_pack(&bin, &idx, rows)? {
        return Ok(found);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    log::info!(
        "packing {} images of {name} into {} (once per shard)",
        rows,
        dir.display()
    );
    let tag = std::process::id();
    let bin_tmp = dir.join(format!("{name}.jpegs.{tag}.tmp"));
    let idx_tmp = dir.join(format!("{name}.offsets.{tag}.tmp"));
    let offsets = write_pack(parquet, &bin_tmp)?;
    if offsets.len() != rows + 1 {
        bail!(
            "{name}: packed {} images, parquet metadata says {rows}",
            offsets.len() - 1
        );
    }
    let mut table = Vec::with_capacity(offsets.len() * 8);
    for offset in &offsets {
        table.extend_from_slice(&offset.to_le_bytes());
    }
    std::fs::write(&idx_tmp, table).with_context(|| format!("write {}", idx_tmp.display()))?;
    std::fs::rename(&bin_tmp, &bin).with_context(|| format!("rename {}", bin.display()))?;
    std::fs::rename(&idx_tmp, &idx).with_context(|| format!("rename {}", idx.display()))?;
    load_pack(&bin, &idx, rows)?.with_context(|| format!("{name}: pack unreadable after write"))
}

/// `None` when the pack is missing or does not match the shard.
fn load_pack(bin: &Path, idx: &Path, rows: usize) -> Result<Option<(File, Vec<u64>)>> {
    let Ok(table) = std::fs::read(idx) else {
        return Ok(None);
    };
    let Ok(pack) = File::open(bin) else {
        return Ok(None);
    };
    let offsets: Vec<u64> = table
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8-byte chunk")))
        .collect();
    let size = pack.metadata().context("stat jpeg pack")?.len();
    let valid = table.len() % 8 == 0
        && offsets.len() == rows + 1
        && offsets.first() == Some(&0)
        && offsets.last() == Some(&size)
        && offsets.windows(2).all(|w| w[0] <= w[1]);
    if !valid {
        log::warn!("{} does not match its shard; rebuilding", idx.display());
        return Ok(None);
    }
    Ok(Some((pack, offsets)))
}

/// Stream every image's bytes from the shard into `out`; returns the offset table.
fn write_pack(parquet: &Path, out: &Path) -> Result<Vec<u64>> {
    let file = File::open(parquet).with_context(|| format!("open {}", parquet.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()
        .context("parquet reader")?;
    let mut sink =
        BufWriter::new(File::create(out).with_context(|| format!("create {}", out.display()))?);
    let mut offsets = vec![0u64];
    let mut at = 0u64;
    for batch in reader {
        let batch = batch.context("read parquet batch")?;
        for row in 0..batch.num_rows() {
            let bytes = image_bytes_from_batch(&batch, row)?;
            sink.write_all(&bytes)?;
            at += bytes.len() as u64;
            offsets.push(at);
        }
    }
    sink.flush()?;
    Ok(offsets)
}

fn image_bytes_from_batch(batch: &RecordBatch, row: usize) -> Result<Vec<u8>> {
    if let Some(col) = batch.column_by_name("image") {
        if let Some(st) = col.as_any().downcast_ref::<StructArray>() {
            let bytes = st
                .column_by_name("bytes")
                .or_else(|| {
                    if st.num_columns() > 0 {
                        Some(st.column(0))
                    } else {
                        None
                    }
                })
                .context("image struct has no bytes")?;
            if let Some(bin) = bytes.as_any().downcast_ref::<BinaryArray>() {
                return Ok(bin.value(row).to_vec());
            }
        }
        if let Some(bin) = col.as_any().downcast_ref::<BinaryArray>() {
            return Ok(bin.value(row).to_vec());
        }
    }
    if let Some(col) = batch.column_by_name("image_bytes") {
        if let Some(bin) = col.as_any().downcast_ref::<BinaryArray>() {
            return Ok(bin.value(row).to_vec());
        }
    }
    bail!(
        "unsupported parquet image layout (columns: {:?})",
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    );
}
