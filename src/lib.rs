//! FPGA-friendly style transfer: MicroAST teacher inference + distillation training framework.
//!
//! # Layout
//!
//! - [backend]  -- backend and device selection (default: Vulkan on the RX 580)
//! - [teacher]  -- MicroAST teacher network, ported 1:1 from the official PyTorch release
//! - [student]  -- the student model contract and the FPGA-oriented tiny student
//! - [data]     -- image IO, pairing datasets, sampling
//! - [losses]   -- pixel and total-variation terms
//! - [train]    -- the training loop, optimiser, scheduler and checkpointing
//! - [eval]     -- image-space quality metrics and contact sheets
//! - [config]   -- JSON run configuration
//! - [quant]        -- [Quantizable] trait, calibration, fake-quant ops, manifest types
//! - [export]       -- [Exportable] trait and [export::Int8Exporter]
//! - [registry]     -- pluggable `student.arch` table (train / calibrate / export / load)
//! - [infer_student] -- type-erased fp32 student for deploy
//! - [supervision]  -- teacher vs pseudo-GT target resolution
//! - [deploy]       -- delivered-model inference (fp32 / sim / int)
//! - [run_log]  -- append-only `metrics.jsonl` per training run
//! - [intref]   -- bit-exact integer interpreter for an exported model (the FPGA reference)

pub mod backend;
pub mod config;
pub mod data;
pub mod deploy;
pub mod eval;
pub mod export;
pub mod infer_student;
pub mod intref;
pub mod losses;
pub mod quant;
pub mod registry;
pub mod run_log;
pub mod student;
pub mod supervision;
pub mod teacher;
pub mod train;

pub use backend::{InferBackend, InferDevice, TrainBackend, TrainDevice};
pub use config::Config;
pub use export::Exportable;
pub use quant::Quantizable;
pub use student::{StudentModel, TinyDwStudent, TinyStudent};
