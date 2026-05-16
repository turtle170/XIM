#![feature(portable_simd)]

pub mod error;
pub mod executor;
pub mod ir;
pub mod quant;
pub mod planner;
pub mod ffi;
pub mod autograd;
pub mod mmap_loader;
pub mod pipeline;
pub mod aot;
pub mod jit;

pub use error::{XimError, Result};
pub use executor::Executor;
pub use ir::{Instruction, OpCode, XimGraph};
pub use quant::{Quantizer, Calibrator};
pub use planner::MemoryPlanner;
pub use autograd::{StochasticQuantizer, Xorshift32};
pub use mmap_loader::MmapLoader;
pub use pipeline::PipelineExecutor;
pub use aot::{AotCompiler, AotExecutor};


