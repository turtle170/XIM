use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OpCode {
    /// Load from variable table to internal memory: LoadVar(var_id, dst_offset)
    LoadVar(u32, u32),
    /// Store from internal memory to output: Store(src_offset, output_id)
    Store(u32, u32),
    /// Add: Add(lhs_offset, rhs_offset, dst_offset)
    Add(u32, u32, u32),
    /// Sub: Sub(lhs_offset, rhs_offset, dst_offset)
    Sub(u32, u32, u32),
    /// Mul: Mul(lhs_offset, rhs_offset, dst_offset)
    Mul(u32, u32, u32),
    
    // --- Vector Operations (Phase 2) ---
    /// AddVec(lhs_offset, rhs_offset, dst_offset, len)
    AddVec(u32, u32, u32, u32),
    /// MulVec(lhs_offset, rhs_offset, dst_offset, len)
    MulVec(u32, u32, u32, u32),
    /// LoadVarVec(var_id, dst_offset, len)
    LoadVarVec(u32, u32, u32),
    /// StoreVec(src_offset, output_id, len)
    StoreVec(u32, u32, u32),

    // --- Training Operations (Phase 4) ---
    /// GradAdd(lhs_offset, rhs_offset, dst_offset, len)
    GradAdd(u32, u32, u32, u32),
    /// GradMul(lhs_offset, rhs_offset, dst_offset, len)
    GradMul(u32, u32, u32, u32),
    /// AdamWStep(weight_offset, grad_offset, m_offset, v_offset, len)
    AdamWStep(u32, u32, u32, u32, u32),

    // --- i8 Inference Crunch (Phase 5) ---
    /// LoadVarVec8(var_id, dst_offset, len)
    LoadVarVec8(u32, u32, u32),
    /// StoreVec8(src_offset, output_id, len)
    StoreVec8(u32, u32, u32),
    /// AddVec8(lhs_offset, rhs_offset, dst_offset, len)
    AddVec8(u32, u32, u32, u32),
    /// MulVec8(lhs_offset, rhs_offset, dst_offset, len)
    MulVec8(u32, u32, u32, u32),

    // --- Accelerated Integer Optimizer (Phase 6) ---
    /// AIOStep(weight_offset, grad_offset, m_offset, v_offset, len, precision)
    /// precision: 0=i8, 1=i16, 2=i32
    AIOStep(u32, u32, u32, u32, u32, u8),
    
    /// BlockScale(src_offset, dst_offset, len)
    /// Converts a raw i16 slice into Block32 format in the scratchpad
    BlockScale(u32, u32, u32),

    /// AIOStepBlock(weight_offset, grad_offset, m_offset, v_offset, scale_offset, len)
    AIOStepBlock(u32, u32, u32, u32, u32, u32),

    /// FillVec(val_i16, dst_offset, len)
    FillVec(i16, u32, u32),

    /// AddScalarVec(lhs_offset, scalar_val, dst_offset, len)
    AddScalarVec(u32, i16, u32, u32),
    /// MulScalarVec(lhs_offset, scalar_val, dst_offset, len)
    MulScalarVec(u32, i16, u32, u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub op: OpCode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XimGraph {
    /// Mapping of variable names to their IDs
    pub variable_map: HashMap<String, u32>,
    /// Flattened instructions
    pub instructions: Vec<Instruction>,
    /// Required scratchpad memory size
    pub memory_size: usize,
}

impl XimGraph {
    pub fn new() -> Self {
        Self {
            variable_map: HashMap::new(),
            instructions: Vec::new(),
            memory_size: 0,
        }
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        debug!("Saving XIM graph to {}", path);
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        info!("Successfully saved XIM graph to {}", path);
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self> {
        debug!("Loading XIM graph from {}", path);
        let data = std::fs::read(path)?;
        let decoded = bincode::deserialize(&data)?;
        info!("Successfully loaded XIM graph from {}", path);
        Ok(decoded)
    }

    pub fn add_instruction(&mut self, op: OpCode) {
        self.instructions.push(Instruction { op });
    }
}

/// Real TorchDynamo Integration
/// This will be used by the Python FFI to build graphs on the fly.
pub struct XimBuilder {
    pub graph: XimGraph,
}

impl XimBuilder {
    pub fn new() -> Self {
        Self { graph: XimGraph::new() }
    }

    pub fn add_op(&mut self, op: OpCode) {
        self.graph.add_instruction(op);
    }

    pub fn finalize(mut self) -> XimGraph {
        crate::planner::MemoryPlanner::plan(&mut self.graph);
        self.graph
    }
}
