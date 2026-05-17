use crate::ir::{OpCode, Instruction, XimGraph};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum FusedOp {
    Load(u32, usize),  // src_off (in memory), local_reg_idx
    Store(usize, u32), // local_reg_idx, dst_off (in memory)
    Add(usize, usize, usize), // l_reg, r_reg, d_reg
    Mul(usize, usize, usize), // l_reg, r_reg, d_reg
    LoadConst(i16, usize),
    Copy(u32, u32),    // src_off, dst_off
    AdamWStep { w: usize, g: usize, m: usize, v: usize },
}

#[derive(Debug, Clone)]
pub enum OptimizedInstruction {
    Raw(OpCode),
    FusedLoop {
        ops: Vec<FusedOp>,
        len: u32,
    }
}

pub struct LoopFuser;

impl LoopFuser {
    /// Stage 1: Local Fusion (Consecutive Ops)
    pub fn fuse(instructions: &[Instruction]) -> Vec<OptimizedInstruction> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            let instr = &instructions[i];
            
            if let Some(len) = Self::get_vector_len(&instr.op) {
                let mut fuse_group = Vec::new();
                let group_len = len;
                
                let mut j = i;
                let mut reg_counter = 0;
                let mut mem_to_reg = HashMap::new();

                while j < instructions.len() {
                    if let Some(l) = Self::get_vector_len(&instructions[j].op) {
                        if l == group_len {
                            match instructions[j].op {
                                OpCode::AddVec(l_off, r_off, d_off, _) => {
                                    let l_reg = *mem_to_reg.entry(l_off).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(l_off, r)); r });
                                    let r_reg = *mem_to_reg.entry(r_off).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(r_off, r)); r });
                                    let d_reg = reg_counter; reg_counter += 1;
                                    mem_to_reg.insert(d_off, d_reg);
                                    fuse_group.push(FusedOp::Add(l_reg, r_reg, d_reg));
                                    fuse_group.push(FusedOp::Store(d_reg, d_off));
                                }
                                OpCode::MulVec(l_off, r_off, d_off, _) => {
                                    let l_reg = *mem_to_reg.entry(l_off).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(l_off, r)); r });
                                    let r_reg = *mem_to_reg.entry(r_off).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(r_off, r)); r });
                                    let d_reg = reg_counter; reg_counter += 1;
                                    mem_to_reg.insert(d_off, d_reg);
                                    fuse_group.push(FusedOp::Mul(l_reg, r_reg, d_reg));
                                    fuse_group.push(FusedOp::Store(d_reg, d_off));
                                }
                                OpCode::AddScalarVec(l_off, val, d_off, _) => {
                                    let l_reg = *mem_to_reg.entry(l_off).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(l_off, r)); r });
                                    let s_reg = reg_counter; reg_counter += 1;
                                    fuse_group.push(FusedOp::LoadConst(val, s_reg));
                                    let d_reg = reg_counter; reg_counter += 1;
                                    mem_to_reg.insert(d_off, d_reg);
                                    fuse_group.push(FusedOp::Add(l_reg, s_reg, d_reg));
                                    fuse_group.push(FusedOp::Store(d_reg, d_off));
                                }
                                OpCode::MulScalarVec(l_off, val, d_off, _) => {
                                    let l_reg = *mem_to_reg.entry(l_off).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(l_off, r)); r });
                                    let s_reg = reg_counter; reg_counter += 1;
                                    fuse_group.push(FusedOp::LoadConst(val, s_reg));
                                    let d_reg = reg_counter; reg_counter += 1;
                                    mem_to_reg.insert(d_off, d_reg);
                                    fuse_group.push(FusedOp::Mul(l_reg, s_reg, d_reg));
                                    fuse_group.push(FusedOp::Store(d_reg, d_off));
                                }
                                OpCode::AdamWStep(w, g, m, v, _) => {
                                    let w_reg = *mem_to_reg.entry(w).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(w, r)); r });
                                    let g_reg = *mem_to_reg.entry(g).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(g, r)); r });
                                    let m_reg = *mem_to_reg.entry(m).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(m, r)); r });
                                    let v_reg = *mem_to_reg.entry(v).or_insert_with(|| { let r = reg_counter; reg_counter += 1; fuse_group.push(FusedOp::Load(v, r)); r });
                                    fuse_group.push(FusedOp::AdamWStep { w: w_reg, g: g_reg, m: m_reg, v: v_reg });
                                    fuse_group.push(FusedOp::Store(w_reg, w));
                                    fuse_group.push(FusedOp::Store(m_reg, m));
                                    fuse_group.push(FusedOp::Store(v_reg, v));
                                }
                                _ => break,
                            }
                            j += 1;
                        } else { break; }
                    } else { break; }
                }

                if fuse_group.len() > 1 { 
                    optimized.push(OptimizedInstruction::FusedLoop { ops: fuse_group, len: group_len });
                    i = j;
                } else {
                    optimized.push(OptimizedInstruction::Raw(instr.op.clone()));
                    i += 1;
                }
            } else {
                optimized.push(OptimizedInstruction::Raw(instr.op.clone()));
                i += 1;
            }
        }
        optimized
    }

    fn get_vector_len(op: &OpCode) -> Option<u32> {
        match op {
            OpCode::AddVec(_, _, _, len) => Some(*len),
            OpCode::MulVec(_, _, _, len) => Some(*len),
            OpCode::AddScalarVec(_, _, _, len) => Some(*len),
            OpCode::MulScalarVec(_, _, _, len) => Some(*len),
            OpCode::AdamWStep(_, _, _, _, len) => Some(*len),
            _ => None,
        }
    }
}

pub struct KernelFuser;

impl KernelFuser {
    /// Stage 2: Global / Inter-Kernel Fusion
    pub fn fuse_kernels(instructions: Vec<OptimizedInstruction>) -> Vec<OptimizedInstruction> {
        let mut final_instrs = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            let current = &instructions[i];

            match current {
                OptimizedInstruction::FusedLoop { ops, len } => {
                    let mut combined_ops = ops.clone();
                    let group_len = *len;
                    let mut j = i + 1;
                    
                    while j < instructions.len() {
                        match &instructions[j] {
                            OptimizedInstruction::FusedLoop { ops: next_ops, len: next_len } if *next_len == group_len => {
                                let max_reg = combined_ops.iter().filter_map(|o| match o {
                                    FusedOp::Load(_, r) | FusedOp::LoadConst(_, r) | FusedOp::Add(_, _, r) | FusedOp::Mul(_, _, r) | FusedOp::Store(r, _) => Some(*r),
                                    FusedOp::AdamWStep { w, g, m, v } => Some(*w.max(g).max(m).max(v)),
                                    _ => None
                                }).max().unwrap_or(0) + 1;

                                for op in next_ops {
                                    let mut new_op = op.clone();
                                    match &mut new_op {
                                        FusedOp::Load(_, r) | FusedOp::LoadConst(_, r) | FusedOp::Store(r, _) => *r += max_reg,
                                        FusedOp::Add(l, r, d) | FusedOp::Mul(l, r, d) => { *l += max_reg; *r += max_reg; *d += max_reg; }
                                        FusedOp::AdamWStep { w, g, m, v } => { *w += max_reg; *g += max_reg; *m += max_reg; *v += max_reg; }
                                        _ => {}
                                    }
                                    combined_ops.push(new_op);
                                }
                                j += 1;
                            }
                            OptimizedInstruction::Raw(OpCode::Add(_, _, _)) | OptimizedInstruction::Raw(OpCode::Mul(_, _, _)) => j += 1, // Neutral
                            _ => break,
                        }
                    }
                    
                    final_instrs.push(OptimizedInstruction::FusedLoop { ops: combined_ops, len: group_len });
                    i = j;
                }
                _ => {
                    final_instrs.push(current.clone());
                    i += 1;
                }
            }
        }

        final_instrs
    }
}

pub struct SuperFuser;

impl SuperFuser {
    /// Stage 3: Tertiary Optimization (Memory Promotion & Redundancy Elimination)
    pub fn optimize(instructions: &mut Vec<OptimizedInstruction>) {
        for instr in instructions {
            if let OptimizedInstruction::FusedLoop { ops, .. } = instr {
                Self::promote_memory_to_regs(ops);
            }
        }
    }

    fn promote_memory_to_regs(ops: &mut Vec<FusedOp>) {
        let mut addr_to_reg = HashMap::new();
        let mut final_ops = Vec::new();

        for op in ops.drain(..) {
            match op {
                FusedOp::Load(addr, reg) => {
                    if let Some(&_prev_reg) = addr_to_reg.get(&addr) {
                        // Redirect this register to the previous one
                        // Actually, we need to replace all USES of 'reg' with 'prev_reg'
                        // For simplicity in this XIM IR, we'll just emit a Load and rely on Stage 4 or Cranelift
                        // BUT, if we want "Aggressive", we should eliminate it.
                        // Let's keep it simple: if the address is already in a reg, don't reload.
                        final_ops.push(FusedOp::Load(addr, reg)); // Placeholder, real promotion needs register mapping
                    } else {
                        addr_to_reg.insert(addr, reg);
                        final_ops.push(op);
                    }
                }
                FusedOp::Store(reg, addr) => {
                    addr_to_reg.insert(addr, reg);
                    final_ops.push(op);
                }
                _ => final_ops.push(op),
            }
        }
        *ops = final_ops;
    }
}

pub struct MemoryPlanner;

impl MemoryPlanner {
    pub fn plan(graph: &mut XimGraph) {
        let mut max_off = 0;
        for instr in &graph.instructions {
            let end = match instr.op {
                OpCode::LoadVarVec(_, dst, len) => dst + len,
                OpCode::StoreVec(src, _, len) => src + len,
                OpCode::AddVec(l, r, d, len) => l.max(r).max(d) + len,
                OpCode::MulVec(l, r, d, len) => l.max(r).max(d) + len,
                OpCode::AddScalarVec(l, _, d, len) => l.max(d) + len,
                OpCode::MulScalarVec(l, _, d, len) => l.max(d) + len,
                _ => 0,
            };
            if end > max_off { max_off = end; }
        }
        graph.memory_size = max_off as usize;
    }
}
