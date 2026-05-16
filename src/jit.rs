use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use crate::ir::{XimGraph, OpCode};
use crate::planner::{LoopFuser, KernelFuser, SuperFuser, OptimizedInstruction, FusedOp};
use crate::error::Result;

pub struct CraneliftCompiler {
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    module: JITModule,
}

impl CraneliftCompiler {
    pub fn new() -> Self {
        let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
        let module = JITModule::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
        }
    }

    pub fn compile(&mut self, graph: &XimGraph) -> Result<*const u8> {
        let local_optimized = LoopFuser::fuse(&graph.instructions);
        let mut optimized = KernelFuser::fuse_kernels(local_optimized);
        SuperFuser::optimize(&mut optimized);
        
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // mem_ptr
        sig.params.push(AbiParam::new(types::I64)); // inputs_ptr
        sig.params.push(AbiParam::new(types::I64)); // outputs_ptr

        let func_id = self.module.declare_function("xim_jit_execute", Linkage::Export, &sig).unwrap();
        self.ctx.func.signature = sig;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let mem_ptr = builder.block_params(entry_block)[0];
            let inputs_ptr = builder.block_params(entry_block)[1];
            let outputs_ptr = builder.block_params(entry_block)[2];

            for instr in optimized {
                match instr {
                    OptimizedInstruction::FusedLoop { ops, len } => {
                        Self::emit_fused_loop(&mut builder, mem_ptr, &ops, len);
                    }
                    OptimizedInstruction::Raw(op) => {
                        Self::emit_raw_op(&mut builder, mem_ptr, inputs_ptr, outputs_ptr, &op);
                    }
                }
            }

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.finalize_definitions().unwrap();

        let code = self.module.get_finalized_function(func_id);
        Ok(code)
    }

    fn emit_fused_loop(builder: &mut FunctionBuilder, mem_ptr: Value, ops: &[FusedOp], len: u32) {
        let loop_block = builder.create_block();
        let exit_block = builder.create_block();
        
        let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 4, 0));
        let zero = builder.ins().iconst(types::I32, 0);
        builder.ins().stack_store(zero, slot, 0);

        builder.ins().jump(loop_block, &[]);
        builder.switch_to_block(loop_block);

        let i = builder.ins().stack_load(types::I32, slot, 0);
        let len_val = builder.ins().iconst(types::I32, len as i64);
        
        let i_64 = builder.ins().uextend(types::I64, i);
        let i_bytes = builder.ins().imul_imm(i_64, 2);

        let mut regs = std::collections::HashMap::new();

        for op in ops {
            match op {
                FusedOp::Load(off, reg) => {
                    let off_bytes = builder.ins().iadd_imm(i_bytes, (*off as i64) * 2);
                    let addr = builder.ins().iadd(mem_ptr, off_bytes);
                    let val = builder.ins().load(types::I16, MemFlags::new(), addr, 0);
                    regs.insert(*reg, val);
                }
                FusedOp::LoadConst(val, reg) => {
                    let v = builder.ins().iconst(types::I16, *val as i64);
                    regs.insert(*reg, v);
                }
                FusedOp::Add(l, r, d) => {
                    let res = builder.ins().iadd(regs[l], regs[r]);
                    regs.insert(*d, res);
                }
                FusedOp::Mul(l, r, d) => {
                    let lv = builder.ins().sextend(types::I32, regs[l]);
                    let rv = builder.ins().sextend(types::I32, regs[r]);
                    let mv = builder.ins().imul(lv, rv);
                    let shifted = builder.ins().sshr_imm(mv, 8);
                    let res = builder.ins().ireduce(types::I16, shifted);
                    regs.insert(*d, res);
                }
                FusedOp::Store(reg, off) => {
                    let off_bytes = builder.ins().iadd_imm(i_bytes, (*off as i64) * 2);
                    let addr = builder.ins().iadd(mem_ptr, off_bytes);
                    builder.ins().store(MemFlags::new(), regs[reg], addr, 0);
                }
                FusedOp::AdamWStep { w, g, m, v } => {
                    let g_val = regs[g];
                    let grad = builder.ins().sextend(types::I32, g_val);
                    
                    let m_val = builder.ins().sextend(types::I32, regs[m]);
                    let m_term1 = builder.ins().imul_imm(m_val, 230);
                    let m_term2 = builder.ins().imul_imm(grad, 26);
                    let m_sum = builder.ins().iadd(m_term1, m_term2);
                    let m_shifted = builder.ins().sshr_imm(m_sum, 8);
                    let m_new = builder.ins().ireduce(types::I16, m_shifted);
                    regs.insert(*m, m_new);

                    let v_val = builder.ins().sextend(types::I32, regs[v]);
                    let v_term1 = builder.ins().imul_imm(v_val, 255);
                    let grad_sq = builder.ins().imul(grad, grad);
                    let grad_sq_scaled = builder.ins().sshr_imm(grad_sq, 8);
                    let v_sum = builder.ins().iadd(v_term1, grad_sq_scaled);
                    let v_shifted = builder.ins().sshr_imm(v_sum, 8);
                    let v_new = builder.ins().ireduce(types::I16, v_shifted);
                    regs.insert(*v, v_new);

                    let m_new_32 = builder.ins().sextend(types::I32, m_new);
                    let update_32 = builder.ins().sshr_imm(m_new_32, 4);
                    let update = builder.ins().ireduce(types::I16, update_32);
                    let w_new = builder.ins().isub(regs[w], update);
                    regs.insert(*w, w_new);
                }
                _ => {}
            }
        }

        let next_i = builder.ins().iadd_imm(i, 1);
        builder.ins().stack_store(next_i, slot, 0);
        let cond_next = builder.ins().icmp(IntCC::UnsignedLessThan, next_i, len_val);
        builder.ins().brif(cond_next, loop_block, &[], exit_block, &[]);

        builder.switch_to_block(exit_block);
        builder.seal_block(loop_block);
        builder.seal_block(exit_block);
    }

    fn emit_raw_op(builder: &mut FunctionBuilder, mem_ptr: Value, inputs_ptr: Value, _outputs_ptr: Value, op: &OpCode) {
        match op {
            OpCode::LoadVarVec(var_id, dst, len) => {
                let loop_block = builder.create_block();
                let exit_block = builder.create_block();
                let slot = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 4, 0));
                let zero = builder.ins().iconst(types::I32, 0);
                builder.ins().stack_store(zero, slot, 0);
                
                builder.ins().jump(loop_block, &[]);
                builder.switch_to_block(loop_block);
                let i = builder.ins().stack_load(types::I32, slot, 0);
                let i_64 = builder.ins().uextend(types::I64, i);
                let i_bytes = builder.ins().imul_imm(i_64, 2);
                
                let off_src = builder.ins().iadd_imm(i_bytes, (*var_id as i64) * 2);
                let s_addr = builder.ins().iadd(inputs_ptr, off_src);
                let val = builder.ins().load(types::I16, MemFlags::new(), s_addr, 0);
                
                let off_dst = builder.ins().iadd_imm(i_bytes, (*dst as i64) * 2);
                let d_addr = builder.ins().iadd(mem_ptr, off_dst);
                builder.ins().store(MemFlags::new(), val, d_addr, 0);

                let next_i = builder.ins().iadd_imm(i, 1);
                builder.ins().stack_store(next_i, slot, 0);
                let len_val = builder.ins().iconst(types::I32, *len as i64);
                let cond = builder.ins().icmp(IntCC::UnsignedLessThan, next_i, len_val);
                builder.ins().brif(cond, loop_block, &[], exit_block, &[]);
                
                builder.switch_to_block(exit_block);
                builder.seal_block(loop_block);
                builder.seal_block(exit_block);
            }
            _ => {}
        }
    }
}
