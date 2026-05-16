use crate::ir::{OpCode, XimGraph};
use crate::error::{Result};
use std::simd::prelude::*;
use tracing::{debug};
use rayon::prelude::*;

const RAYON_CHUNK_SIZE: usize = 65536;

pub struct Executor {
    /// Internal memory (scratchpad) for i16 VM
    pub memory: Vec<i16>,
    /// Internal memory (scratchpad) for i8 VM
    pub memory_i8: Vec<i8>,
    /// Internal memory (scratchpad) for i32 VM (High Dynamic Range)
    pub memory_i32: Vec<i32>,
    /// Scale memory for Block Floating Point
    memory_f32: Vec<f32>,
}

impl Executor {
    pub fn new(memory_size: usize) -> Self {
        debug!("Initializing Executor with {} elements of scratchpad", memory_size);
        Self {
            memory: vec![0; memory_size],
            memory_i8: Vec::new(),
            memory_i32: Vec::new(),
            memory_f32: Vec::new(),
        }
    }

    fn ensure_buffers(&mut self, graph: &XimGraph) {
        let size = graph.memory_size;
        if self.memory.len() < size { self.memory.resize(size, 0); }
        
        let mut needs_i8 = false;
        let mut needs_i32 = false;
        let mut needs_f32 = false;
        
        for instr in &graph.instructions {
            match instr.op {
                OpCode::LoadVarVec8(_, _, _) | OpCode::StoreVec8(_, _, _) | OpCode::AddVec8(_, _, _, _) | OpCode::MulVec8(_, _, _, _) => needs_i8 = true,
                OpCode::AIOStep(_, _, _, _, _, 2) => needs_i32 = true,
                OpCode::BlockScale(_, _, _) | OpCode::AIOStepBlock(_, _, _, _, _, _) => needs_f32 = true,
                _ => {}
            }
        }
        
        if needs_i8 && self.memory_i8.len() < size { self.memory_i8.resize(size, 0); }
        if needs_i32 && self.memory_i32.len() < size { self.memory_i32.resize(size, 0); }
        if needs_f32 {
            let f32_size = size / 32 + 1;
            if self.memory_f32.len() < f32_size { self.memory_f32.resize(f32_size, 0.0); }
        }
    }

    /// Executes the graph using the provided inputs and writes to the output buffer.
    #[inline(always)]
    pub fn execute(&mut self, graph: &XimGraph, inputs: &[i16], outputs: &mut [i16]) -> Result<()> {
        self.ensure_buffers(graph);

        let mem = &mut self.memory;

        for (_idx, instr) in graph.instructions.iter().enumerate() {
            match instr.op {
                OpCode::LoadVar(var_id, dst) => {
                    mem[dst as usize] = inputs[var_id as usize];
                }
                OpCode::Store(src, out_id) => {
                    outputs[out_id as usize] = mem[src as usize];
                }
                OpCode::Add(lhs, rhs, dst) => {
                    mem[dst as usize] = mem[lhs as usize].saturating_add(mem[rhs as usize]);
                }
                OpCode::Sub(lhs, rhs, dst) => {
                    mem[dst as usize] = mem[lhs as usize].saturating_sub(mem[rhs as usize]);
                }
                OpCode::Mul(lhs, rhs, dst) => {
                    let res = (mem[lhs as usize] as i32 * mem[rhs as usize] as i32) >> 8;
                    mem[dst as usize] = res.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
                
                OpCode::LoadVarVec(var_id, dst, len) => {
                    mem[dst as usize..(dst + len) as usize].copy_from_slice(&inputs[var_id as usize..(var_id + len) as usize]);
                }
                OpCode::StoreVec(src, out_id, len) => {
                    outputs[out_id as usize..(out_id + len) as usize].copy_from_slice(&mem[src as usize..(src + len) as usize]);
                }
                OpCode::AddVec(lhs, rhs, dst, len) | OpCode::GradAdd(lhs, rhs, dst, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let r = std::slice::from_raw_parts(ptr.add(rhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);
                        
                        let work = |d: &mut [i16], l: &[i16], r: &[i16]| {
                            let mut i = 0;
                            while i + 16 <= d.len() {
                                let a = i16x16::from_slice(&l[i..]);
                                let b = i16x16::from_slice(&r[i..]);
                                a.saturating_add(b).copy_to_slice(&mut d[i..]);
                                i += 16;
                            }
                            while i < d.len() { d[i] = l[i].saturating_add(r[i]); i += 1; }
                        };

                        if len > RAYON_CHUNK_SIZE {
                            d.par_chunks_mut(RAYON_CHUNK_SIZE).zip(l.par_chunks(RAYON_CHUNK_SIZE)).zip(r.par_chunks(RAYON_CHUNK_SIZE)).for_each(|((cd, cl), cr)| work(cd, cl, cr));
                        } else {
                            work(d, l, r);
                        }
                    }
                }
                OpCode::MulVec(lhs, rhs, dst, len) | OpCode::GradMul(lhs, rhs, dst, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let r = std::slice::from_raw_parts(ptr.add(rhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);

                        let work = |d: &mut [i16], l: &[i16], r: &[i16]| {
                            let mut i = 0;
                            while i + 8 <= d.len() {
                                let a = i16x8::from_slice(&l[i..]);
                                let b = i16x8::from_slice(&r[i..]);
                                let res = (a.cast::<i32>() * b.cast::<i32>()) >> Simd::splat(8);
                                res.clamp(Simd::splat(i16::MIN as i32), Simd::splat(i16::MAX as i32)).cast::<i16>().copy_to_slice(&mut d[i..]);
                                i += 8;
                            }
                            while i < d.len() {
                                let res = (l[i] as i32 * r[i] as i32) >> 8;
                                d[i] = res.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                i += 1;
                            }
                        };

                        if len > RAYON_CHUNK_SIZE {
                            d.par_chunks_mut(RAYON_CHUNK_SIZE).zip(l.par_chunks(RAYON_CHUNK_SIZE)).zip(r.par_chunks(RAYON_CHUNK_SIZE)).for_each(|((cd, cl), cr)| work(cd, cl, cr));
                        } else {
                            work(d, l, r);
                        }
                    }
                }
                
                OpCode::AdamWStep(w, g, m, v, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let w_s = std::slice::from_raw_parts_mut(ptr.add(w as usize), len);
                        let g_s = std::slice::from_raw_parts(ptr.add(g as usize), len);
                        let m_s = std::slice::from_raw_parts_mut(ptr.add(m as usize), len);
                        let v_s = std::slice::from_raw_parts_mut(ptr.add(v as usize), len);

                        w_s.par_chunks_mut(RAYON_CHUNK_SIZE).zip(g_s.par_chunks(RAYON_CHUNK_SIZE)).zip(m_s.par_chunks_mut(RAYON_CHUNK_SIZE)).zip(v_s.par_chunks_mut(RAYON_CHUNK_SIZE))
                            .for_each(|(((w, g), m), v)| {
                                for i in 0..w.len() {
                                    let grad = g[i] as i32;
                                    let m_val = (m[i] as i32 * 230 + grad * 26) >> 8;
                                    m[i] = m_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                    let v_val = (v[i] as i32 * 255 + ((grad * grad) >> 8) * 1) >> 8;
                                    v[i] = v_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                    w[i] = w[i].saturating_sub((m_val >> 4) as i16);
                                }
                            });
                    }
                }

                OpCode::LoadVarVec8(var_id, dst, len) => {
                    let len = len as usize;
                    let mem8 = &mut self.memory_i8;
                    let input_slice = &inputs[var_id as usize..(var_id as usize + len)];
                    let target_slice = &mut mem8[dst as usize..(dst as usize + len)];
                    
                    let mut i = 0;
                    while i + 16 <= len {
                        let a = i16x16::from_slice(&input_slice[i..]);
                        let res = a >> Simd::splat(2);
                        res.clamp(Simd::splat(i8::MIN as i16), Simd::splat(i8::MAX as i16)).cast::<i8>().copy_to_slice(&mut target_slice[i..]);
                        i += 16;
                    }
                    while i < len { target_slice[i] = (input_slice[i] >> 2).clamp(i8::MIN as i16, i8::MAX as i16) as i8; i += 1; }
                }
                OpCode::StoreVec8(src, out_id, len) => {
                    let len = len as usize;
                    let mem8 = &self.memory_i8;
                    let source_slice = &mem8[src as usize..(src as usize + len)];
                    let target_slice = &mut outputs[out_id as usize..(out_id as usize + len)];
                    
                    let mut i = 0;
                    while i + 16 <= len {
                        let a = i8x16::from_slice(&source_slice[i..]).cast::<i16>();
                        let res = a << Simd::splat(2);
                        res.copy_to_slice(&mut target_slice[i..]);
                        i += 16;
                    }
                    while i < len { target_slice[i] = (source_slice[i] as i16) << 2; i += 1; }
                }
                OpCode::AddVec8(lhs, rhs, dst, len) => {
                    let len = len as usize;
                    let mem8 = &mut self.memory_i8;
                    unsafe {
                        let ptr = mem8.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let r = std::slice::from_raw_parts(ptr.add(rhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);
                        
                        let work = |d: &mut [i8], l: &[i8], r: &[i8]| {
                            let mut i = 0;
                            while i + 32 <= d.len() {
                                let a = i8x32::from_slice(&l[i..]);
                                let b = i8x32::from_slice(&r[i..]);
                                a.saturating_add(b).copy_to_slice(&mut d[i..]);
                                i += 32;
                            }
                            while i < d.len() { d[i] = l[i].saturating_add(r[i]); i += 1; }
                        };

                        if len > RAYON_CHUNK_SIZE {
                            d.par_chunks_mut(RAYON_CHUNK_SIZE).zip(l.par_chunks(RAYON_CHUNK_SIZE)).zip(r.par_chunks(RAYON_CHUNK_SIZE)).for_each(|((cd, cl), cr)| work(cd, cl, cr));
                        } else {
                            work(d, l, r);
                        }
                    }
                }
                OpCode::MulVec8(lhs, rhs, dst, len) => {
                    let len = len as usize;
                    let mem8 = &mut self.memory_i8;
                    unsafe {
                        let ptr = mem8.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let r = std::slice::from_raw_parts(ptr.add(rhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);

                        let work = |d: &mut [i8], l: &[i8], r: &[i8]| {
                            let mut i = 0;
                            while i + 16 <= d.len() {
                                let a = i8x16::from_slice(&l[i..]).cast::<i16>();
                                let b = i8x16::from_slice(&r[i..]).cast::<i16>();
                                let res = (a * b) >> Simd::splat(6);
                                res.clamp(Simd::splat(i8::MIN as i16), Simd::splat(i8::MAX as i16)).cast::<i8>().copy_to_slice(&mut d[i..]);
                                i += 16;
                            }
                            while i < d.len() {
                                let res = (l[i] as i16 * r[i] as i16) >> 6;
                                d[i] = res.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
                                i += 1;
                            }
                        };

                        if len > RAYON_CHUNK_SIZE {
                            d.par_chunks_mut(RAYON_CHUNK_SIZE).zip(l.par_chunks(RAYON_CHUNK_SIZE)).zip(r.par_chunks(RAYON_CHUNK_SIZE)).for_each(|((cd, cl), cr)| work(cd, cl, cr));
                        } else {
                            work(d, l, r);
                        }
                    }
                }

                // --- Accelerated Integer Optimizer (Phase 6) ---
                OpCode::AIOStep(w, g, m, v, len, precision) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let w_s = std::slice::from_raw_parts_mut(ptr.add(w as usize), len);
                        let g_s = std::slice::from_raw_parts(ptr.add(g as usize), len);
                        
                        match precision {
                            2 => { // i32 High Dynamic Range
                                let mem32 = &mut self.memory_i32;
                                let (m_s, v_s) = mem32[m as usize..].split_at_mut((v as usize).saturating_sub(m as usize));
                                let v_s = &mut v_s[0..len];
                                let m_s = &mut m_s[0..len];
                                
                                w_s.par_chunks_mut(RAYON_CHUNK_SIZE).zip(g_s.par_chunks(RAYON_CHUNK_SIZE)).zip(m_s.par_chunks_mut(RAYON_CHUNK_SIZE)).zip(v_s.par_chunks_mut(RAYON_CHUNK_SIZE))
                                    .for_each(|(((w, g), m), v)| {
                                        for i in 0..w.len() {
                                            let grad = g[i] as i32;
                                            m[i] = (m[i] * 900 + grad * 124) >> 10;
                                            v[i] = (v[i] * 999 + ((grad * grad) >> 8) * 1) >> 10;
                                            w[i] = w[i].saturating_sub((m[i] >> 8) as i16);
                                        }
                                    });
                            }
                            _ => { // Default to i16
                                let (m_s, v_s) = mem[m as usize..].split_at_mut((v as usize).saturating_sub(m as usize));
                                let v_s = &mut v_s[0..len];
                                let m_s = &mut m_s[0..len];

                                w_s.par_chunks_mut(RAYON_CHUNK_SIZE).zip(g_s.par_chunks(RAYON_CHUNK_SIZE)).zip(m_s.par_chunks_mut(RAYON_CHUNK_SIZE)).zip(v_s.par_chunks_mut(RAYON_CHUNK_SIZE))
                                    .for_each(|(((w, g), m), v)| {
                                        for i in 0..w.len() {
                                            let grad = g[i] as i32;
                                            m[i] = ((m[i] as i32 * 230 + grad * 26) >> 8) as i16;
                                            v[i] = ((v[i] as i32 * 255 + ((grad * grad) >> 8) * 1) >> 8) as i16;
                                            w[i] = w[i].saturating_sub(m[i] >> 4);
                                        }
                                    });
                            }
                        }
                    }
                }
                OpCode::BlockScale(src, _dst, len) => {
                    let len = len as usize;
                    for i in (0..len).step_by(32) {
                        let chunk_len = (32).min(len - i);
                        let mut max = 1;
                        for j in 0..chunk_len {
                            let val = mem[src as usize + i + j].abs();
                            if val > max { max = val; }
                        }
                        self.memory_f32[i / 32] = max as f32 / 32767.0;
                    }
                }
                OpCode::AIOStepBlock(w, g, m, v, s, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let w_s = std::slice::from_raw_parts_mut(ptr.add(w as usize), len);
                        let g_s = std::slice::from_raw_parts(ptr.add(g as usize), len);
                        let m_s = std::slice::from_raw_parts_mut(ptr.add(m as usize), len);
                        let v_s = std::slice::from_raw_parts_mut(ptr.add(v as usize), len);
                        let s_s = &mut self.memory_f32[s as usize..];

                        for b in 0..(len / 32) {
                            let block_scale = s_s[b];
                            for i in 0..32 {
                                let idx = b * 32 + i;
                                let grad = g_s[idx] as f32 * block_scale;
                                let grad_i16 = (grad * 256.0) as i32;
                                let m_val = (m_s[idx] as i32 * 230 + grad_i16 * 26) >> 8;
                                m_s[idx] = m_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                let v_val = (v_s[idx] as i32 * 255 + ((grad_i16 * grad_i16) >> 8) * 1) >> 8;
                                v_s[idx] = v_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                w_s[idx] = w_s[idx].saturating_sub((m_val >> 4) as i16);
                            }
                        }
                    }
                }
                OpCode::FillVec(val, dst, len) => {
                    let target = &mut mem[dst as usize..(dst + len) as usize];
                    target.fill(val);
                }
                OpCode::AddScalarVec(lhs, val, dst, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);
                        
                        let work = |d: &mut [i16], l: &[i16]| {
                            let mut i = 0;
                            let val_simd = i16x16::splat(val);
                            while i + 16 <= d.len() {
                                i16x16::from_slice(&l[i..]).saturating_add(val_simd).copy_to_slice(&mut d[i..]);
                                i += 16;
                            }
                            while i < d.len() { d[i] = l[i].saturating_add(val); i += 1; }
                        };

                        if len > RAYON_CHUNK_SIZE {
                            d.par_chunks_mut(RAYON_CHUNK_SIZE).zip(l.par_chunks(RAYON_CHUNK_SIZE)).for_each(|(cd, cl)| work(cd, cl));
                        } else {
                            work(d, l);
                        }
                    }
                }
                OpCode::MulScalarVec(lhs, val, dst, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);

                        let work = |d: &mut [i16], l: &[i16]| {
                            let mut i = 0;
                            let val_simd = i16x8::splat(val).cast::<i32>();
                            while i + 8 <= d.len() {
                                let a = i16x8::from_slice(&l[i..]);
                                let res = (a.cast::<i32>() * val_simd) >> Simd::splat(8);
                                res.clamp(Simd::splat(i16::MIN as i32), Simd::splat(i16::MAX as i32)).cast::<i16>().copy_to_slice(&mut d[i..]);
                                i += 8;
                            }
                            while i < d.len() {
                                let res = (l[i] as i32 * val as i32) >> 8;
                                d[i] = res.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                i += 1;
                            }
                        };

                        if len > RAYON_CHUNK_SIZE {
                            d.par_chunks_mut(RAYON_CHUNK_SIZE).zip(l.par_chunks(RAYON_CHUNK_SIZE)).for_each(|(cd, cl)| work(cd, cl));
                        } else {
                            work(d, l);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn execute_i8(&mut self, graph: &XimGraph, _inputs: &[i8], _outputs: &mut [i8]) -> Result<()> {
        let mem = &mut self.memory_i8;

        for instr in &graph.instructions {
            match instr.op {
                OpCode::AddVec8(lhs, rhs, dst, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let r = std::slice::from_raw_parts(ptr.add(rhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);
                        let mut i = 0;
                        while i + 32 <= len {
                            i8x32::from_slice(&l[i..]).saturating_add(i8x32::from_slice(&r[i..])).copy_to_slice(&mut d[i..]);
                            i += 32;
                        }
                        while i < len { d[i] = l[i].saturating_add(r[i]); i += 1; }
                    }
                }
                OpCode::MulVec8(lhs, rhs, dst, len) => {
                    let len = len as usize;
                    unsafe {
                        let ptr = mem.as_mut_ptr();
                        let l = std::slice::from_raw_parts(ptr.add(lhs as usize), len);
                        let r = std::slice::from_raw_parts(ptr.add(rhs as usize), len);
                        let d = std::slice::from_raw_parts_mut(ptr.add(dst as usize), len);
                        let mut i = 0;
                        while i + 16 <= len {
                            let a = i8x16::from_slice(&l[i..]).cast::<i16>();
                            let b = i8x16::from_slice(&r[i..]).cast::<i16>();
                            ((a * b) >> Simd::splat(6)).clamp(Simd::splat(i8::MIN as i16), Simd::splat(i8::MAX as i16)).cast::<i8>().copy_to_slice(&mut d[i..]);
                            i += 16;
                        }
                        while i < len { d[i] = ((l[i] as i16 * r[i] as i16) >> 6).clamp(i8::MIN as i16, i8::MAX as i16) as i8; i += 1; }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
