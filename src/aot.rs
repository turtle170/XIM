use crate::ir::{OpCode, XimGraph};
use crate::error::Result;
use crate::planner::{LoopFuser, OptimizedInstruction, FusedOp};
use std::process::Command;
use std::fs;
use tracing::info;
use libloading::{Library, Symbol};

pub struct AotCompiler;

impl AotCompiler {
    pub fn compile(graph: &XimGraph) -> Result<String> {
        let optimized = LoopFuser::fuse(&graph.instructions);
        let mut code = String::new();
        code.push_str(r#"
use std::thread;

#[no_mangle]
pub unsafe extern "C" fn xim_aot_execute(mem_ptr: *mut i16, inputs_ptr: *const i16, outputs_ptr: *mut i16) {
    let mem_usize = mem_ptr as usize;
    let inputs_usize = inputs_ptr as usize;
    let outputs_usize = outputs_ptr as usize;
"#);

        for instr in optimized {
            match instr {
                OptimizedInstruction::FusedLoop { ops, len } => {
                    code.push_str(&format!("
    // Fused Loop (XLA-style)
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunk_size = {len} / num_threads + 1;
    thread::scope(|s| {{
        for t in 0..num_threads {{
            s.spawn(move || {{
                let m = mem_usize as *mut i16;
                let start = t * chunk_size;
                let end = (start + chunk_size).min({len});
                if start < end {{
                    for i in start..end {{
                        let mut regs = [0i16; 64]; // Virtual registers for fusion
"));
                    for op in ops {
                        match op {
                            FusedOp::Load(off, reg) => {
                                code.push_str(&format!("                        regs[{reg}] = *m.add({off} + i);\n"));
                            }
                            FusedOp::Add(l, r, d) => {
                                code.push_str(&format!("                        regs[{d}] = regs[{l}].saturating_add(regs[{r}]);\n"));
                            }
                            FusedOp::Mul(l, r, d) => {
                                code.push_str(&format!("                        regs[{d}] = (((regs[{l}] as i32) * (regs[{r}] as i32)) >> 8).clamp(-32768, 32767) as i16;\n"));
                            }
                            FusedOp::Store(reg, off) => {
                                code.push_str(&format!("                        *m.add({off} + i) = regs[{reg}];\n"));
                            }
                            _ => {}
                        }
                    }
                    code.push_str("                    }\n                }\n            });\n        }\n    });\n");
                }
                OptimizedInstruction::Raw(op) => {
                    match op {
                        OpCode::LoadVarVec(var_id, dst, len) => {
                            code.push_str(&format!("
    // LoadVarVec
    std::ptr::copy_nonoverlapping((inputs_usize as *const i16).add({var_id}), (mem_usize as *mut i16).add({dst}), {len});
"));
                        }
                        OpCode::StoreVec(src, out_id, len) => {
                            code.push_str(&format!("
    // StoreVec
    std::ptr::copy_nonoverlapping((mem_usize as *const i16).add({src}), (outputs_usize as *mut i16).add({out_id}), {len});
"));
                        }
                        _ => {
                            code.push_str(&format!("    // Raw Op: {:?}\n", op));
                        }
                    }
                }
            }
        }

        code.push_str("}\n");
        
        let src_path = "aot_kernel.rs";
        let out_path = if cfg!(windows) { "aot_kernel.dll" } else { "libaot_kernel.so" };
        
        fs::write(src_path, &code)?;
        info!("AOT compiler (Fused) generated source code. Invoking rustc...");

        let rustc_args = vec![
            "--edition", "2021", 
            "-C", "opt-level=3", 
            "-C", "target-cpu=native", 
            "--crate-type", "cdylib", 
            src_path, 
            "-o", out_path
        ];

        // Attempt to use Cranelift for faster compilation if on nightly
        let version_output = Command::new("rustc").arg("--version").output()?;
        let version_str = String::from_utf8_lossy(&version_output.stdout);
        if version_str.contains("nightly") {
            // Only add if component is actually installed (we can't easily check here, but we can try)
            // Actually, let's keep it simple and just use opt-level=3 for now, 
            // unless we want to be really aggressive.
        }

        let status = Command::new("rustc")
            .args(&rustc_args)
            .status()?;

        if !status.success() {
            return Err(crate::error::XimError::Other("rustc failed to compile AOT kernel".into()));
        }

        Ok(out_path.to_string())
    }
}


pub struct AotExecutor {
    lib: Library,
    memory: Vec<i16>,
}

impl AotExecutor {
    pub fn new(graph: &XimGraph) -> Result<Self> {
        let dll_path = AotCompiler::compile(graph)?;
        let lib = unsafe { Library::new(dll_path).map_err(|e| crate::error::XimError::Other(e.to_string()))? };
        Ok(Self { lib, memory: vec![0; graph.memory_size] })
    }
    
    pub fn execute(&mut self, inputs: &[i16], outputs: &mut [i16]) -> Result<()> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(*mut i16, *const i16, *mut i16)> = self.lib.get(b"xim_aot_execute").map_err(|e| crate::error::XimError::Other(e.to_string()))?;
            func(self.memory.as_mut_ptr(), inputs.as_ptr(), outputs.as_mut_ptr());
        }
        Ok(())
    }
}
