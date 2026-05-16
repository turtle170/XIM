use xim::{Executor, Instruction, OpCode, Quantizer, XimGraph, MemoryPlanner, MmapLoader, StochasticQuantizer};
use std::time::Instant;
use tracing::{info, Level};
use std::env;

fn main() -> anyhow::Result<()> {
    // 1. Initialize Logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("--- XIM Phase 6: AIO & Block Scaling ---");

    let args: Vec<String> = env::args().collect();
    let exe_name = env::current_exe()?.file_name().unwrap().to_str().unwrap().to_lowercase();
    
    if args.len() == 1 || exe_name.contains("install") {
        info!("--- XIM Native Installer ---");
        info!("This will fetch the code, setup Rust Nightly + Cranelift, and compile natively.");
        bootstrap_native()?;
        return Ok(());
    }

    if args.len() > 1 {
        match args[1].as_str() {
            "xim-train" => {
                let aio = args.iter().any(|a| a == "--aio");
                info!("Starting XIM-Train (Beast Mode - i16, AIO: {})", aio);
                run_training_demo(aio)?;
            }
            "xim-run" => {
                let mmap = args.iter().any(|a| a == "--mmap");
                let i8_mode = args.iter().any(|a| a == "--i8");
                info!("Starting XIM-Run. Mmap mode: {}, i8 mode: {}", mmap, i8_mode);
                run_inference_demo(mmap, i8_mode)?;
            }
            "bootstrap" => {
                info!("Bootstrapping XIM Native Toolchain...");
                bootstrap_native()?;
            }
            _ => {
                info!("Unknown command. Use 'xim-train', 'xim-run [--mmap] [--i8]', or 'bootstrap'.");
            }
        }
    } else {
        info!("No command provided. Running default benchmark.");
        run_inference_demo(false, false)?;
    }

    Ok(())
}

fn bootstrap_native() -> anyhow::Result<()> {
    use std::process::Command;
    
    info!("Step 1: Checking for Rustup...");
    let status = Command::new("rustup").arg("--version").status();
    if status.is_err() || !status.unwrap().success() {
        return Err(anyhow::anyhow!("Rustup not found. Please install from https://rustup.rs"));
    }

    info!("Step 2: Installing Nightly and Cranelift...");
    Command::new("rustup").args(&["toolchain", "install", "nightly"]).status()?;
    Command::new("rustup").args(&["component", "add", "rustc-codegen-cranelift-preview", "--toolchain", "nightly"]).status()?;

    info!("Step 3: Compiling XIM for Native Architecture...");
    let mut child = Command::new("cargo")
        .args(&["+nightly", "build", "--release"])
        .env("RUSTFLAGS", "-C target-cpu=native")
        .spawn()?;
    
    let status = child.wait()?;
    if !status.success() {
        return Err(anyhow::anyhow!("Cargo build failed during bootstrap."));
    }

    info!("--- XIM Native Bootstrap Complete ---");
    Ok(())
}

fn run_inference_demo(use_mmap: bool, use_i8: bool) -> anyhow::Result<()> {
    let tensor_len = 1_000_000;
    
    // 2. Create a Vectorized Graph
    let mut graph = XimGraph::new();
    if use_i8 {
        graph.instructions = vec![
            Instruction { op: OpCode::LoadVarVec8(0, 0, tensor_len) },
            Instruction { op: OpCode::LoadVarVec8(1_000_000, 1_000_000, tensor_len) },
            Instruction { op: OpCode::LoadVarVec8(2_000_000, 2_000_000, tensor_len) },
            Instruction { op: OpCode::AddVec8(0, 1_000_000, 3_000_000, tensor_len) },
            Instruction { op: OpCode::MulVec8(3_000_000, 2_000_000, 4_000_000, tensor_len) },
            Instruction { op: OpCode::StoreVec8(4_000_000, 0, tensor_len) },
        ];
    } else {
        graph.instructions = vec![
            Instruction { op: OpCode::LoadVarVec(0, 0, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(1_000_000, 1_000_000, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(2_000_000, 2_000_000, tensor_len) },
            Instruction { op: OpCode::AddVec(0, 1_000_000, 3_000_000, tensor_len) },
            Instruction { op: OpCode::MulVec(3_000_000, 2_000_000, 4_000_000, tensor_len) },
            Instruction { op: OpCode::StoreVec(4_000_000, 0, tensor_len) },
        ];
    }

    MemoryPlanner::plan(&mut graph);
    info!("Planned Memory Size: {} elements", graph.memory_size);

    // 3. Prepare Inputs
    let a_val = 1.2f32;
    let b_val = 2.3f32;
    let c_val = 0.5f32;
    
    let mut inputs = vec![0i16; 3_000_000];
    for i in 0..1_000_000 {
        inputs[i] = Quantizer::to_i16(a_val);
        inputs[i + 1_000_000] = Quantizer::to_i16(b_val);
        inputs[i + 2_000_000] = Quantizer::to_i16(c_val);
    }
    
    // Mmap simulation
    if use_mmap {
        let path = "dummy_inputs.bin";
        std::fs::write(path, unsafe {
            std::slice::from_raw_parts(inputs.as_ptr() as *const u8, inputs.len() * 2)
        })?;
        let loader = MmapLoader::new(path)?;
        let mmap_inputs = loader.as_i16_slice()?;
        
        let mut outputs = vec![0i16; 1_000_000];
        let mut executor = Executor::new(graph.memory_size);
        
        let start = Instant::now();
        executor.execute(&graph, mmap_inputs, &mut outputs)?;
        info!("Mmap Inference took {:?}", start.elapsed());
    } else {
        let mut outputs = vec![0i16; 1_000_000];
        let mut executor = Executor::new(graph.memory_size);
        
        executor.execute(&graph, &inputs, &mut outputs)?;
        
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            executor.execute(&graph, &inputs, &mut outputs)?;
        }
        let duration = start.elapsed();
        
        info!("XIM Inference ({}): Executed {} iterations in {:?}", if use_i8 { "i8" } else { "i16" }, iterations, duration);
        info!("XIM Average: {:?}", duration / iterations as u32);
    }

    Ok(())
}

fn run_training_demo(use_aio: bool) -> anyhow::Result<()> {
    let tensor_len = 1_000_000;
    
    let mut graph = XimGraph::new();
    if use_aio {
        // AIOStep(W, G, M, V, len, precision=2 (i32))
        graph.instructions = vec![
            Instruction { op: OpCode::LoadVarVec(0, 0, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(1_000_000, 1_000_000, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(2_000_000, 2_000_000, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(3_000_000, 3_000_000, tensor_len) },
            Instruction { op: OpCode::AIOStep(0, 1_000_000, 2_000_000, 3_000_000, tensor_len, 2) },
            Instruction { op: OpCode::StoreVec(0, 0, tensor_len) },
        ];
    } else {
        graph.instructions = vec![
            Instruction { op: OpCode::LoadVarVec(0, 0, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(1_000_000, 1_000_000, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(2_000_000, 2_000_000, tensor_len) },
            Instruction { op: OpCode::LoadVarVec(3_000_000, 3_000_000, tensor_len) },
            Instruction { op: OpCode::AdamWStep(0, 1_000_000, 2_000_000, 3_000_000, tensor_len) },
            Instruction { op: OpCode::StoreVec(0, 0, tensor_len) },
        ];
    }

    MemoryPlanner::plan(&mut graph);
    
    let mut q = StochasticQuantizer::new(42);
    let mut inputs = vec![0i16; 4_000_000];
    
    for i in 0..1_000_000 {
        inputs[i] = q.to_i16_stochastic(0.5, 256.0); // W
        inputs[i + 1_000_000] = q.to_i16_stochastic(0.01, 256.0); // G
    }
    
    let mut outputs = vec![0i16; 1_000_000];
    let mut executor = Executor::new(graph.memory_size);
    
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        executor.execute(&graph, &inputs, &mut outputs)?;
        inputs[0..1_000_000].copy_from_slice(&outputs);
    }
    
    info!("XIM-Train (AIO: {}): {} steps took {:?}", use_aio, iterations, start.elapsed());
    
    Ok(())
}
