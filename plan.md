# Role: Principal Compiler & Systems Engineer
# Task: Scaffold the XIM (Accelerated Integer Math) Backend
# Language: Rust
## Objective
Design and implement the core scaffolding for **XIM**, a custom Just-In-Time (JIT) compiler backend designed to execute mathematical computation graphs using strictly `i16` quantized integer math. The system must "crush" floating-point representations into a custom fixed-point Bytecode Intermediate Representation (IR), save/load it as a `.xim` file, and execute it with extreme efficiency.

## Constraints & Execution Environment
1. **Language:** Rust. Use idiomatic, zero-cost abstractions.
2. **Hardware Target:** strictly CPU-optimized inference. Do NOT include any GPU, CUDA, or external hardware accelerator dependencies. Rely on CPU-level optimizations (e.g., SIMD, cache-locality, tight loop unrolling).
3. **Performance Target:** "Beast Mode" execution. The executor loop must be allocation-free on the hot path. Use cache-line aligned memory where appropriate.
4. **Safety:** Implement saturating arithmetic to handle quantization overflows natively without panicking.

## Architecture & Implementation Steps

### 1. The Quantizer (Frontend)
- Implement a `Quantizer` struct that converts `f32` inputs to `i16` using a defined fixed-point scaling factor (e.g., Q8.8 or Q10.6).
- Include utility functions for bidirectional conversion (`f32` <-> `i16`).

### 2. The XIM IR & Bytecode (The `.xim` Format)
- Define a compact `OpCode` enum (`Add`, `Sub`, `Mul`, `LoadVar`, `Store`, etc).
- Define the `XimGraph` struct containing:
  - A variable table (mapping input tensors to memory offsets).
  - A flat `Vec<Instruction>` representing the fused computation.
- Implement serialization/deserialization for the `XimGraph` to read/write custom `.xim` binary files.

### 3. The Executor (The JIT Runtime)
- Build a lightweight Virtual Machine / Executor that takes a loaded `XimGraph` and a pre-allocated input buffer (the variables).
- Implement an ultra-fast instruction dispatch loop (`match` statement over the bytecodes).
- The hot loop must read from the input buffer, apply the `i16` math using saturating operations (e.g., `saturating_add`, `saturating_mul`), and write to an output buffer.

### 4. ML Framework Integration Stubs
- Create placeholder traits/interfaces demonstrating how a PyTorch FX Graph or a JAX/MLIR `Quant` dialect output would lower into the `XimGraph` structure.

## Deliverables
Generate the complete Rust scaffolding, organized into a logical module structure (`quant.rs`, `ir.rs`, `executor.rs`). Prioritize code that compiles, includes necessary `#![feature(...)]` flags if SIMD is used, and features a `main.rs` that executes a simple $(A + B) \times C$ test case end-to-end.

# Role: Principal Compiler & Systems Engineer
# Task: Phase 2 XIM Evolution - Tensor SIMD, Memory Planning & FFI

## Objective
Elevate the XIM (Accelerated Integer Math) backend from scalar operations to highly optimized tensor computations. Strictly targeting CPU execution, this phase introduces SIMD vectorization, advanced memory liveness planning, and a Python FFI bridge to prepare for real-world PyTorch (TorchDynamo) integration.

## Constraints & Environment
1. **Language:** Rust (Nightly allowed if utilizing `std::simd` / `portable_simd`).
2. **Hardware Target:** CPU only. Maximize cache locality and strictly utilize CPU-level vectorization. 
3. **Performance:** Absolute zero memory allocation during the `execute` hot loop. All memory must be statically analyzed and pre-allocated.

## Architecture & Implementation Steps

### 1. Vectorized Instruction Set & SIMD Executor
- Extend the `OpCode` enum to support batched/tensor operations (e.g., `AddVec`, `MulVec`).
- Update the `XimGraph` IR to store tensor lengths/strides alongside memory offsets.
- Modify the `Executor`. For vectorized opcodes, process chunks of `i16` data using SIMD lanes (e.g., `std::simd::i16x16` or `i16x32` depending on architecture), falling back to scalar saturating math for the tail elements.

### 2. Static Memory Planner (Register Allocation / Arena)
- Implement a `MemoryPlanner` that performs a liveness analysis on the `XimGraph` before execution.
- Track when intermediate variables are created and when they are last read (consumed).
- Generate a compacted memory layout that aggressively reuses scratchpad byte-offsets for dead variables, minimizing the total required `L1/L2` cache footprint.

### 3. PyO3 Python Bridge (The PyTorch Gateway)
- Scaffold a Python binding module using the `pyo3` crate.
- Create a Python class `XimEngine` that wraps the loaded `XimGraph` and the pre-allocated scratchpad.
- Implement an `execute_graph` method that accepts Python memory views or Numpy arrays.
- **Zero-Copy Goal:** The bridge must take the continuous float array pointers, run the parallel/vectorized `Quantizer` to load them into the `i16` scratchpad, run the `execute` hot loop, and write back the dequantized results to the output pointer.

## Deliverables
Generate the updated module structures (update `executor.rs` with SIMD, create `planner.rs` for memory, create `ffi.rs` for PyO3). Update `main.rs` to include a benchmark of a vectorized operation (e.g., an element-wise operation on an array of 10,000 floats) to prove the memory planner and SIMD execution work seamlessly together.


PHASE 3:
# Role: Principal Compiler & DevOps Engineer
# Task: Phase 3 XIM Productionization - Real Backends, GitHub CI, and WinGet Distribution

## Objective
Transition XIM from a prototype to a production-ready compiler backend. This phase eliminates all mock implementations, establishes a real-world PyTorch (TorchDynamo) integration, automates the release pipeline to GitHub, and prepares the WinGet manifest for Windows distribution.

## Constraints & Environment
1. **Repository:** https://github.com/turtle170/XIM.git
2. **System Target:** Windows 11 (24H2), Intel i5-12500T (OptiPlex 5000 Micro).
3. **No Mocks Policy:** Every "placeholder" or "stub" in the codebase must be replaced with functional logic. If an operation isn't supported, it must throw a clear, catchable `XimError`.
4. **Hardware focus:** Optimized for AVX-2/SIMD on the i5-12500T.

## Architecture & Implementation Steps

### 1. Real TorchDynamo Integration (The "No Mocks" Backend)
- Replace the `TorchFxImporter` stub with a functional `torch.library` registration.
- Implement a real `Backend` class for `torch.compile` that:
    - Traverses the FX Graph.
    - Performs **Auto-Calibration**: Instead of fixed Q8.8, implement a calibration pass that observes tensor ranges and calculates the optimal `scale` and `zero_point` for each layer.
    - Generates the `.xim` bytecode on the fly.
- Implement the "Slow Fill" mechanism: The `.xim` template stays in memory while the executor updates only the dynamic input buffers (pointers) for each inference pass.

### 2. Hardening & Error Handling
- Replace all `println!` with a structured logging crate (e.g., `tracing`).
- Implement a comprehensive `XimError` enum using `thiserror`.
- Ensure the `Executor` handles tensor shape mismatches and quantization overflows with specific error codes rather than generic panics.

### 3. GitHub Integration & CI/CD
- Initialize the Git repository and push the current codebase to `https://github.com/turtle170/XIM.git`.
- Create a `.github/workflows/build.yml` file that:
    - Runs `cargo test` and `cargo fmt`.
    - Builds a release-optimized binary for `x86_64-pc-windows-msvc`.
    - Automatically creates a GitHub Release when a tag is pushed.

### 4. WinGet Distribution
- Create a `manifests/` directory containing the YAML files required for **WinGet**.
- The manifest must target the `xim.exe` CLI tool, allowing users to install XIM via `winget install turtle170.XIM`.
- Include a `winget-releaser` Action to automate the submission to the Windows Package Manager repository.

## Deliverables
1. **Code:** A fully functional `torch.compile` backend that runs a real (albeit small) neural network entirely in `i16` math.
2. **Scripts:** A `deploy.sh` (or `.ps1`) that pushes to GitHub and triggers the CI/CD pipeline.
3. **Verification:** A benchmark comparison showing the speedup of the "No Mock" XIM executor vs. standard PyTorch CPU execution on your i5-12500T.\





# Role: Lead Systems Architect & High-Performance Computing (HPC) Engineer
# Task: Phase 4 "Titan" Expansion - Massive Model Streaming & Integer Training

## Objective
Transform XIM into a world-class HPC backend capable of:
1. Executing 70B+ parameter models using out-of-core memory mapping and multi-threaded tiling.
2. Supporting "Beast Mode" integer training via Stochastic Rounding and i16 Autograd.
3. Full production deployment to GitHub and Windows Package Manager (WinGet).

## Constraints & Environment
- **Target:** High-core count CPUs (AVX-512/AVX-2).
- **Memory Strategy:** Zero-copy `mmap` for models exceeding physical RAM; NUMA-aware allocation for high-RAM systems.
- **Repository:** https://github.com/turtle170/XIM.git
- **Distribution:** WinGet (turtle170.XIM) via GitHub Actions.

## Architecture & Implementation Steps

### 1. The "Titan" Executor (Massive Model Support)
- **NUMA-Aware Rayon Integration:** Implement parallel work-stealing using the `rayon` crate. Ensure threads are pinned to physical cores to avoid cache-thrashing on large models.
- **Memory Mapping (`memmap2`):** Replace standard file loading with `mmap`. This allows XIM to handle models larger than RAM (70B+) by letting the OS manage page-swapping at the hardware level.
- **Cache Tiling (L3 Optimization):** Implement a blocked GEMM (General Matrix Multiplication) kernel. Break massive `i16` matrices into tiles that fit perfectly within the CPU’s L3 cache to maintain the <5µs latency baseline.

### 2. XIM-Train (Integer Autograd & 8-bit AdamW)
- **Stochastic Rounding:** Implement a high-speed Xorshift RNG in the quantizer. When down-casting grads to `i16`, use stochastic rounding to preserve gradient flow.
- **Backward Pass IR:** Expand `OpCode` to include gradient operations (`GradAdd`, `GradMul`).
- **Fused Optimizer:** Integrate an 8-bit AdamW optimizer kernel directly into the `.xim` execution loop to update weights in-place, eliminating extra memory passes.

### 3. Production Hardening (No Mocks)
- **Full Removal of Stubs:** All remaining mocks in `importer.rs` and `executor.rs` must be replaced with hardened, vectorized Rust logic.
- **Validation Suite:** Create a `stress_test.rs` that verifies i16 convergence on a small MNIST-style training loop to prove the Autograd logic works.

### 4. Deployment Pipeline
- **GitHub Actions:** 
    - Auto-generate `SHA256` checksums for every release.
    - Automate `cargo-dist` for multi-platform binary builds.
- **WinGet Manifest:** Generate the `Microsoft.Winget.Create` manifest. Ensure the installer correctly sets up `XIM_PATH` in the Windows environment variables.

## Deliverables
1. **Source:** Updated `executor.rs` (Rayon/SIMD), new `autograd.rs` (Stochastic Rounding), and `mmap_loader.rs`.
2. **CI/CD:** `.github/workflows/winget-release.yml`.
3. **CLI:** A functional `xim-train` command and `xim-run --mmap` for large-scale inference.