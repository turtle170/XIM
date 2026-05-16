# XIM: Accelerated Integer Math

**XIM** (Accelerated Integer Math) is a high-performance, low-latency deep learning execution engine designed for strict 16-bit integer quantization. It bridges the gap between Python-based research (PyTorch/JAX) and native machine-code performance through an optimizing JIT compiler.

## Key Features

- **Titan VM Architecture**: A multi-precision scratchpad executor supporting `i8`, `i16`, `i32`, and `f32` (Block Floating Point) operations.
- **Hierarchical Loop Fusion**: A 3-stage optimization pipeline that collapses high-level graphs into monolithic native kernels:
  - **Stage 1 (Local)**: Merges consecutive vector operations.
  - **Stage 2 (Global)**: Merges loops across non-vector operation boundaries.
  - **Stage 3 (SuperFuser)**: Performs Memory-to-Register promotion and fuses complex optimizers (e.g., AdamW).
- **Cranelift JIT Backend**: Generates optimized machine code in milliseconds directly in memory, bypassing heavy external toolchains like LLVM/rustc for inference.
- **Pipeline Execution**: Dedicated multi-threaded executor with layer-wise streaming via `crossbeam-channel` for large Transformer models.
- **Deep Integration**:
  - **PyTorch**: Native `torch.compile` backend via TorchDynamo.
  - **JAX**: Custom XIM primitives and `jax.pure_callback` integration.

## Architecture

XIM treats deep learning graphs as a sequence of integer opcodes. The compilation flow is:
1. **FX/JAX Lowering**: Python graphs are lowered to `.xim` IR.
2. **Memory Planning**: Static liveness analysis minimizes scratchpad memory footprint.
3. **Hierarchical Fusion**: The Planner identifies fusion groups to minimize memory bandwidth.
4. **JIT Compilation**: Cranelift transforms fused groups into native machine code micro-kernels.

## Performance (500M Model Benchmark)

| Backend | Iteration Time | Speedup |
| :--- | :--- | :--- |
| PyTorch Eager | ~1.1s | 1.0x |
| **XIM SuperFused JIT** | **~0.15s** (Steady State) | **~7x** |

*Note: Initial compilation of aggressive fusion groups for large models carries a one-time latency, after which kernels run at native speeds.*

## Installation

### WinGet (Recommended for Windows)
XIM is available on WinGet. This will automatically setup the required Rust toolchain and perform a native compilation for your CPU.
```powershell
winget install turtle170.XIM
# After installation, run xim to perform native bootstrapping
xim
```

### Build from Source
```powershell
# Clone the repository
git clone https://github.com/turtle170/XIM.git
cd XIM

# Build the Python extension
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release
cp target/release/xim.dll xim.pyd
```

## Usage

### PyTorch Integration
```python
import torch
import xim_torch

model = MyModel()
# Use XIM as a torch.compile backend
optimized_model = torch.compile(model, backend="xim")

output = optimized_model(input_tensor)
```

## License
MIT
