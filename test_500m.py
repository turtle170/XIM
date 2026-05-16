import torch
import time
import xim_torch

# Define a 500M parameter model (representative scale)
class Model500M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 500 million elements total across multiple large operations
        # We'll use 5 iterations of 100M operations
        self.num_elements = 100_000_000 
        self.param1 = torch.nn.Parameter(torch.randn(self.num_elements))
        self.param2 = torch.nn.Parameter(torch.randn(self.num_elements))
        self.param3 = torch.nn.Parameter(torch.randn(self.num_elements))
        self.param4 = torch.nn.Parameter(torch.randn(self.num_elements))
        self.param5 = torch.nn.Parameter(torch.randn(self.num_elements))

    def forward(self, x):
        # Sequence of operations to total 500M params/ops
        x = (x + self.param1) * 0.5
        x = (x + self.param2) * 0.5
        x = (x + self.param3) * 0.5
        x = (x + self.param4) * 0.5
        x = (x + self.param5) * 0.5
        return x

def benchmark():
    print("--- XIM 500M Model Benchmark ---")
    model = Model500M().cpu()
    x = torch.randn(100_000_000)

    # 1. Standard PyTorch
    print("Running PyTorch Eager...")
    start = time.time()
    for _ in range(3):
        out_eager = model(x)
    pytorch_time = (time.time() - start) / 3
    print(f"PyTorch Eager Time: {pytorch_time:.4f}s")

    # 2. XIM Backend
    print("Compiling with XIM...")
    try:
        compiled_model = torch.compile(model, backend=xim_torch.xim_backend)
        
        print("Warmup XIM...")
        compiled_model(x) # Triggers compilation
        
        print("Running XIM...")
        start = time.time()
        for _ in range(3):
            out_xim = compiled_model(x)
        xim_time = (time.time() - start) / 3
        print(f"XIM Backend Time: {xim_time:.4f}s")
        print(f"Speedup: {pytorch_time / xim_time:.2f}x")
    except Exception as e:
        print(f"XIM Compilation failed: {e}")

if __name__ == "__main__":
    benchmark()
