import xim
import numpy as np

print("--- XIM Python FFI Test ---")

# 1. Initialize Engine
try:
    engine = xim.XimEngine("vector_test.xim")
    print("Engine initialized successfully.")

    # 2. Prepare Inputs
    # We used 30000 inputs in the Rust test (A, B, C stacked)
    a = np.full(10000, 1.2, dtype=np.float32)
    b = np.full(10000, 2.3, dtype=np.float32)
    c = np.full(10000, 0.5, dtype=np.float32)
    inputs = np.concatenate([a, b, c])

    # 3. Execute
    results = engine.execute_graph(inputs)
    
    # 4. Verify
    print(f"Result shape: {results.shape}")
    print(f"Sample Result: {results[0]} (Expected: {(1.2 + 2.3) * 0.5})")

except Exception as e:
    print(f"Error: {e}")
