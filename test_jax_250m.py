import jax
import jax.numpy as jnp
import xim
import numpy as np
import time
import os
from jax_plugin import xim_run

# 250M Parameter Model Configuration
# Assume a Transformer-like structure
NUM_LAYERS = 12
HIDDEN_DIM = 2048 # ~4.2M params per layer roughly, wait, standard is 12 layers of 1024 to get ~150M. Let's do 12 layers of 2048 to get ~250M.
BATCH_SIZE = 32

def generate_layer_graph(hidden_dim):
    # Generates a dummy .xim IR representing a single layer computation (e.g., Matrix Multiply + Add)
    compiler = xim.XimCompiler()
    
    # Simple FFN block approximation
    # Input size: hidden_dim
    # Params size: hidden_dim * 2
    
    compiler.add_load_vec(0, 0, hidden_dim) # Load inputs
    compiler.add_load_vec(hidden_dim, hidden_dim, hidden_dim) # Load weights 1
    compiler.add_mul_vec(0, hidden_dim, 0, hidden_dim) # inputs * weights 1
    
    compiler.add_load_vec(hidden_dim * 2, hidden_dim, hidden_dim) # Load weights 2
    compiler.add_add_vec(0, hidden_dim, 0, hidden_dim) # result + weights 2
    
    compiler.add_store_vec(0, 0, hidden_dim) # Store result
    
    path = f"layer_{os.getpid()}.xim"
    compiler.compile_and_save(path)
    return path

def setup_xim_pipeline():
    print("Setting up XIM Pipelined Multi-VM Backend...")
    layer_path = generate_layer_graph(HIDDEN_DIM)
    
    graphs = [layer_path] * NUM_LAYERS
    params = [np.random.randn(HIDDEN_DIM * 2).astype(np.float32) for _ in range(NUM_LAYERS)]
    in_sizes = [HIDDEN_DIM] * NUM_LAYERS
    out_sizes = [HIDDEN_DIM] * NUM_LAYERS
    
    engine = xim.XimPipelineEngine(graphs, params, in_sizes, out_sizes)
    return engine

def xla_layer(x, w1, w2):
    return (x * w1) + w2

@jax.jit
def xla_model(x, weights):
    for i in range(NUM_LAYERS):
        x = xla_layer(x, weights[i][0], weights[i][1])
    return x

@jax.jit(static_argnames=('engine',))
def xim_model(x, engine):
    return xim_run(x, engine, BATCH_SIZE)

def main():
    print("--- XIM vs XLA JAX Benchmark (250M Pipelined) ---")
    np.random.seed(42)
    
    inputs = np.random.randn(BATCH_SIZE * HIDDEN_DIM).astype(np.float32)
    inputs_jax = jnp.array(inputs)
    
    weights_xla = []
    for _ in range(NUM_LAYERS):
        weights_xla.append((
            jnp.array(np.random.randn(BATCH_SIZE * HIDDEN_DIM).astype(np.float32)), 
            jnp.array(np.random.randn(BATCH_SIZE * HIDDEN_DIM).astype(np.float32))
        ))
    
    # 1. Benchmark XLA
    print("Warming up XLA...")
    _ = xla_model(inputs_jax, weights_xla).block_until_ready()
    
    print("Running XLA Benchmark...")
    start_xla = time.time()
    _ = xla_model(inputs_jax, weights_xla).block_until_ready()
    xla_time = time.time() - start_xla
    print(f"XLA Time: {xla_time:.4f}s")
    
    # 2. Benchmark XIM
    engine = setup_xim_pipeline()
    print("Warming up XIM Pipeline...")
    _ = xim_model(inputs_jax, engine).block_until_ready()
    
    print("Running XIM Pipelined Benchmark...")
    start_xim = time.time()
    _ = xim_model(inputs_jax, engine).block_until_ready()
    xim_time = time.time() - start_xim
    print(f"XIM Time: {xim_time:.4f}s")
    
    print(f"\nSpeedup (XLA / XIM): {xla_time / xim_time:.2f}x")

if __name__ == "__main__":
    main()
