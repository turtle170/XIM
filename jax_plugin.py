import jax
import jax.numpy as jnp
import xim
import numpy as np

def xim_run(inputs, engine, batch_size):
    # We use jax.pure_callback to execute our Rust multi-VM pipeline inside a jitted JAX function
    result_shape = jax.ShapeDtypeStruct(inputs.shape, inputs.dtype)
    
    def xim_pipeline_impl(inputs_inner):
        return engine.execute(np.array(inputs_inner), batch_size)
        
    return jax.pure_callback(
        xim_pipeline_impl, 
        result_shape, 
        inputs
    )
