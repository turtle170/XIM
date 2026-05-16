use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray, PyArrayMethods};
use crate::ir::{XimGraph, OpCode, XimBuilder};
use crate::executor::Executor;
use crate::quant::{Quantizer, Calibrator};
use crate::aot::AotExecutor;
use crate::pipeline::PipelineExecutor;
use crate::jit::CraneliftCompiler;
use tracing_subscriber;

#[pyclass]
pub struct XimPipelineEngine {
    pipeline: PipelineExecutor,
}

#[pymethods]
impl XimPipelineEngine {
    #[new]
    pub fn new(
        graph_paths: Vec<String>,
        parameters: Vec<PyReadonlyArray1<'_, f32>>,
        layer_input_sizes: Vec<usize>,
        layer_output_sizes: Vec<usize>,
    ) -> PyResult<Self> {
        let _ = tracing_subscriber::fmt::try_init();
        
        use rayon::prelude::*;
        let mut graphs = Vec::new();
        for path in graph_paths {
            let graph = XimGraph::load_from_file(&path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            graphs.push(graph);
        }

        let mut q_params = Vec::new();
        for p in parameters {
            let slice = p.as_slice()?;
            let q_data: Vec<i16> = slice.par_iter().map(|&x| Quantizer::to_i16(x)).collect();
            q_params.push(q_data);
        }

        let pipeline = PipelineExecutor::new(graphs, q_params, layer_input_sizes, layer_output_sizes);
        
        Ok(Self { pipeline })
    }

    pub fn execute<'py>(
        &self, 
        py: Python<'py>, 
        inputs: PyReadonlyArray1<'py, f32>, 
        batch_size: usize
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = inputs.as_slice()?;
        
        use rayon::prelude::*;
        let q_inputs: Vec<i16> = input_slice.par_iter().map(|&x| Quantizer::to_i16(x)).collect();

        let q_outputs = self.pipeline.execute(&q_inputs, batch_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        let f_outputs: Vec<f32> = q_outputs.par_iter().map(|&x| Quantizer::to_f32(x)).collect();
        Ok(f_outputs.into_pyarray(py))
    }
}

#[pyclass]
pub struct XimEngine {
    graph: XimGraph,
    executor: Executor,
    aot_executor: Option<AotExecutor>,
    jit_func: Option<usize>,
    parameters: std::collections::HashMap<u32, Vec<i16>>,
    input_buffer: Vec<i16>,
    output_buffer: Vec<i16>,
}

#[pymethods]
impl XimEngine {
    #[new]
    pub fn new(xim_path: &str) -> PyResult<Self> {
        let _ = tracing_subscriber::fmt::try_init();
        let graph = XimGraph::load_from_file(xim_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        let executor = Executor::new(graph.memory_size);
        Ok(Self { 
            graph, 
            executor, 
            aot_executor: None,
            jit_func: None,
            parameters: std::collections::HashMap::new(),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        })
    }

    pub fn compile_jit(&mut self) -> PyResult<()> {
        let mut compiler = CraneliftCompiler::new();
        let code_ptr = compiler.compile(&self.graph)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Cranelift JIT failed: {}", e)))?;
        self.jit_func = Some(code_ptr as usize);
        Ok(())
    }

    pub fn compile_aot(&mut self) -> PyResult<()> {
        let aot_exec = AotExecutor::new(&self.graph)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("AOT Compilation failed: {}", e)))?;
        self.aot_executor = Some(aot_exec);
        Ok(())
    }

    pub fn set_parameter(&mut self, id: u32, data: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        use rayon::prelude::*;
        let slice = data.as_slice()?;
        let q_data: Vec<i16> = slice.par_iter().map(|&x| Quantizer::to_i16(x)).collect();
        self.parameters.insert(id, q_data);
        Ok(())
    }

    pub fn execute_graph_with_params<'py>(&mut self, py: Python<'py>, inputs: PyReadonlyArray1<'py, f32>, output_size: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = inputs.as_slice()?;
        let dyn_len = input_slice.len();
        
        let mut total_param_len = 0;
        for p in self.parameters.values() { total_param_len += p.len(); }
        let total_input_len = dyn_len + total_param_len;

        let first_time = self.input_buffer.len() != total_input_len;
        if first_time {
            self.input_buffer.resize(total_input_len, 0);
        }

        use rayon::prelude::*;
        let (dyn_part, param_part) = self.input_buffer.split_at_mut(dyn_len);
        
        let input_usize = input_slice.as_ptr() as usize;
        dyn_part.par_chunks_mut(65536).enumerate().for_each(|(chunk_idx, chunk)| {
            let input_ptr = input_usize as *const f32;
            let start = chunk_idx * 65536;
            for i in 0..chunk.len() {
                chunk[i] = Quantizer::to_i16(unsafe { *input_ptr.add(start + i) });
            }
        });

        if first_time {
            let mut curr_off = 0;
            let mut sorted_keys: Vec<_> = self.parameters.keys().collect();
            sorted_keys.sort();
            for k in sorted_keys {
                let p = self.parameters.get(k).unwrap();
                param_part[curr_off..curr_off + p.len()].copy_from_slice(p);
                curr_off += p.len();
            }
        }

        if self.output_buffer.len() != output_size {
            self.output_buffer.resize(output_size, 0);
        }

        if let Some(jit_ptr) = self.jit_func {
            let func: unsafe extern "C" fn(*mut i16, *const i16, *mut i16) = unsafe { std::mem::transmute(jit_ptr) };
            unsafe { func(self.executor.memory.as_mut_ptr(), self.input_buffer.as_ptr(), self.output_buffer.as_mut_ptr()); }
        } else if let Some(ref mut aot) = self.aot_executor {
            aot.execute(&self.input_buffer, &mut self.output_buffer)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("AOT execution failed: {}", e)))?;
        } else {
            self.executor.execute(&self.graph, &self.input_buffer, &mut self.output_buffer)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        }

        let py_array = unsafe { PyArray1::<f32>::new(py, [output_size], false) };
        let out_usize = unsafe { py_array.as_array_mut().as_mut_ptr() as usize };
        
        self.output_buffer.par_chunks(65536).enumerate().for_each(|(chunk_idx, chunk)| {
            let out_ptr = out_usize as *mut f32;
            let start = chunk_idx * 65536;
            for i in 0..chunk.len() {
                unsafe { *out_ptr.add(start + i) = Quantizer::to_f32(chunk[i]); }
            }
        });

        Ok(py_array)
    }

    pub fn execute_graph_i8<'py>(&mut self, py: Python<'py>, inputs: PyReadonlyArray1<'py, f32>, output_size: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = inputs.as_slice()?;
        
        let q_inputs: Vec<i8> = input_slice.iter().map(|&x| Quantizer::to_i8(x)).collect();
        let mut q_outputs = vec![0i8; output_size];

        self.executor.execute_i8(&self.graph, &q_inputs, &mut q_outputs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        let f_outputs: Vec<f32> = q_outputs.iter().map(|&x| Quantizer::to_f32_i8(x)).collect();
        Ok(f_outputs.into_pyarray(py))
    }
}

#[pyclass]
pub struct XimCompiler {
    builder: Option<XimBuilder>,
}

#[pymethods]
impl XimCompiler {
    #[new]
    pub fn new() -> Self {
        Self { builder: Some(XimBuilder::new()) }
    }

    pub fn add_load_vec(&mut self, var_id: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::LoadVarVec(var_id, dst, len));
        }
    }

    pub fn add_load_vec8(&mut self, var_id: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::LoadVarVec8(var_id, dst, len));
        }
    }

    pub fn add_store_vec(&mut self, src: u32, out_id: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::StoreVec(src, out_id, len));
        }
    }

    pub fn add_store_vec8(&mut self, src: u32, out_id: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::StoreVec8(src, out_id, len));
        }
    }

    pub fn add_add_vec(&mut self, lhs: u32, rhs: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::AddVec(lhs, rhs, dst, len));
        }
    }

    pub fn add_add_vec8(&mut self, lhs: u32, rhs: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::AddVec8(lhs, rhs, dst, len));
        }
    }

    pub fn add_mul_vec(&mut self, lhs: u32, rhs: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::MulVec(lhs, rhs, dst, len));
        }
    }

    pub fn add_mul_vec8(&mut self, lhs: u32, rhs: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::MulVec8(lhs, rhs, dst, len));
        }
    }

    pub fn add_aio_step(&mut self, weight: u32, grad: u32, m: u32, v: u32, len: u32, precision: u8) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::AIOStep(weight, grad, m, v, len, precision));
        }
    }

    pub fn add_block_scale(&mut self, src: u32, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::BlockScale(src, dst, len));
        }
    }

    pub fn add_fill_vec(&mut self, val: i16, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::FillVec(val, dst, len));
        }
    }

    pub fn add_add_scalar_vec(&mut self, lhs: u32, val: i16, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::AddScalarVec(lhs, val, dst, len));
        }
    }

    pub fn add_mul_scalar_vec(&mut self, lhs: u32, val: i16, dst: u32, len: u32) {
        if let Some(ref mut b) = self.builder {
            b.add_op(OpCode::MulScalarVec(lhs, val, dst, len));
        }
    }

    pub fn compile_and_save(&mut self, path: &str) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            let graph = builder.finalize();
            graph.save_to_file(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Builder already finalized"))
        }
    }
}

#[pyclass]
pub struct XimCalibrator {
    inner: Calibrator,
}

#[pymethods]
impl XimCalibrator {
    #[new]
    pub fn new() -> Self {
        Self { inner: Calibrator::new() }
    }

    pub fn observe(&mut self, data: PyReadonlyArray1<'_, f32>) {
        if let Ok(slice) = data.as_slice() {
            self.inner.observe_slice(slice);
        }
    }

    pub fn get_scale(&self) -> f32 {
        self.inner.get_scale()
    }
}

#[pymodule]
fn xim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<XimEngine>()?;
    m.add_class::<XimPipelineEngine>()?;
    m.add_class::<XimCompiler>()?;
    m.add_class::<XimCalibrator>()?;
    Ok(())
}
