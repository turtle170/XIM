use xim::{Executor, Instruction, OpCode, XimGraph, MemoryPlanner, StochasticQuantizer, Quantizer};

#[test]
fn test_stochastic_rounding_convergence() {
    let mut q = StochasticQuantizer::new(12345);
    
    // If we accumulate many tiny gradients that are smaller than 1/256 (the Q8.8 resolution),
    // normal rounding would clip them to 0.0 and weights would never change.
    // Stochastic rounding should allow the weight to eventually update.
    
    let scale = 256.0;
    let small_grad_f32 = 0.001; // 0.001 * 256 = 0.256. Normal round is 0.
    
    let mut accumulated_updates = 0;
    for _ in 0..1000 {
        let update = q.to_i16_stochastic(small_grad_f32, scale);
        accumulated_updates += update as i32;
    }
    
    // Expected expected value is ~0.256 * 1000 = 256
    assert!(accumulated_updates > 200 && accumulated_updates < 300, "Accumulated updates: {}", accumulated_updates);
}

#[test]
fn test_adamw_convergence_mock() {
    let tensor_len = 10;
    let mut graph = XimGraph::new();
    
    graph.instructions = vec![
        Instruction { op: OpCode::LoadVarVec(0, 0, tensor_len) }, // W
        Instruction { op: OpCode::LoadVarVec(10, 10, tensor_len) }, // G
        Instruction { op: OpCode::LoadVarVec(20, 20, tensor_len) }, // M
        Instruction { op: OpCode::LoadVarVec(30, 30, tensor_len) }, // V
        Instruction { op: OpCode::AdamWStep(0, 10, 20, 30, tensor_len) },
        Instruction { op: OpCode::StoreVec(0, 0, tensor_len) }, // Output new W
    ];
    MemoryPlanner::plan(&mut graph);
    
    let mut executor = Executor::new(graph.memory_size);
    let mut inputs = vec![0i16; 40];
    
    // Init weights to 1.0 (256 in Q8.8)
    for i in 0..10 { inputs[i] = 256; }
    
    // Provide a constant positive gradient
    let grad_val = Quantizer::to_i16(0.1); // ~26
    for i in 10..20 { inputs[i] = grad_val; }
    
    let mut outputs = vec![0i16; 10];
    
    // Run AdamW step
    for _ in 0..10 {
        executor.execute(&graph, &inputs, &mut outputs).unwrap();
        // Feedback W
        inputs[0..10].copy_from_slice(&outputs);
        // Feedback M
        // M is at internal offset 20. For this mock test we don't have store instructions for M and V.
        // We'll just verify W decreased due to positive gradient.
    }
    
    // W should have decreased
    assert!(outputs[0] < 256, "Weight should decrease: {}", outputs[0]);
}
