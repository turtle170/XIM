use crate::ir::XimGraph;
use crate::executor::Executor;
use crate::error::Result;
use crossbeam_channel::bounded;
use std::thread;

pub struct PipelineExecutor {
    graphs: Vec<XimGraph>,
    parameters: Vec<Vec<i16>>,
    layer_input_sizes: Vec<usize>,
    layer_output_sizes: Vec<usize>,
}

impl PipelineExecutor {
    pub fn new(
        graphs: Vec<XimGraph>, 
        parameters: Vec<Vec<i16>>,
        layer_input_sizes: Vec<usize>,
        layer_output_sizes: Vec<usize>,
    ) -> Self {
        Self {
            graphs,
            parameters,
            layer_input_sizes,
            layer_output_sizes,
        }
    }

    /// Execute pipeline by chunking the batch size.
    /// `inputs` is [batch_size * input_dim]
    /// returns output [batch_size * output_dim]
    pub fn execute(&self, inputs: &[i16], batch_size: usize) -> Result<Vec<i16>> {
        let num_layers = self.graphs.len();
        if num_layers == 0 {
            return Ok(inputs.to_vec());
        }

        let _input_dim = self.layer_input_sizes[0];
        let output_dim = *self.layer_output_sizes.last().unwrap();
        
        let mut final_outputs = vec![0i16; batch_size * output_dim];

        // Create channels between layers
        // We will process token by token (or chunk of tokens)
        // Let's process chunk of size 1 (1 token) for max pipeline concurrency
        let chunk_size = 1;
        let num_chunks = batch_size / chunk_size;

        let mut senders = Vec::new();
        let mut receivers = Vec::new();

        for _ in 0..num_layers {
            let (tx, rx) = bounded::<Vec<i16>>(2); // small buffer for streaming
            senders.push(tx);
            receivers.push(rx);
        }
        
        let (final_tx, final_rx) = bounded::<(usize, Vec<i16>)>(batch_size);

        thread::scope(|s| {
            // Spawn Layer Threads
            for i in 0..num_layers {
                let rx = if i == 0 { None } else { Some(receivers[i - 1].clone()) };
                let tx = senders[i].clone();
                let graph = &self.graphs[i];
                let params = &self.parameters[i];
                let in_size = self.layer_input_sizes[i];
                let out_size = self.layer_output_sizes[i];
                let final_tx = final_tx.clone();
                let is_last = i == num_layers - 1;

                s.spawn(move || {
                    let mut executor = Executor::new(graph.memory_size);
                    let mut input_buffer = vec![0i16; in_size + params.len()];
                    input_buffer[in_size..].copy_from_slice(params);
                    
                    let mut output_buffer = vec![0i16; out_size];

                    for c in 0..num_chunks {
                        let token_data = if let Some(ref r) = rx {
                            r.recv().unwrap()
                        } else {
                            // Layer 0 receives from main thread
                            vec![] // Handled below
                        };

                        if i != 0 {
                            input_buffer[..in_size].copy_from_slice(&token_data);
                        } else {
                            // wait for signal from main or read directly from shared inputs?
                            // Actually for Layer 0 we can read directly
                            let start = c * in_size;
                            input_buffer[..in_size].copy_from_slice(&inputs[start..start + in_size]);
                        }

                        // Execute VM for this layer
                        executor.execute(graph, &input_buffer, &mut output_buffer).unwrap();

                        if is_last {
                            final_tx.send((c, output_buffer.clone())).unwrap();
                        } else {
                            tx.send(output_buffer.clone()).unwrap();
                        }
                    }
                });
            }

            // Main thread collects results
            for _ in 0..num_chunks {
                let (c, out_data) = final_rx.recv().unwrap();
                let start = c * output_dim;
                final_outputs[start..start + output_dim].copy_from_slice(&out_data);
            }
        });

        Ok(final_outputs)
    }
}
