import torch
import torch.fx
import xim
import numpy as np
import os
from torch.fx.node import Node
import operator

class XimBackend:
    def __init__(self):
        self.compiler = xim.XimCompiler()
        self.memory_offset = 0
        self.node_to_offset = {}
        self.output_size = 0
        self.placeholders = []

    def compile(self, gm: torch.fx.GraphModule, example_inputs):
        print("--- XIM TorchDynamo Backend Lowering (500M Param Storage) ---")
        
        # 1. First pass: Identify placeholders and parameters
        input_offset = 0
        params_info = [] # (node, numel)
        
        # We need to maintain a specific order: [dynamic_inputs..., static_params...]
        dynamic_nodes = []
        static_nodes = []
        
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                dynamic_nodes.append(node)
            elif node.op == 'get_attr':
                static_nodes.append(node)

        # Dynamic Inputs first
        for node in dynamic_nodes:
            numel = self._get_numel(node)
            self.node_to_offset[node] = input_offset
            input_offset += numel
            self.placeholders.append(node)

        # Static parameters next
        param_start_offset = input_offset
        for node in static_nodes:
            numel = self._get_numel(node)
            self.node_to_offset[node] = input_offset
            input_offset += numel
            self.placeholders.append(node)

        self.memory_offset = 0

        # 2. Second pass: Emit instructions
        for node in gm.graph.nodes:
            if node.op == 'placeholder' or node.op == 'get_attr':
                dst = self._alloc_mem(node)
                src_idx = self.node_to_offset[node]
                numel = self._get_numel(node)
                self.compiler.add_load_vec(src_idx, dst, numel)
                self.node_to_offset[node] = dst
                
            elif node.op == 'call_function':
                self._handle_call_function(node)
            
            elif node.op == 'output':
                self._handle_output(node)

        # 3. Finalize and Build Engine
        xim_path = f"torch_model_{os.getpid()}.xim"
        self.compiler.compile_and_save(xim_path)
        engine = xim.XimEngine(xim_path)
        
        # Enable Cranelift JIT Compilation
        try:
            print("Compiling XIM graph to Native Assembly (Cranelift JIT)...")
            engine.compile_jit()
        except Exception as e:
            print(f"Warning: JIT Compilation failed, falling back to VM. Error: {e}")

        # Pre-load parameters into engine
        for i, node in enumerate(static_nodes):
            attr = getattr(gm, node.target)
            engine.set_parameter(i, attr.detach().cpu().numpy().flatten().astype(np.float32))

        def run(*args):
            # args are dynamic placeholders
            if len(args) == 1:
                flat_dynamic = args[0].detach().cpu().numpy().flatten().astype(np.float32)
            else:
                flat_dynamic = np.concatenate([a.detach().cpu().numpy().flatten() for a in args]).astype(np.float32)
            
            results = engine.execute_graph_with_params(flat_dynamic, self.output_size)
            return (torch.from_numpy(results).to(args[0].device),)

        return run

    def _get_numel(self, node):
        if hasattr(node, 'meta') and 'val' in node.meta:
            v = node.meta['val']
            if hasattr(v, 'numel'):
                return v.numel()
        return 100_000_000 

    def _alloc_mem(self, node):
        numel = self._get_numel(node)
        offset = self.memory_offset
        self.node_to_offset[node] = offset
        self.memory_offset += numel
        return offset

    def _handle_call_function(self, node):
        target = node.target
        is_add = target in [torch.ops.aten.add.Tensor, torch.ops.aten.add.default, torch.ops.aten.add, operator.add, "add"]
        is_mul = target in [torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default, torch.ops.aten.mul, operator.mul, "mul"]
        
        if is_add:
            self._handle_bin_op(node, "add")
        elif is_mul:
            self._handle_bin_op(node, "mul")

    def _handle_bin_op(self, node, op_type):
        args = node.args
        lhs_node = args[0]
        rhs_node = args[1]
        numel = self._get_numel(node)
        
        is_lhs_scalar = not isinstance(lhs_node, Node)
        is_rhs_scalar = not isinstance(rhs_node, Node)

        if is_lhs_scalar or is_rhs_scalar:
            scalar_val = lhs_node if is_lhs_scalar else rhs_node
            vec_node = rhs_node if is_lhs_scalar else lhs_node
            
            vec_off = self.node_to_offset.get(vec_node)
            if vec_off is None:
                return

            q_val = int(float(scalar_val) * 256.0)
            dst = self._alloc_mem(node)
            
            if op_type == "add":
                self.compiler.add_add_scalar_vec(vec_off, q_val, dst, numel)
            else:
                self.compiler.add_mul_scalar_vec(vec_off, q_val, dst, numel)
            return

        lhs_off = self.node_to_offset.get(lhs_node)
        rhs_off = self.node_to_offset.get(rhs_node)
        
        if lhs_off is None or rhs_off is None:
            return

        dst = self._alloc_mem(node)
        if op_type == "add":
            self.compiler.add_add_vec(lhs_off, rhs_off, dst, numel)
        else:
            self.compiler.add_mul_vec(lhs_off, rhs_off, dst, numel)

    def _get_or_create_const(self, arg, numel):
        if isinstance(arg, Node):
            return self.node_to_offset.get(arg)
        try:
            val = float(arg)
            q_val = int(val * 256.0)
            dst = self.memory_offset
            self.memory_offset += numel
            self.compiler.add_fill_vec(q_val, dst, numel)
            return dst
        except:
            return None

    def _handle_output(self, node):
        src_node = node.args[0]
        if isinstance(src_node, (list, tuple)):
            src_node = src_node[0]
        
        src_off = self.node_to_offset.get(src_node)
        if src_off is not None:
            numel = self._get_numel(src_node)
            self.compiler.add_store_vec(src_off, 0, numel)
            self.output_size = numel

def xim_backend(gm: torch.fx.GraphModule, example_inputs):
    return XimBackend().compile(gm, example_inputs)
