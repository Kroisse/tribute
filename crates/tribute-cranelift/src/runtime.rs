//! Runtime function declarations and helpers
//!
//! This module defines the external runtime functions that compiled
//! Tribute code will call for operations like memory allocation,
//! string manipulation, and built-in functions.

use cranelift_codegen::ir::{AbiParam, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module};

use crate::errors::{BoxError, CompilationResult};
use crate::types::TributeTypes;

/// Runtime function signatures and IDs
pub struct RuntimeFunctions {
    /// Allocate a new value
    pub value_new: FuncId,
    /// Free a value
    pub value_free: FuncId,
    /// Create a number value
    pub value_from_number: FuncId,
    /// Create a string value
    pub value_from_string: FuncId,
    /// Create a static string value from .rodata section
    pub value_from_static_string: FuncId,
    /// Get number from value (with type check)
    pub value_to_number: FuncId,
    /// Clone a value (for reference counting)
    pub value_clone: FuncId,
    
    /// Arithmetic operations
    pub value_add: FuncId,
    pub value_sub: FuncId,
    pub value_mul: FuncId,
    pub value_div: FuncId,
    
    /// String operations
    pub string_concat: FuncId,
    pub string_interpolate: FuncId,
    
    /// Built-in functions
    pub builtin_print_line: FuncId,
    pub builtin_input_line: FuncId,
}

impl RuntimeFunctions {
    /// Declare all runtime functions in the module
    pub fn declare_all<M: Module>(module: &mut M) -> CompilationResult<Self> {
        let pointer = TributeTypes::pointer_type();
        let call_conv = CallConv::SystemV;  // TODO: Make platform-specific
        
        // Helper to create signatures
        let sig_with_params = |params: Vec<AbiParam>, returns: Vec<AbiParam>| {
            let mut sig = Signature::new(call_conv);
            sig.params.extend(params);
            sig.returns.extend(returns);
            sig
        };
        
        // value_new() -> TrHandle
        let value_new = module.declare_function(
            "tr_value_new",
            Linkage::Import,
            &sig_with_params(vec![], vec![AbiParam::new(pointer)])
        ).box_err()?;
        
        // value_free(handle: TrHandle) -> void
        let value_free = module.declare_function(
            "tr_value_free",
            Linkage::Import,
            &sig_with_params(vec![AbiParam::new(pointer)], vec![])
        ).box_err()?;
        
        // value_from_number(num: f64) -> TrHandle
        let value_from_number = module.declare_function(
            "tr_value_from_number",
            Linkage::Import,
            &sig_with_params(
                vec![TributeTypes::number_param()],
                vec![AbiParam::new(pointer)]
            )
        ).box_err()?;
        
        // value_from_string(data: *char, len: i64) -> TrHandle
        let value_from_string = module.declare_function(
            "tr_value_from_string",
            Linkage::Import,
            &sig_with_params(
                vec![AbiParam::new(pointer), AbiParam::new(TributeTypes::size_type())],
                vec![AbiParam::new(pointer)]
            )
        ).box_err()?;
        
        // value_from_static_string(offset: u32, len: u32) -> TrHandle
        let value_from_static_string = module.declare_function(
            "tr_value_from_static_string",
            Linkage::Import,
            &sig_with_params(
                vec![AbiParam::new(cranelift_codegen::ir::types::I32), AbiParam::new(cranelift_codegen::ir::types::I32)],
                vec![AbiParam::new(pointer)]
            )
        ).box_err()?;
        
        // value_to_number(handle: TrHandle) -> f64
        let value_to_number = module.declare_function(
            "tr_value_to_number",
            Linkage::Import,
            &sig_with_params(
                vec![AbiParam::new(pointer)],
                vec![TributeTypes::number_param()]
            )
        ).box_err()?;
        
        // value_clone(handle: TrHandle) -> TrHandle
        let value_clone = module.declare_function(
            "tr_value_clone",
            Linkage::Import,
            &sig_with_params(
                vec![AbiParam::new(pointer)],
                vec![AbiParam::new(pointer)]
            )
        ).box_err()?;
        
        // Arithmetic operations: (left: TrHandle, right: TrHandle) -> TrHandle
        let binary_op_sig = sig_with_params(
            vec![AbiParam::new(pointer), AbiParam::new(pointer)],
            vec![AbiParam::new(pointer)]
        );
        
        let value_add = module.declare_function("tr_value_add", Linkage::Import, &binary_op_sig).box_err()?;
        let value_sub = module.declare_function("tr_value_sub", Linkage::Import, &binary_op_sig).box_err()?;
        let value_mul = module.declare_function("tr_value_mul", Linkage::Import, &binary_op_sig).box_err()?;
        let value_div = module.declare_function("tr_value_div", Linkage::Import, &binary_op_sig).box_err()?;
        
        // String operations
        let string_concat = module.declare_function(
            "tr_string_concat",
            Linkage::Import,
            &binary_op_sig
        ).box_err()?;
        
        // string_interpolate(format: TrHandle, args: *TrHandle, count: i64) -> TrHandle
        let string_interpolate = module.declare_function(
            "tr_string_interpolate",
            Linkage::Import,
            &sig_with_params(
                vec![AbiParam::new(pointer), AbiParam::new(pointer), AbiParam::new(TributeTypes::size_type())],
                vec![AbiParam::new(pointer)]
            )
        ).box_err()?;
        
        // Built-in functions
        // print_line(handle: TrHandle) -> void
        let builtin_print_line = module.declare_function(
            "tr_builtin_print_line",
            Linkage::Import,
            &sig_with_params(vec![AbiParam::new(pointer)], vec![])
        ).box_err()?;
        
        // input_line() -> TrHandle
        let builtin_input_line = module.declare_function(
            "tr_builtin_input_line",
            Linkage::Import,
            &sig_with_params(vec![], vec![AbiParam::new(pointer)])
        ).box_err()?;
        
        Ok(RuntimeFunctions {
            value_new,
            value_free,
            value_from_number,
            value_from_string,
            value_from_static_string,
            value_to_number,
            value_clone,
            value_add,
            value_sub,
            value_mul,
            value_div,
            string_concat,
            string_interpolate,
            builtin_print_line,
            builtin_input_line,
        })
    }
}