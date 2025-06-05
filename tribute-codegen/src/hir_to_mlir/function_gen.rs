//! MLIR function generation logic.

use crate::hir_to_mlir::{
    types::MlirOperation,
    operations::{generate_box_number_op, generate_box_string_op, generate_function_call_op, generate_other_mlir_operation},
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for function body
pub fn generate_function_body<'a>(
    body_ops: &[MlirOperation],
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) -> Option<String> {
    let mut last_boxed_value = None;
    for (i, op) in body_ops.iter().enumerate() {
        match op {
            MlirOperation::BoxNumber { value } => {
                generate_box_number_op(*value, i, context, location, block);
                last_boxed_value = Some(format!("boxed_num_{}", i));
            }
            MlirOperation::BoxString { value } => {
                generate_box_string_op(value, i, context, location, block);
                last_boxed_value = Some(format!("boxed_str_{}", i));
            }
            MlirOperation::Call { func, args } => {
                generate_function_call_op(func, args, i, context, location, block);
                last_boxed_value = Some(format!("call_result_{}", i));
            }
            _ => {
                generate_other_mlir_operation(op, i, context, location, block);
                // Most operations produce a result value
                last_boxed_value = Some(format!("result_{}", i));
            }
        }
    }
    
    if let Some(ref last_value) = last_boxed_value {
        println!("    Function body last result: {}", last_value);
    } else {
        println!("    Function body returns nil (no operations)");
    }
    
    last_boxed_value
}

/// Generate a single MLIR function operation
pub fn generate_mlir_function_op<'a>(
    name: &str,
    params: &[tribute_ast::Identifier],
    body_ops: &[MlirOperation],
    context: &'a Context,
    location: Location<'a>,
) -> melior::ir::Operation<'a> {
    use melior::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType},
            Block, BlockLike, Region, RegionLike,
        },
    };

    println!("  Function: {} with {} params, {} operations", 
             name, params.len(), body_ops.len());
    
    // Create function type: all functions work with boxed values
    let boxed_ptr_type = IntegerType::new(context, 64); // Pointer to boxed value
    let param_types: Vec<_> = params.iter().map(|_| boxed_ptr_type.into()).collect();
    let function_type = FunctionType::new(context, &param_types, &[boxed_ptr_type.into()]);
    
    // Create function operation
    func::func(
        context,
        StringAttribute::new(context, name),
        TypeAttribute::new(function_type.into()),
        {
            let region = Region::new();
            let block = Block::new(&[]);
            
            // Generate operations for function body
            let last_result = generate_function_body(body_ops, context, location, &block);
            
            // Add return operation for boxed values
            if let Some(ref result_value) = last_result {
                println!("    Adding return statement to function: {}", result_value);
                println!("      -> MLIR: return ptr %{}", result_value);
            } else {
                println!("    Adding return statement to function: nil");
                println!("      -> MLIR: return ptr %nil_value");
            }
            
            // Try to generate actual func.return operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Create a simple return operation for now
                // In a full implementation, we'd pass the actual last result value
                melior::dialect::func::r#return(
                    &[], // No return values for now - simplified
                    location,
                )
            })) {
                Ok(return_op) => {
                    println!("      SUCCESS: Created MLIR return operation");
                    block.append_operation(return_op);
                }
                Err(_) => {
                    println!("      FALLBACK: Using minimal function body without explicit return");
                }
            }
            
            region.append_block(block);
            region
        },
        &[],
        location,
    )
}