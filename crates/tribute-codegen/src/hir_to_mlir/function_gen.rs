//! MLIR function generation logic.

use crate::hir_to_mlir::{
    types::MlirOperation,
    operations::{generate_box_number_op, generate_box_string_op, generate_function_call_op, generate_other_mlir_operation},
};
use melior::{
    ir::{Block, Location},
    Context,
};
use tribute_ast::{CompilationPhase, Db, Diagnostic};

/// Generate MLIR operations for function body
pub fn generate_function_body<'a>(
    db: &dyn Db,
    body_ops: &[MlirOperation],
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) -> Option<String> {
    let mut last_boxed_value = None;
    for (i, op) in body_ops.iter().enumerate() {
        match op {
            MlirOperation::BoxNumber { value } => {
                generate_box_number_op(db, *value, i, context, location, block);
                last_boxed_value = Some(format!("boxed_num_{}", i));
            }
            MlirOperation::BoxString { value } => {
                generate_box_string_op(db, value, i, context, location, block);
                last_boxed_value = Some(format!("boxed_str_{}", i));
            }
            MlirOperation::Call { func, args } => {
                generate_function_call_op(db, func, args, i, context, location, block);
                last_boxed_value = Some(format!("call_result_{}", i));
            }
            _ => {
                generate_other_mlir_operation(db, op, i, context, location, block);
                // Most operations produce a result value
                last_boxed_value = Some(format!("result_{}", i));
            }
        }
    }
    
    if let Some(ref last_value) = last_boxed_value {
        Diagnostic::debug()
            .message(format!("Function body last result: {}", last_value))
            .phase(CompilationPhase::HirLowering)
            .accumulate(db);
    } else {
        Diagnostic::debug()
            .message("Function body returns nil (no operations)")
            .phase(CompilationPhase::HirLowering)
            .accumulate(db);
    }
    
    last_boxed_value
}

/// Generate a single MLIR function operation
pub fn generate_mlir_function_op<'a>(
    db: &dyn Db,
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

    Diagnostic::debug()
        .message(format!("Function: {} with {} params, {} operations", 
                         name, params.len(), body_ops.len()))
        .phase(CompilationPhase::HirLowering)
        .accumulate(db);
    
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
            let last_result = generate_function_body(db, body_ops, context, location, &block);
            
            // Add return operation for boxed values
            if let Some(ref result_value) = last_result {
                Diagnostic::debug()
                    .message(format!("Adding return statement to function: {}", result_value))
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
                Diagnostic::debug()
                    .message(format!("MLIR: return ptr %{}", result_value))
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
            } else {
                Diagnostic::debug()
                    .message("Adding return statement to function: nil")
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
                Diagnostic::debug()
                    .message("MLIR: return ptr %nil_value")
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
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
                    Diagnostic::debug()
                        .message("Created MLIR return operation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(return_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using minimal function body without explicit return")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
            
            region.append_block(block);
            region
        },
        &[],
        location,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir_to_mlir::types::MlirOperation;
    use melior::{
        Context,
        dialect::DialectRegistry,
        utility::register_all_dialects,
        ir::{BlockLike, Location, operation::OperationLike, RegionLike},
    };
    use tribute_ast::{Identifier, TributeDatabaseImpl};

    fn setup_test_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        
        context
    }

    #[test]
    fn test_generate_function_body_with_operations() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let block = Block::new(&[]);
        
        let operations = vec![
            MlirOperation::BoxNumber { value: 42 },
            MlirOperation::BoxString { value: "test".to_string() },
            MlirOperation::Return { value: Some("result".to_string()) },
        ];
        
        TributeDatabaseImpl::default().attach(|db| {
            let result = generate_function_body(db, &operations, &context, location, &block);
            
            // Should return the last boxed value name
            assert_eq!(result, Some("result_2".to_string()));
            
            // Block should have operations
            assert!(block.first_operation().is_some());
        });
    }

    #[test]
    fn test_generate_function_body_empty() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let block = Block::new(&[]);
        
        let operations = vec![];
        
        TributeDatabaseImpl::default().attach(|db| {
            let result = generate_function_body(db, &operations, &context, location, &block);
            
            // Should return None for empty body
            assert_eq!(result, None);
        });
    }

    #[test]
    fn test_generate_mlir_function_op_simple() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        
        let name = "test_function";
        let params: Vec<Identifier> = vec!["x".to_string(), "y".to_string()];
        let operations = vec![
            MlirOperation::BoxNumber { value: 10 },
            MlirOperation::Return { value: Some("result".to_string()) },
        ];
        
        TributeDatabaseImpl::default().attach(|db| {
            let func_op = generate_mlir_function_op(db, name, &params, &operations, &context, location);
            
            // Verify function has a region with a block
            let region = func_op.region(0);
            assert!(region.is_ok(), "Function should have a region");
            
            let region = region.unwrap();
            assert!(region.first_block().is_some(), "Region should have a block");
        });
    }

    #[test]
    fn test_generate_mlir_function_op_no_params() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        
        let name = "main";
        let params: Vec<Identifier> = vec![];
        let operations = vec![
            MlirOperation::BoxString { value: "Hello!".to_string() },
            MlirOperation::Call { 
                func: "print_line".to_string(), 
                args: vec!["string_arg".to_string()] 
            },
        ];
        
        TributeDatabaseImpl::default().attach(|db| {
            let func_op = generate_mlir_function_op(db, name, &params, &operations, &context, location);
            
            // Verify the function operation was created successfully
            let num_regions = func_op.regions().count();
            assert_eq!(num_regions, 1, "Function should have exactly one region");
        });
    }
}