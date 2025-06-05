//! MLIR operation generation functions for individual operations.

use crate::hir_to_mlir::types::{MlirListOperation, MlirOperation, expected_type_name};
use melior::{
    ir::{Block, BlockLike, Location, operation::OperationLike},
    Context,
};

/// Generate MLIR operation for boxing a number
pub fn generate_box_number_op<'a>(
    value: i64,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    println!("    Creating boxed number: {}", value);
    
    // Try to generate actual MLIR operations
    use melior::{
        dialect::{arith, func},
        ir::{
            attribute::IntegerAttribute,
            r#type::IntegerType,
        },
    };
    
    // Create i64 constant for the number value
    let i64_type = IntegerType::new(context, 64);
    
    // Try to create the constant - this will test if dialect is properly registered
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let number_constant = arith::constant(
            context,
            IntegerAttribute::new(i64_type.into(), value).into(),
            location,
        );
        
        // Try to create function call in same catch block to avoid partial state
        let call_op = func::call(
            context,
            melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_box_number"),
            &[number_constant.result(0).unwrap().into()],
            &[i64_type.into()],
            location,
        );
        
        (number_constant, call_op)
    })) {
        Ok((number_constant, call_op)) => {
            println!("      SUCCESS: Created MLIR constant for {}", value);
            block.append_operation(number_constant);
            block.append_operation(call_op);
            println!("      SUCCESS: Created MLIR function call");
        }
        Err(_) => {
            println!("      FALLBACK: Using text representation due to MLIR issue");
            let boxed_name = format!("boxed_num_{}", index);
            println!("      -> MLIR: %{} = call @tribute_box_number(i64 {})", boxed_name, value);
        }
    }
}

/// Generate MLIR operation for boxing a string
pub fn generate_box_string_op<'a>(
    value: &str,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    println!("    Creating boxed string: \"{}\"", value);
    
    // Try to generate actual MLIR operations for string boxing
    use melior::{
        dialect::func,
        ir::{
            attribute::IntegerAttribute,
            r#type::IntegerType,
        },
    };
    
    // Create string length constant
    let i64_type = IntegerType::new(context, 64);
    let ptr_type = IntegerType::new(context, 64); // Simplified pointer type
    
    // Try to create the string boxing operations
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Create length constant
        let length_constant = melior::dialect::arith::constant(
            context,
            IntegerAttribute::new(i64_type.into(), value.len() as i64).into(),
            location,
        );
        
        // Create call to tribute_box_string(data, length)
        // For now, we'll create a simple call - in a full implementation,
        // we'd need to create a global string constant first
        let call_op = func::call(
            context,
            melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_box_string"),
            &[length_constant.result(0).unwrap().into()], // Simplified: just pass length
            &[ptr_type.into()],
            location,
        );
        
        (length_constant, call_op)
    })) {
        Ok((length_constant, call_op)) => {
            println!("      SUCCESS: Created MLIR string length constant: {}", value.len());
            block.append_operation(length_constant);
            block.append_operation(call_op);
            println!("      SUCCESS: Created MLIR string boxing call");
        }
        Err(_) => {
            println!("      FALLBACK: Using text representation for string boxing");
            let boxed_name = format!("boxed_str_{}", index);
            println!("      -> MLIR: %{} = call @tribute_box_string(ptr @str_literal_{}, i64 {})", 
                     boxed_name, value.len(), value.len());
        }
    }
}

/// Generate MLIR operations for other operation types
pub fn generate_other_mlir_operation<'a>(
    op: &MlirOperation,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    _block: &Block<'a>,
) {
    match op {
        MlirOperation::Unbox { boxed_value, expected_type } => {
            println!("    Unboxing {} as {:?}", boxed_value, expected_type);
            let unboxed_name = format!("unboxed_{}_{}", expected_type_name(expected_type), index);
            println!("      -> MLIR: %{} = call @tribute_unbox_{}(ptr %{})", 
                     unboxed_name, expected_type_name(expected_type), boxed_value);
        }
        MlirOperation::GcRetain { boxed_value } => {
            println!("    GC Retain: {}", boxed_value);
            println!("      -> MLIR: call @tribute_retain(ptr %{})", boxed_value);
        }
        MlirOperation::GcRelease { boxed_value } => {
            println!("    GC Release: {}", boxed_value);
            println!("      -> MLIR: call @tribute_release(ptr %{})", boxed_value);
        }
        MlirOperation::Call { func, args } => {
            generate_function_call_op(func, args, index, context, location, _block);
        }
        MlirOperation::ListOp { operation } => {
            generate_list_operation_op(operation, index, context, location, _block);
        }
        _ => {
            println!("    Operation: {:?} (not yet implemented)", op);
        }
    }
}

/// Generate MLIR operation for function calls
pub fn generate_function_call_op<'a>(
    func_name: &str,
    args: &[String],
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    println!("    Creating function call: {} with {} arguments", func_name, args.len());
    
    if func_name.starts_with("builtin_") {
        generate_builtin_function_call(func_name, args, index, context, location, block);
    } else {
        // Generate actual func.call for user-defined functions
        println!("      User function call: {}", func_name);
        
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use melior::{
                dialect::func,
                ir::r#type::IntegerType,
            };
            
            let ptr_type = IntegerType::new(context, 64);
            
            // Create call to user function
            let call_op = func::call(
                context,
                melior::ir::attribute::FlatSymbolRefAttribute::new(context, func_name),
                &[], // We'd need actual argument values here
                &[ptr_type.into()], // Return type
                location,
            );
            
            call_op
        })) {
            Ok(call_op) => {
                println!("      SUCCESS: Created MLIR user function call: {}", func_name);
                block.append_operation(call_op);
            }
            Err(_) => {
                println!("      FALLBACK: Using text representation for user function call");
                println!("        -> MLIR: %result_{} = call @{}({})", 
                         index, func_name, 
                         args.iter().map(|arg| format!("ptr %{}", arg)).collect::<Vec<_>>().join(", "));
            }
        }
    }
}

/// Generate MLIR operations for builtin function calls
pub fn generate_builtin_function_call(
    func_name: &str, 
    args: &[String], 
    index: usize,
    context: &Context,
    location: Location<'_>,
    block: &Block<'_>,
) {
    let builtin_name = &func_name[8..]; // Remove "builtin_" prefix
    println!("      Builtin function: {}", builtin_name);
    
    // Helper function to create runtime function calls
    let create_runtime_call = |runtime_func: &str| -> std::result::Result<melior::ir::Operation<'_>, Box<dyn std::error::Error>> {
        use melior::{
            dialect::func,
            ir::r#type::IntegerType,
        };
        
        let ptr_type = IntegerType::new(context, 64);
        
        // For most runtime functions, we expect 2 pointer arguments and return 1 pointer
        let call_op = func::call(
            context,
            melior::ir::attribute::FlatSymbolRefAttribute::new(context, runtime_func),
            &[], // We'd need actual argument values here
            &[ptr_type.into()],
            location,
        );
        
        Ok(call_op)
    };
    
    match builtin_name {
        "+" => {
            println!("      Generating boxed arithmetic addition");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_add_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    println!("      SUCCESS: Created MLIR addition call");
                    block.append_operation(call_op);
                }
                _ => {
                    println!("      FALLBACK: Using text representation for addition");
                    let result_name = format!("add_result_{}", index);
                    println!("        -> MLIR: %{} = call @tribute_add_boxed(ptr %{}, ptr %{})", 
                             result_name, 
                             args.first().unwrap_or(&"arg0".to_string()),
                             args.get(1).unwrap_or(&"arg1".to_string()));
                }
            }
        }
        "-" => {
            println!("      Generating boxed arithmetic subtraction");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_sub_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    println!("      SUCCESS: Created MLIR subtraction call");
                    block.append_operation(call_op);
                }
                _ => {
                    println!("      FALLBACK: Using text representation for subtraction");
                    let result_name = format!("sub_result_{}", index);
                    println!("        -> MLIR: %{} = call @tribute_sub_boxed(ptr %{}, ptr %{})", 
                             result_name, 
                             args.first().unwrap_or(&"arg0".to_string()),
                             args.get(1).unwrap_or(&"arg1".to_string()));
                }
            }
        }
        "*" => {
            println!("      Generating boxed arithmetic multiplication");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_mul_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    println!("      SUCCESS: Created MLIR multiplication call");
                    block.append_operation(call_op);
                }
                _ => {
                    println!("      FALLBACK: Using text representation for multiplication");
                    let result_name = format!("mul_result_{}", index);
                    println!("        -> MLIR: %{} = call @tribute_mul_boxed(ptr %{}, ptr %{})", 
                             result_name, 
                             args.first().unwrap_or(&"arg0".to_string()),
                             args.get(1).unwrap_or(&"arg1".to_string()));
                }
            }
        }
        "/" => {
            println!("      Generating boxed arithmetic division");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_div_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    println!("      SUCCESS: Created MLIR division call");
                    block.append_operation(call_op);
                }
                _ => {
                    println!("      FALLBACK: Using text representation for division");
                    let result_name = format!("div_result_{}", index);
                    println!("        -> MLIR: %{} = call @tribute_div_boxed(ptr %{}, ptr %{})", 
                             result_name, 
                             args.first().unwrap_or(&"arg0".to_string()),
                             args.get(1).unwrap_or(&"arg1".to_string()));
                }
            }
        }
        "%" => {
            println!("      Generating boxed arithmetic modulo");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_mod_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    println!("      SUCCESS: Created MLIR modulo call");
                    block.append_operation(call_op);
                }
                _ => {
                    println!("      FALLBACK: Using text representation for modulo");
                    let result_name = format!("mod_result_{}", index);
                    println!("        -> MLIR: %{} = call @tribute_mod_boxed(ptr %{}, ptr %{})", 
                             result_name, 
                             args.first().unwrap_or(&"arg0".to_string()),
                             args.get(1).unwrap_or(&"arg1".to_string()));
                }
            }
        }
        "=" | "==" => {
            println!("      Generating boxed equality comparison");
            let result_name = format!("eq_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_eq_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.first().unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "<" => {
            println!("      Generating boxed less than comparison");
            let result_name = format!("lt_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_lt_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.first().unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        ">" => {
            println!("      Generating boxed greater than comparison");
            let result_name = format!("gt_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_gt_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.first().unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "<=" => {
            println!("      Generating boxed less than or equal comparison");
            let result_name = format!("le_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_le_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.first().unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        ">=" => {
            println!("      Generating boxed greater than or equal comparison");
            let result_name = format!("ge_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_ge_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.first().unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "print_line" => {
            println!("      Would generate call to print_line builtin");
        }
        "input_line" => {
            println!("      Would generate call to input_line builtin");
        }
        _ => {
            println!("      Unknown builtin: {}", builtin_name);
        }
    }
}

/// Generate MLIR operations for list operations
pub fn generate_list_operation_op<'a>(
    operation: &MlirListOperation,
    index: usize,
    _context: &'a Context,
    _location: Location<'a>,
    _block: &Block<'a>,
) {
    match operation {
        MlirListOperation::CreateEmpty { capacity } => {
            println!("    Creating empty list with capacity: {}", capacity);
            let list_name = format!("empty_list_{}", index);
            println!("      -> MLIR: %{} = call @tribute_box_list_empty(i64 {})", list_name, capacity);
        }
        MlirListOperation::CreateFromArray { elements } => {
            println!("    Creating list from {} elements", elements.len());
            let list_name = format!("array_list_{}", index);
            println!("      -> MLIR: %{} = call @tribute_box_list_from_array(ptr %elements, i64 {})", 
                     list_name, elements.len());
        }
        MlirListOperation::Get { list, index: list_index } => {
            println!("    List get: {}[{}]", list, list_index);
            let result_name = format!("list_get_{}", index);
            println!("      -> MLIR: %{} = call @tribute_list_get(ptr %{}, i64 %{})", 
                     result_name, list, list_index);
            println!("      -> O(1) random access");
        }
        MlirListOperation::Set { list, index: list_index, value } => {
            println!("    List set: {}[{}] = {}", list, list_index, value);
            println!("      -> MLIR: call @tribute_list_set(ptr %{}, i64 %{}, ptr %{})", 
                     list, list_index, value);
            println!("      -> O(1) random access modification");
        }
        MlirListOperation::Push { list, value } => {
            println!("    List push: {}.push({})", list, value);
            println!("      -> MLIR: call @tribute_list_push(ptr %{}, ptr %{})", list, value);
            println!("      -> Amortized O(1) append with automatic resize");
        }
        MlirListOperation::Pop { list } => {
            println!("    List pop: {}.pop()", list);
            let result_name = format!("list_pop_{}", index);
            println!("      -> MLIR: %{} = call @tribute_list_pop(ptr %{})", result_name, list);
            println!("      -> O(1) removal from end");
        }
        MlirListOperation::Length { list } => {
            println!("    List length: {}.length", list);
            let result_name = format!("list_len_{}", index);
            println!("      -> MLIR: %{} = call @tribute_list_length(ptr %{})", result_name, list);
            println!("      -> O(1) length access");
        }
    }
}