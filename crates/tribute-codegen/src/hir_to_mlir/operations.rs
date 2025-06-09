//! MLIR operation generation functions for individual operations.

use crate::hir_to_mlir::types::{expected_type_name, MlirListOperation, MlirOperation};
use melior::{
    ir::{operation::OperationLike, Block, BlockLike, Location},
    Context,
};
use tribute_ast::{CompilationPhase, Db, Diagnostic};

/// Generate MLIR operation for boxing a number
pub fn generate_box_number_op<'a>(
    db: &dyn Db,
    value: i64,
    _index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    Diagnostic::debug()
        .message(format!("Creating boxed number: {}", value))
        .phase(CompilationPhase::HirLowering)
        .accumulate(db);

    // Try to generate actual MLIR operations
    use melior::{
        dialect::{arith, func},
        ir::{attribute::IntegerAttribute, r#type::IntegerType},
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
            Diagnostic::debug()
                .message(format!("Created MLIR constant for {}", value))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            block.append_operation(number_constant);
            block.append_operation(call_op);
            Diagnostic::debug()
                .message("Created MLIR function call")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        Err(_) => {
            Diagnostic::error()
                .message(format!("Failed to create MLIR constant for {}", value))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
    }
}

/// Generate MLIR operation for boxing a string
pub fn generate_box_string_op<'a>(
    db: &dyn Db,
    value: &str,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    Diagnostic::debug()
        .message(format!("Creating boxed string: \"{}\"", value))
        .phase(CompilationPhase::HirLowering)
        .accumulate(db);

    // Try to generate actual MLIR operations for string boxing
    use melior::{
        dialect::func,
        ir::{attribute::IntegerAttribute, r#type::IntegerType},
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
            Diagnostic::debug()
                .message(format!(
                    "Created MLIR string length constant: {}",
                    value.len()
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            block.append_operation(length_constant);
            block.append_operation(call_op);
            Diagnostic::debug()
                .message("Created MLIR string boxing call")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        Err(_) => {
            Diagnostic::warning()
                .message("Using text representation for string boxing")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let boxed_name = format!("boxed_str_{}", index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_box_string(ptr @str_literal_{}, i64 {})",
                    boxed_name,
                    value.len(),
                    value.len()
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
    }
}

/// Generate MLIR operations for other operation types
pub fn generate_other_mlir_operation<'a>(
    db: &dyn Db,
    op: &MlirOperation,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    _block: &Block<'a>,
) {
    match op {
        MlirOperation::Unbox {
            boxed_value,
            expected_type,
        } => {
            Diagnostic::debug()
                .message(format!("Unboxing {} as {:?}", boxed_value, expected_type))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let unboxed_name = format!("unboxed_{}_{}", expected_type_name(expected_type), index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_unbox_{}(ptr %{})",
                    unboxed_name,
                    expected_type_name(expected_type),
                    boxed_value
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        MlirOperation::GcRetain { boxed_value } => {
            Diagnostic::debug()
                .message(format!("GC Retain: {}", boxed_value))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual GC retain call
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_retain
                let retain_call = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_retain"),
                    &[], // TODO: Pass actual SSA value when tracking is implemented
                    &[ptr_type.into()], // Returns pointer
                    location,
                );

                retain_call
            })) {
                Ok(retain_call) => {
                    Diagnostic::debug()
                        .message("Created MLIR GC retain call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    _block.append_operation(retain_call);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for GC retain")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    Diagnostic::debug()
                        .message(format!("MLIR: call @tribute_retain(ptr %{})", boxed_value))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        MlirOperation::GcRelease { boxed_value } => {
            Diagnostic::debug()
                .message(format!("GC Release: {}", boxed_value))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual GC release call
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let _ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_release (void return)
                let release_call = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_release"),
                    &[], // TODO: Pass actual SSA value when tracking is implemented
                    &[], // No return value (void)
                    location,
                );

                release_call
            })) {
                Ok(release_call) => {
                    Diagnostic::debug()
                        .message("Created MLIR GC release call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    _block.append_operation(release_call);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for GC release")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    Diagnostic::debug()
                        .message(format!("MLIR: call @tribute_release(ptr %{})", boxed_value))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        MlirOperation::Call { func, args } => {
            generate_function_call_op(db, func, args, index, context, location, _block);
        }
        MlirOperation::ListOp { operation } => {
            generate_list_operation_op(db, operation, index, context, location, _block);
        }
        _ => {
            Diagnostic::debug()
                .message(format!("Operation: {:?} (not yet implemented)", op))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
    }
}

/// Generate MLIR operation for function calls
pub fn generate_function_call_op<'a>(
    db: &dyn Db,
    func_name: &str,
    args: &[String],
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    Diagnostic::debug()
        .message(format!(
            "Creating function call: {} with {} arguments",
            func_name,
            args.len()
        ))
        .phase(CompilationPhase::HirLowering)
        .accumulate(db);

    if func_name.starts_with("builtin_") {
        generate_builtin_function_call(db, func_name, args, index, context, location, block);
    } else {
        // Generate actual func.call for user-defined functions
        Diagnostic::debug()
            .message(format!("User function call: {}", func_name))
            .phase(CompilationPhase::HirLowering)
            .accumulate(db);

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use melior::{dialect::func, ir::r#type::IntegerType};

            let ptr_type = IntegerType::new(context, 64);

            // Create call to user function
            let call_op = func::call(
                context,
                melior::ir::attribute::FlatSymbolRefAttribute::new(context, func_name),
                &[],                // We'd need actual argument values here
                &[ptr_type.into()], // Return type
                location,
            );

            call_op
        })) {
            Ok(call_op) => {
                Diagnostic::debug()
                    .message(format!("Created MLIR user function call: {}", func_name))
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
                block.append_operation(call_op);
            }
            Err(_) => {
                Diagnostic::warning()
                    .message("Using text representation for user function call")
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
                Diagnostic::debug()
                    .message(format!(
                        "MLIR: %result_{} = call @{}({})",
                        index,
                        func_name,
                        args.iter()
                            .map(|arg| format!("ptr %{}", arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .phase(CompilationPhase::HirLowering)
                    .accumulate(db);
            }
        }
    }
}

/// Generate MLIR operations for builtin function calls
pub fn generate_builtin_function_call(
    db: &dyn Db,
    func_name: &str,
    args: &[String],
    index: usize,
    context: &Context,
    location: Location<'_>,
    block: &Block<'_>,
) {
    let builtin_name = &func_name[8..]; // Remove "builtin_" prefix
    Diagnostic::debug()
        .message(format!("Builtin function: {}", builtin_name))
        .phase(CompilationPhase::HirLowering)
        .accumulate(db);

    // Helper function to create runtime function calls
    let create_runtime_call = |runtime_func: &str| -> std::result::Result<
        melior::ir::Operation<'_>,
        Box<dyn std::error::Error>,
    > {
        use melior::{dialect::func, ir::r#type::IntegerType};

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
            Diagnostic::debug()
                .message("Generating boxed arithmetic addition")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_add_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    Diagnostic::debug()
                        .message("Created MLIR addition call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                _ => {
                    Diagnostic::warning()
                        .message("Using text representation for addition")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("add_result_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_add_boxed(ptr %{}, ptr %{})",
                            result_name,
                            args.first().unwrap_or(&"arg0".to_string()),
                            args.get(1).unwrap_or(&"arg1".to_string())
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        "-" => {
            Diagnostic::debug()
                .message("Generating boxed arithmetic subtraction")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_sub_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    Diagnostic::debug()
                        .message("Created MLIR subtraction call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                _ => {
                    Diagnostic::warning()
                        .message("Using text representation for subtraction")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("sub_result_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_sub_boxed(ptr %{}, ptr %{})",
                            result_name,
                            args.first().unwrap_or(&"arg0".to_string()),
                            args.get(1).unwrap_or(&"arg1".to_string())
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        "*" => {
            Diagnostic::debug()
                .message("Generating boxed arithmetic multiplication")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_mul_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    Diagnostic::debug()
                        .message("Created MLIR multiplication call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                _ => {
                    Diagnostic::warning()
                        .message("Using text representation for multiplication")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("mul_result_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_mul_boxed(ptr %{}, ptr %{})",
                            result_name,
                            args.first().unwrap_or(&"arg0".to_string()),
                            args.get(1).unwrap_or(&"arg1".to_string())
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        "/" => {
            Diagnostic::debug()
                .message("Generating boxed arithmetic division")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_div_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    Diagnostic::debug()
                        .message("Created MLIR division call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                _ => {
                    Diagnostic::warning()
                        .message("Using text representation for division")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("div_result_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_div_boxed(ptr %{}, ptr %{})",
                            result_name,
                            args.first().unwrap_or(&"arg0".to_string()),
                            args.get(1).unwrap_or(&"arg1".to_string())
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        "%" => {
            Diagnostic::debug()
                .message("Generating boxed arithmetic modulo")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_runtime_call("tribute_mod_boxed")
            })) {
                Ok(Ok(call_op)) => {
                    Diagnostic::debug()
                        .message("Created MLIR modulo call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                _ => {
                    Diagnostic::warning()
                        .message("Using text representation for modulo")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("mod_result_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_mod_boxed(ptr %{}, ptr %{})",
                            result_name,
                            args.first().unwrap_or(&"arg0".to_string()),
                            args.get(1).unwrap_or(&"arg1".to_string())
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        "=" | "==" => {
            Diagnostic::debug()
                .message("Generating boxed equality comparison")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let result_name = format!("eq_result_{}", index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_eq_boxed(ptr %{}, ptr %{})",
                    result_name,
                    args.first().unwrap_or(&"arg0".to_string()),
                    args.get(1).unwrap_or(&"arg1".to_string())
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        "<" => {
            Diagnostic::debug()
                .message("Generating boxed less than comparison")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let result_name = format!("lt_result_{}", index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_lt_boxed(ptr %{}, ptr %{})",
                    result_name,
                    args.first().unwrap_or(&"arg0".to_string()),
                    args.get(1).unwrap_or(&"arg1".to_string())
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        ">" => {
            Diagnostic::debug()
                .message("Generating boxed greater than comparison")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let result_name = format!("gt_result_{}", index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_gt_boxed(ptr %{}, ptr %{})",
                    result_name,
                    args.first().unwrap_or(&"arg0".to_string()),
                    args.get(1).unwrap_or(&"arg1".to_string())
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        "<=" => {
            Diagnostic::debug()
                .message("Generating boxed less than or equal comparison")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let result_name = format!("le_result_{}", index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_le_boxed(ptr %{}, ptr %{})",
                    result_name,
                    args.first().unwrap_or(&"arg0".to_string()),
                    args.get(1).unwrap_or(&"arg1".to_string())
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        ">=" => {
            Diagnostic::debug()
                .message("Generating boxed greater than or equal comparison")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            let result_name = format!("ge_result_{}", index);
            Diagnostic::debug()
                .message(format!(
                    "MLIR: %{} = call @tribute_ge_boxed(ptr %{}, ptr %{})",
                    result_name,
                    args.first().unwrap_or(&"arg0".to_string()),
                    args.get(1).unwrap_or(&"arg1".to_string())
                ))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        "print_line" => {
            Diagnostic::debug()
                .message("Generating print_line builtin")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let _ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_print_line(ptr) -> void
                let print_call = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(
                        context,
                        "tribute_print_line",
                    ),
                    &[], // TODO: Pass actual argument when SSA tracking is implemented
                    &[], // void return
                    location,
                );

                print_call
            })) {
                Ok(print_call) => {
                    Diagnostic::debug()
                        .message("Created MLIR print_line call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(print_call);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for print_line")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: call @tribute_print_line(ptr %{})",
                            args.first().unwrap_or(&"arg0".to_string())
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        "input_line" => {
            Diagnostic::debug()
                .message("Generating input_line builtin")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_input_line() -> ptr
                let input_call = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(
                        context,
                        "tribute_input_line",
                    ),
                    &[],                // No arguments
                    &[ptr_type.into()], // Returns pointer to string
                    location,
                );

                input_call
            })) {
                Ok(input_call) => {
                    Diagnostic::debug()
                        .message("Created MLIR input_line call")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(input_call);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for input_line")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("input_result_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_input_line()",
                            result_name
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        _ => {
            Diagnostic::warning()
                .message(format!("Unknown builtin: {}", builtin_name))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
    }
}

/// Generate MLIR operations for list operations
pub fn generate_list_operation_op<'a>(
    db: &dyn Db,
    operation: &MlirListOperation,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    block: &Block<'a>,
) {
    match operation {
        MlirListOperation::CreateEmpty { capacity } => {
            Diagnostic::debug()
                .message(format!("Creating empty list with capacity: {}", capacity))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{
                    dialect::{arith, func},
                    ir::{attribute::IntegerAttribute, r#type::IntegerType},
                };

                let i64_type = IntegerType::new(context, 64);
                let ptr_type = IntegerType::new(context, 64); // Simplified pointer type

                // Create capacity constant
                let capacity_constant = arith::constant(
                    context,
                    IntegerAttribute::new(i64_type.into(), *capacity as i64).into(),
                    location,
                );

                // Create call to tribute_box_list_empty
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(
                        context,
                        "tribute_box_list_empty",
                    ),
                    &[capacity_constant.result(0).unwrap().into()],
                    &[ptr_type.into()],
                    location,
                );

                (capacity_constant, call_op)
            })) {
                Ok((capacity_constant, call_op)) => {
                    Diagnostic::debug()
                        .message(format!(
                            "Created MLIR list creation with capacity {}",
                            capacity
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(capacity_constant);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for list creation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let list_name = format!("empty_list_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_box_list_empty(i64 {})",
                            list_name, capacity
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        MlirListOperation::CreateFromArray { elements } => {
            Diagnostic::debug()
                .message(format!("Creating list from {} elements", elements.len()))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{
                    dialect::{arith, func},
                    ir::{attribute::IntegerAttribute, r#type::IntegerType},
                };

                let i64_type = IntegerType::new(context, 64);
                let ptr_type = IntegerType::new(context, 64);

                // Create length constant
                let length_constant = arith::constant(
                    context,
                    IntegerAttribute::new(i64_type.into(), elements.len() as i64).into(),
                    location,
                );

                // Create call to tribute_box_list_from_array
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(
                        context,
                        "tribute_box_list_from_array",
                    ),
                    &[length_constant.result(0).unwrap().into()], // Simplified: just pass length for now
                    &[ptr_type.into()],
                    location,
                );

                (length_constant, call_op)
            })) {
                Ok((length_constant, call_op)) => {
                    Diagnostic::debug()
                        .message(format!(
                            "Created MLIR list from array with {} elements",
                            elements.len()
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(length_constant);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for array list creation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let list_name = format!("array_list_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_box_list_from_array(ptr %elements, i64 {})",
                            list_name,
                            elements.len()
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
        }
        MlirListOperation::Get {
            list,
            index: list_index,
        } => {
            Diagnostic::debug()
                .message(format!("List get: {}[{}]", list, list_index))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_list_get
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_list_get"),
                    &[], // TODO: Pass actual list and index SSA values
                    &[ptr_type.into()],
                    location,
                );

                call_op
            })) {
                Ok(call_op) => {
                    Diagnostic::debug()
                        .message("Created MLIR list get operation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for list get")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("list_get_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_list_get(ptr %{}, i64 %{})",
                            result_name, list, list_index
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
            Diagnostic::debug()
                .message("O(1) random access")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        MlirListOperation::Set {
            list,
            index: list_index,
            value,
        } => {
            Diagnostic::debug()
                .message(format!("List set: {}[{}] = {}", list, list_index, value))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let _ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_list_set (void return)
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_list_set"),
                    &[], // TODO: Pass actual list, index, and value SSA values
                    &[], // void return
                    location,
                );

                call_op
            })) {
                Ok(call_op) => {
                    Diagnostic::debug()
                        .message("Created MLIR list set operation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for list set")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: call @tribute_list_set(ptr %{}, i64 %{}, ptr %{})",
                            list, list_index, value
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
            Diagnostic::debug()
                .message("O(1) random access modification")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        MlirListOperation::Push { list, value } => {
            Diagnostic::debug()
                .message(format!("List push: {}.push({})", list, value))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let _ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_list_push (void return)
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(
                        context,
                        "tribute_list_push",
                    ),
                    &[], // TODO: Pass actual list and value SSA values
                    &[], // void return
                    location,
                );

                call_op
            })) {
                Ok(call_op) => {
                    Diagnostic::debug()
                        .message("Created MLIR list push operation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for list push")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: call @tribute_list_push(ptr %{}, ptr %{})",
                            list, value
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
            Diagnostic::debug()
                .message("Amortized O(1) append with automatic resize")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        MlirListOperation::Pop { list } => {
            Diagnostic::debug()
                .message(format!("List pop: {}.pop()", list))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let ptr_type = IntegerType::new(context, 64);

                // Create call to tribute_list_pop
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(context, "tribute_list_pop"),
                    &[],                // TODO: Pass actual list SSA value
                    &[ptr_type.into()], // Returns popped element
                    location,
                );

                call_op
            })) {
                Ok(call_op) => {
                    Diagnostic::debug()
                        .message("Created MLIR list pop operation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for list pop")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("list_pop_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_list_pop(ptr %{})",
                            result_name, list
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
            Diagnostic::debug()
                .message("O(1) removal from end")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
        MlirListOperation::Length { list } => {
            Diagnostic::debug()
                .message(format!("List length: {}.length", list))
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);

            // Try to generate actual MLIR operation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use melior::{dialect::func, ir::r#type::IntegerType};

                let i64_type = IntegerType::new(context, 64);

                // Create call to tribute_list_length
                let call_op = func::call(
                    context,
                    melior::ir::attribute::FlatSymbolRefAttribute::new(
                        context,
                        "tribute_list_length",
                    ),
                    &[],                // TODO: Pass actual list SSA value
                    &[i64_type.into()], // Returns length as i64
                    location,
                );

                call_op
            })) {
                Ok(call_op) => {
                    Diagnostic::debug()
                        .message("Created MLIR list length operation")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    block.append_operation(call_op);
                }
                Err(_) => {
                    Diagnostic::warning()
                        .message("Using text representation for list length")
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                    let result_name = format!("list_len_{}", index);
                    Diagnostic::debug()
                        .message(format!(
                            "MLIR: %{} = call @tribute_list_length(ptr %{})",
                            result_name, list
                        ))
                        .phase(CompilationPhase::HirLowering)
                        .accumulate(db);
                }
            }
            Diagnostic::debug()
                .message("O(1) length access")
                .phase(CompilationPhase::HirLowering)
                .accumulate(db);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir_to_mlir::BoxedType;
    use melior::{
        dialect::DialectRegistry,
        ir::{Block, BlockLike, Location, Module, Region},
        utility::register_all_dialects,
        Context,
    };
    use tribute_ast::TributeDatabaseImpl;

    fn setup_test_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        context
    }

    #[test]
    fn test_generate_box_number_op() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let _module = Module::new(location);
        let _region = Region::new();
        let block = Block::new(&[]);

        TributeDatabaseImpl::default().attach(|db| {
            // Test boxing a number
            generate_box_number_op(db, 42, 0, &context, location, &block);
        });

        // Verify the operations were added correctly
        let first_op = block
            .first_operation()
            .expect("Block should have operations");
        let second_op = first_op
            .next_in_block()
            .expect("Block should have a second operation for function call");

        // Check that we have constant and function call operations
        assert!(first_op.to_string().contains("arith.constant"));
        assert!(second_op
            .to_string()
            .contains("func.call @tribute_box_number"));

        assert!(
            second_op.next_in_block().is_none(),
            "Second operation should be the last one"
        );
    }

    #[test]
    fn test_generate_box_string_op() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let _module = Module::new(location);
        let _region = Region::new();
        let block = Block::new(&[]);

        TributeDatabaseImpl::default().attach(|db| {
            // Test boxing a string
            generate_box_string_op(db, "Hello, test!", 0, &context, location, &block);
        });

        // Verify the operations were added correctly
        let first_op = block
            .first_operation()
            .expect("Block should have operations");
        let second_op = first_op
            .next_in_block()
            .expect("Block should have a second operation for function call");

        // Check that we have length constant and function call operations
        assert!(first_op.to_string().contains("arith.constant"));
        assert!(second_op
            .to_string()
            .contains("func.call @tribute_box_string"));

        assert!(
            second_op.next_in_block().is_none(),
            "Second operation should be the last one"
        );
    }

    #[test]
    fn test_generate_gc_operations() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let _module = Module::new(location);
        let _region = Region::new();
        let block = Block::new(&[]);

        TributeDatabaseImpl::default().attach(|db| {
            // Test GC retain
            let retain_op = MlirOperation::GcRetain {
                boxed_value: "test_value".to_string(),
            };
            generate_other_mlir_operation(db, &retain_op, 0, &context, location, &block);

            // Test GC release
            let release_op = MlirOperation::GcRelease {
                boxed_value: "test_value".to_string(),
            };
            generate_other_mlir_operation(db, &release_op, 1, &context, location, &block);
        });

        // Verify both operations were added by checking that we have operations
        // and that there's more than one
        let first_op = block
            .first_operation()
            .expect("Block should have operations");
        let second_op = first_op
            .next_in_block()
            .expect("Block should have a second operation");
        assert_ne!(
            first_op, second_op,
            "First and second operations should be different"
        );
        assert!(first_op.to_string().contains("func.call @tribute_retain"));
        assert!(second_op.to_string().contains("func.call @tribute_release"));

        assert!(
            second_op.next_in_block().is_none(),
            "Second operation should be the last one"
        );
    }

    #[test]
    fn test_generate_builtin_arithmetic() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let block = Block::new(&[]);

        TributeDatabaseImpl::default().attach(|db| {
            // Test arithmetic operations
            let args = vec!["arg0".to_string(), "arg1".to_string()];

            // Test addition
            generate_builtin_function_call(db, "builtin_+", &args, 0, &context, location, &block);

            // Test subtraction
            generate_builtin_function_call(db, "builtin_-", &args, 1, &context, location, &block);

            // Test multiplication
            generate_builtin_function_call(db, "builtin_*", &args, 2, &context, location, &block);

            // Test division
            generate_builtin_function_call(db, "builtin_/", &args, 3, &context, location, &block);
        });

        // Verify arithmetic operations were generated
        let mut current_op = block.first_operation();
        let mut op_count = 0;
        let expected_functions = [
            "tribute_add_boxed",
            "tribute_sub_boxed",
            "tribute_mul_boxed",
            "tribute_div_boxed",
        ];

        while let Some(op) = current_op {
            assert!(op.to_string().contains("func.call"));
            assert!(op.to_string().contains(expected_functions[op_count]));
            current_op = op.next_in_block();
            op_count += 1;
        }

        assert_eq!(op_count, 4, "Should have exactly 4 arithmetic operations");
    }

    #[test]
    fn test_generate_builtin_io() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let block = Block::new(&[]);

        TributeDatabaseImpl::default().attach(|db| {
            // Test print_line
            let print_args = vec!["string_to_print".to_string()];
            generate_builtin_function_call(
                db,
                "builtin_print_line",
                &print_args,
                0,
                &context,
                location,
                &block,
            );

            // Test input_line
            let no_args = vec![];
            generate_builtin_function_call(
                db,
                "builtin_input_line",
                &no_args,
                1,
                &context,
                location,
                &block,
            );
        });

        // Verify I/O operations were generated correctly
        let first_op = block
            .first_operation()
            .expect("Block should have operations");
        let second_op = first_op
            .next_in_block()
            .expect("Block should have a second operation");

        // Check that we have the correct I/O function calls
        assert!(first_op
            .to_string()
            .contains("func.call @tribute_print_line"));
        assert!(second_op
            .to_string()
            .contains("func.call @tribute_input_line"));

        assert!(
            second_op.next_in_block().is_none(),
            "Second operation should be the last one"
        );
    }

    #[test]
    fn test_generate_list_operations() {
        let context = setup_test_context();
        let location = Location::unknown(&context);
        let block = Block::new(&[]);

        TributeDatabaseImpl::default().attach(|db| {
            // Test list creation with capacity
            let create_op = MlirListOperation::CreateEmpty { capacity: 10 };
            generate_list_operation_op(db, &create_op, 0, &context, location, &block);

            // Verify operations were generated for CreateEmpty
            let first_op = block
                .first_operation()
                .expect("Block should have operations after CreateEmpty");
            let second_op = first_op
                .next_in_block()
                .expect("Block should have a second operation for function call");

            // Check that we have constant and function call operations for list creation
            assert!(first_op.to_string().contains("arith.constant"));
            assert!(second_op
                .to_string()
                .contains("func.call @tribute_box_list_empty"));

            // Test CreateFromArray
            let array_op = MlirListOperation::CreateFromArray {
                elements: vec!["elem1".to_string(), "elem2".to_string()],
            };
            TributeDatabaseImpl::default().attach(|db| {
                generate_list_operation_op(db, &array_op, 1, &context, location, &block);
            });

            // Should now have 4 operations total (2 from CreateEmpty + 2 from CreateFromArray)
            let mut current_op = block.first_operation();
            let mut op_count = 0;
            while let Some(op) = current_op {
                op_count += 1;
                current_op = op.next_in_block();
            }
            assert_eq!(
                op_count, 4,
                "Should have 4 operations after 2 list creations"
            );

            // Test other list operations that generate single operations
            let test_ops = vec![
                MlirListOperation::Push {
                    list: "my_list".to_string(),
                    value: "new_value".to_string(),
                },
                MlirListOperation::Get {
                    list: "my_list".to_string(),
                    index: "0".to_string(),
                },
                MlirListOperation::Set {
                    list: "test_list".to_string(),
                    index: "1".to_string(),
                    value: "new_val".to_string(),
                },
                MlirListOperation::Pop {
                    list: "test_list".to_string(),
                },
                MlirListOperation::Length {
                    list: "test_list".to_string(),
                },
            ];

            // All these should complete without panicking and add operations
            for (i, op) in test_ops.iter().enumerate() {
                generate_list_operation_op(db, op, i + 2, &context, location, &block);
            }
        });

        // Count total operations - should be 4 (from creation ops) + 5 (from other ops) = 9
        let mut current_op = block.first_operation();
        let mut op_count = 0;
        let expected_functions = [
            "tribute_box_list_empty",      // CreateEmpty
            "tribute_box_list_from_array", // CreateFromArray
            "tribute_list_push",           // Push
            "tribute_list_get",            // Get
            "tribute_list_set",            // Set
            "tribute_list_pop",            // Pop
            "tribute_list_length",         // Length
        ];
        let mut func_count = 0;

        while let Some(op) = current_op {
            let op_str = op.to_string();
            if op_str.contains("func.call") {
                // Check that this function call is one we expect
                assert!(
                    expected_functions.iter().any(|&func| op_str.contains(func)),
                    "Operation should contain one of the expected list functions: {}",
                    op_str
                );
                func_count += 1;
            }
            op_count += 1;
            current_op = op.next_in_block();
        }

        assert_eq!(op_count, 9, "Should have 9 total operations");
        assert_eq!(func_count, 7, "Should have 7 function call operations");
    }

    #[test]
    fn test_expected_type_name() {
        assert_eq!(expected_type_name(&BoxedType::Number), "number");
        assert_eq!(expected_type_name(&BoxedType::String), "string");
        assert_eq!(expected_type_name(&BoxedType::Boolean), "boolean");
        assert_eq!(expected_type_name(&BoxedType::Function), "function");
        assert_eq!(expected_type_name(&BoxedType::List), "list");
        assert_eq!(expected_type_name(&BoxedType::Nil), "nil");
    }
}
