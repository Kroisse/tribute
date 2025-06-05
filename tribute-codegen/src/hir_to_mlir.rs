//! HIR to MLIR lowering for Tribute programs.

use crate::error::Result;
use melior::{
    ir::{Block, BlockLike, Location, Module, operation::OperationLike},
    Context,
};
use tribute_hir::{HirProgram, HirFunction, HirExpr, Expr};

/// Salsa query to generate MLIR module from HIR program.
#[salsa::tracked]
pub fn generate_mlir_module<'db>(db: &'db dyn salsa::Database, hir_program: HirProgram<'db>) -> MlirModule<'db> {
    let mut function_results = Vec::new();
    
    // Generate MLIR for all functions in the HIR program
    let functions = hir_program.functions(db);
    
    for (name, hir_function) in functions {
        let mlir_func = generate_mlir_function(db, hir_function);
        function_results.push((name.clone(), mlir_func));
    }
    
    MlirModule::new(db, function_results)
}

/// Tracked MLIR module representation
#[salsa::tracked]
pub struct MlirModule<'db> {
    #[return_ref]
    pub functions: Vec<(tribute_ast::Identifier, MlirFunction<'db>)>,
}

/// Tracked MLIR function representation
#[salsa::tracked]
pub struct MlirFunction<'db> {
    pub name: tribute_ast::Identifier,
    #[return_ref]
    pub params: Vec<tribute_ast::Identifier>,
    #[return_ref]
    pub body: Vec<MlirOperation>,
    pub span: tribute_ast::SimpleSpan,
}

/// MLIR operation representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MlirOperation {
    /// Create a boxed number value
    BoxNumber { value: i64 },
    /// Create a boxed string value  
    BoxString { value: String },
    /// Unbox a value to get its content
    Unbox { boxed_value: String, expected_type: BoxedType },
    /// Function call (func.call) - all functions work with boxed values
    Call { func: String, args: Vec<String> },
    /// Variable reference - always returns a boxed value
    Variable { name: String },
    /// Return operation - always returns a boxed value
    Return { value: Option<String> },
    /// GC operations
    GcRetain { boxed_value: String },
    GcRelease { boxed_value: String },
    GcCollect,
    /// List operations
    ListOp { operation: MlirListOperation },
    /// Placeholder for unimplemented operations
    Placeholder { description: String },
}

/// Types that can be stored in boxed values
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoxedType {
    Number,
    String,
    Boolean,
    Function,
    List,
    Nil,
}

/// List operation representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MlirListOperation {
    /// Create empty list: tribute_box_list_empty(capacity)
    CreateEmpty { capacity: usize },
    /// Create from array: tribute_box_list_from_array(elements, count) 
    CreateFromArray { elements: Vec<String> },
    /// Get element: tribute_list_get(list, index) - O(1)
    Get { list: String, index: String },
    /// Set element: tribute_list_set(list, index, value) - O(1)
    Set { list: String, index: String, value: String },
    /// Push element: tribute_list_push(list, value) - Amortized O(1)
    Push { list: String, value: String },
    /// Pop element: tribute_list_pop(list) - O(1)
    Pop { list: String },
    /// Get length: tribute_list_length(list) - O(1)
    Length { list: String },
}

/// Generate MLIR operations for function body
fn generate_function_body<'a>(
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

/// Generate MLIR operation for boxing a number
fn generate_box_number_op<'a>(
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
fn generate_box_string_op<'a>(
    value: &str,
    index: usize,
    context: &'a Context,
    location: Location<'a>,
    _block: &Block<'a>,
) {
    println!("    Creating boxed string: \"{}\"", value);
    // This would generate:
    // 1. %str_data = call @tribute_alloc_string(i64 {})  // string length
    // 2. call @llvm.memcpy(%str_data, @string_literal_{}, i64 {})
    // 3. %boxed = call @tribute_box_string(%str_data, i64 {})
    let boxed_name = format!("boxed_str_{}", index);
    println!("      -> MLIR: %{} = call @tribute_box_string(ptr @str_literal_{}, i64 {})", 
             boxed_name, value.len(), value.len());
}

/// Generate MLIR operations for other operation types
fn generate_other_mlir_operation<'a>(
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
fn generate_function_call_op<'a>(
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
        println!("        -> MLIR: %result_{} = call @{}({})", 
                 index, func_name, 
                 args.iter().map(|arg| format!("ptr %{}", arg)).collect::<Vec<_>>().join(", "));
        
        // TODO: Generate actual func.call operation when MLIR context is properly set up
        // let call_op = func::call(
        //     context,
        //     melior::ir::attribute::FlatSymbolRefAttribute::new(context, func_name),
        //     &[...], // Convert args to MLIR values
        //     &[ptr_type.into()], // Return type
        //     location,
        // );
        // block.append_operation(call_op);
    }
}

/// Generate MLIR operations for builtin function calls
fn generate_builtin_function_call(
    func_name: &str, 
    args: &[String], 
    index: usize,
    _context: &Context,
    _location: Location<'_>,
    _block: &Block<'_>,
) {
    let builtin_name = &func_name[8..]; // Remove "builtin_" prefix
    println!("      Builtin function: {}", builtin_name);
    
    match builtin_name {
        "+" => {
            println!("      Generating boxed arithmetic addition");
            let result_name = format!("add_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_add_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "-" => {
            println!("      Generating boxed arithmetic subtraction");
            let result_name = format!("sub_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_sub_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "*" => {
            println!("      Generating boxed arithmetic multiplication");
            let result_name = format!("mul_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_mul_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "/" => {
            println!("      Generating boxed arithmetic division");
            let result_name = format!("div_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_div_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "%" => {
            println!("      Generating boxed arithmetic modulo");
            let result_name = format!("mod_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_mod_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "=" | "==" => {
            println!("      Generating boxed equality comparison");
            let result_name = format!("eq_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_eq_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "<" => {
            println!("      Generating boxed less than comparison");
            let result_name = format!("lt_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_lt_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        ">" => {
            println!("      Generating boxed greater than comparison");
            let result_name = format!("gt_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_gt_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        "<=" => {
            println!("      Generating boxed less than or equal comparison");
            let result_name = format!("le_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_le_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
                     args.get(1).unwrap_or(&"arg1".to_string()));
        }
        ">=" => {
            println!("      Generating boxed greater than or equal comparison");
            let result_name = format!("ge_result_{}", index);
            println!("        -> MLIR: %{} = call @tribute_ge_boxed(ptr %{}, ptr %{})", 
                     result_name, 
                     args.get(0).unwrap_or(&"arg0".to_string()),
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
fn generate_list_operation_op<'a>(
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

/// Generate a single MLIR function operation
fn generate_mlir_function_op<'a>(
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
            
            // TODO: Generate actual func.return operation when MLIR context is properly set up
            // let return_op = func::r#return(
            //     context,
            //     &[last_result_value], // Last computed value or nil
            //     location,
            // );
            // block.append_operation(return_op);
            
            region.append_block(block);
            region
        },
        &[],
        location,
    )
}

/// Convenience function to convert MlirModule to actual MLIR Module
/// This bridges between Salsa-tracked data and MLIR API
pub fn mlir_module_to_melior<'a>(
    db: &dyn salsa::Database,
    mlir_module: MlirModule<'_>,
    context: &'a Context,
    location: Location<'a>,
) -> Result<Module<'a>> {
    use melior::ir::{Module, BlockLike};

    let module = Module::new(location);
    let functions = mlir_module.functions(db);
    
    println!("Generating MLIR for {} functions", functions.len());
    
    for (name, mlir_func) in functions.iter() {
        let params = mlir_func.params(db);
        let body_ops = mlir_func.body(db);
        
        // Generate MLIR function operation
        let function_op = generate_mlir_function_op(name, &params, &body_ops, context, location);
        
        // Add function to module
        module.body().append_operation(function_op);
        println!("    Successfully generated MLIR function: {}", name);
    }
    
    println!("Successfully generated MLIR module with {} functions", functions.len());
    Ok(module)
}

/// Salsa query to generate MLIR function from HIR function.
#[salsa::tracked]
pub fn generate_mlir_function<'db>(db: &'db dyn salsa::Database, hir_function: HirFunction<'db>) -> MlirFunction<'db> {
    let name = hir_function.name(db).clone();
    let params = hir_function.params(db).clone();
    let span = hir_function.span(db);
    let hir_body = hir_function.body(db);
    
    // Generate MLIR operations for function body
    let mut mlir_operations = Vec::new();
    
    for hir_expr in hir_body {
        let operations = generate_mlir_expression(db, hir_expr);
        mlir_operations.extend(operations.operations(db).clone());
    }
    
    // Add return operation if function doesn't end with one
    if mlir_operations.is_empty() || !matches!(mlir_operations.last(), Some(MlirOperation::Return { .. })) {
        mlir_operations.push(MlirOperation::Return { value: None });
    }
    
    MlirFunction::new(db, name, params, mlir_operations, span)
}

/// Tracked MLIR expression result
#[salsa::tracked]
pub struct MlirExpressionResult<'db> {
    #[return_ref]
    pub operations: Vec<MlirOperation>,
    pub result_value: Option<String>,
}

/// Salsa query to generate MLIR operations from HIR expression.
#[salsa::tracked]
pub fn generate_mlir_expression<'db>(db: &'db dyn salsa::Database, hir_expr: HirExpr<'db>) -> MlirExpressionResult<'db> {
    let expr = hir_expr.expr(db);
    let _span = hir_expr.span(db);
    
    let mut operations = Vec::new();
    let result_value = match expr {
        Expr::Number(n) => {
            operations.push(MlirOperation::BoxNumber { value: n });
            Some(format!("boxed_num_{}", n))
        }
        Expr::String(s) => {
            operations.push(MlirOperation::BoxString { value: s.clone() });
            Some(format!("boxed_str_{}", s.len()))
        }
        Expr::Variable(var) => {
            operations.push(MlirOperation::Variable { name: var.clone() });
            Some(var.clone())
        }
        Expr::Call { func, args } => {
            // Generate MLIR for function call
            let func_name = match &func.0 {
                Expr::Variable(name) => name.clone(),
                _ => "<complex_func>".to_string(),
            };
            
            let mut arg_names = Vec::new();
            for arg in args {
                let arg_result = generate_mlir_expression(db, HirExpr::new(db, arg.0.clone(), arg.1));
                operations.extend(arg_result.operations(db).clone());
                if let Some(arg_value) = arg_result.result_value(db) {
                    arg_names.push(arg_value.clone());
                }
            }
            
            operations.push(MlirOperation::Call { func: func_name.clone(), args: arg_names });
            Some(format!("call_{}", func_name))
        }
        Expr::Let { var: _, value: _, body: _ } => {
            operations.push(MlirOperation::Placeholder { 
                description: "let binding".to_string() 
            });
            Some("let_result".to_string())
        }
        Expr::Match { expr: _, cases: _ } => {
            operations.push(MlirOperation::Placeholder { 
                description: "pattern matching".to_string() 
            });
            Some("match_result".to_string())
        }
        Expr::Builtin { name, args } => {
            // Handle builtin functions
            let mut arg_names = Vec::new();
            for arg in args {
                let arg_result = generate_mlir_expression(db, HirExpr::new(db, arg.0.clone(), arg.1));
                operations.extend(arg_result.operations(db).clone());
                if let Some(arg_value) = arg_result.result_value(db) {
                    arg_names.push(arg_value.clone());
                }
            }
            
            // For now, treat builtins as function calls
            operations.push(MlirOperation::Call { 
                func: format!("builtin_{}", name),
                args: arg_names
            });
            Some(format!("builtin_{}", name))
        }
        Expr::Block(exprs) => {
            let mut last_result = None;
            for expr in exprs {
                let expr_result = generate_mlir_expression(db, HirExpr::new(db, expr.0.clone(), expr.1));
                operations.extend(expr_result.operations(db).clone());
                last_result = expr_result.result_value(db).clone();
            }
            last_result
        }
    };
    
    MlirExpressionResult::new(db, operations, result_value)
}

/// Helper function to get string representation of BoxedType
fn expected_type_name(boxed_type: &BoxedType) -> &'static str {
    match boxed_type {
        BoxedType::Number => "number",
        BoxedType::String => "string", 
        BoxedType::Boolean => "boolean",
        BoxedType::Function => "function",
        BoxedType::List => "list",
        BoxedType::Nil => "nil",
    }
}

#[allow(dead_code)]
struct MlirContext {
    // Symbol table for tracking variables and functions
    // Type information
    // Scope management
}

// TODO: Implement actual melior MLIR generation
//
// This implementation currently generates comprehensive logging and MLIR textual representation
// but doesn't create actual melior Operations. The next step would be to:
//
// 1. **Function Generation**:
//    - Use func::func() to create actual function operations
//    - Handle function types properly
//    - Create function body blocks
//
// 2. **Operation Generation**:
//    - Use arith::constant() for MlirOperation::Constant
//    - Use func::call() for MlirOperation::Call
//    - Handle variables and SSA values
//
// 3. **Block and Region Management**:
//    - Create proper basic blocks
//    - Handle control flow
//    - Manage SSA value lifetimes
//
// 4. **Error Handling**:
//    - Proper error types for MLIR generation failures
//    - Source location tracking
//    - Diagnostic integration