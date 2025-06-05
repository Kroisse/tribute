//! HIR to MLIR lowering for Tribute programs.

use crate::error::Result;
use melior::{
    ir::{Location, Module},
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

/// Convenience function to convert MlirModule to actual MLIR Module
/// This bridges between Salsa-tracked data and MLIR API
pub fn mlir_module_to_melior<'a>(
    db: &dyn salsa::Database,
    mlir_module: MlirModule<'_>,
    context: &'a Context,
    location: Location<'a>,
) -> Result<Module<'a>> {
    use melior::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType},
            Block, BlockLike, Region, RegionLike,
        },
    };

    let module = Module::new(location);
    let functions = mlir_module.functions(db);
    
    println!("Generating MLIR for {} functions", functions.len());
    
    for (name, mlir_func) in functions.iter() {
        let params = mlir_func.params(db);
        let body_ops = mlir_func.body(db);
        println!("  Function: {} with {} params, {} operations", 
                 name, params.len(), body_ops.len());
        
        // Create function type: all functions work with boxed values
        // For now: () -> !tribute.boxed or (params...) -> !tribute.boxed
        // In actual implementation, we'd define a custom dialect for boxed values
        // For demonstration, we'll use i64 as a placeholder for boxed pointer
        let boxed_ptr_type = IntegerType::new(context, 64); // Pointer to boxed value
        let param_types: Vec<_> = params.iter().map(|_| boxed_ptr_type.into()).collect();
        let function_type = FunctionType::new(context, &param_types, &[boxed_ptr_type.into()]);
        
        // Create function operation
        let function_op = func::func(
            context,
            StringAttribute::new(context, &name),
            TypeAttribute::new(function_type.into()),
            {
                let region = Region::new();
                let block = Block::new(&[]);
                
                // Generate operations for function body
                let mut last_boxed_value = None;
                for (i, op) in body_ops.iter().enumerate() {
                    match op {
                        MlirOperation::BoxNumber { value } => {
                            println!("    Creating boxed number: {}", value);
                            // This would generate:
                            // 1. %ptr = call @tribute_alloc(i64 24) // sizeof(tribute_boxed_t)
                            // 2. %typed_ptr = bitcast %ptr to %tribute_boxed_t*
                            // 3. store i32 TYPE_NUMBER, %typed_ptr.type
                            // 4. store i32 1, %typed_ptr.ref_count  
                            // 5. store i64 %value, %typed_ptr.value.number
                            let boxed_name = format!("boxed_num_{}", i);
                            println!("      -> MLIR: %{} = call @tribute_box_number(i64 {})", boxed_name, value);
                            last_boxed_value = Some(boxed_name);
                        }
                        MlirOperation::BoxString { value } => {
                            println!("    Creating boxed string: \"{}\"", value);
                            // This would generate:
                            // 1. %str_data = call @tribute_alloc_string(i64 {})  // string length
                            // 2. call @llvm.memcpy(%str_data, @string_literal_{}, i64 {})
                            // 3. %boxed = call @tribute_box_string(%str_data, i64 {})
                            let boxed_name = format!("boxed_str_{}", i);
                            println!("      -> MLIR: %{} = call @tribute_box_string(ptr @str_literal_{}, i64 {})", 
                                     boxed_name, value.len(), value.len());
                            last_boxed_value = Some(boxed_name);
                        }
                        MlirOperation::Unbox { boxed_value, expected_type } => {
                            println!("    Unboxing {} as {:?}", boxed_value, expected_type);
                            // This would generate runtime type checking and extraction:
                            // 1. %type = load i32, %boxed_value.type
                            // 2. %ok = icmp eq i32 %type, TYPE_{}
                            // 3. br i1 %ok, label %extract, label %type_error
                            // 4. extract: %value = load {}, %boxed_value.value.{}
                            let unboxed_name = format!("unboxed_{}_{}", expected_type_name(expected_type), i);
                            println!("      -> MLIR: %{} = call @tribute_unbox_{}(ptr %{})", 
                                     unboxed_name, expected_type_name(expected_type), boxed_value);
                        }
                        MlirOperation::GcRetain { boxed_value } => {
                            println!("    GC Retain: {}", boxed_value);
                            // This would generate:
                            // %old_count = atomicrmw add i32* %boxed_value.ref_count, i32 1 acq_rel
                            println!("      -> MLIR: call @tribute_retain(ptr %{})", boxed_value);
                        }
                        MlirOperation::GcRelease { boxed_value } => {
                            println!("    GC Release: {}", boxed_value);
                            // This would generate:
                            // %old_count = atomicrmw sub i32* %boxed_value.ref_count, i32 1 acq_rel  
                            // %is_zero = icmp eq i32 %old_count, 1
                            // br i1 %is_zero, label %deallocate, label %continue
                            println!("      -> MLIR: call @tribute_release(ptr %{})", boxed_value);
                        }
                        MlirOperation::GcCollect => {
                            println!("    GC Collect triggered");
                            // This would generate a call to the mark-and-sweep collector
                            println!("      -> MLIR: call @tribute_gc_collect()");
                        }
                        MlirOperation::ListOp { operation } => {
                            match operation {
                                MlirListOperation::CreateEmpty { capacity } => {
                                    println!("    Creating empty list with capacity: {}", capacity);
                                    let list_name = format!("empty_list_{}", i);
                                    println!("      -> MLIR: %{} = call @tribute_box_list_empty(i64 {})", list_name, capacity);
                                    last_boxed_value = Some(list_name);
                                }
                                MlirListOperation::CreateFromArray { elements } => {
                                    println!("    Creating list from {} elements", elements.len());
                                    let list_name = format!("array_list_{}", i);
                                    println!("      -> MLIR: %{} = call @tribute_box_list_from_array(ptr %elements, i64 {})", 
                                             list_name, elements.len());
                                    last_boxed_value = Some(list_name);
                                }
                                MlirListOperation::Get { list, index } => {
                                    println!("    List get: {}[{}]", list, index);
                                    let result_name = format!("list_get_{}", i);
                                    println!("      -> MLIR: %{} = call @tribute_list_get(ptr %{}, i64 %{})", 
                                             result_name, list, index);
                                    println!("      -> O(1) random access");
                                    last_boxed_value = Some(result_name);
                                }
                                MlirListOperation::Set { list, index, value } => {
                                    println!("    List set: {}[{}] = {}", list, index, value);
                                    println!("      -> MLIR: call @tribute_list_set(ptr %{}, i64 %{}, ptr %{})", 
                                             list, index, value);
                                    println!("      -> O(1) random access modification");
                                }
                                MlirListOperation::Push { list, value } => {
                                    println!("    List push: {}.push({})", list, value);
                                    println!("      -> MLIR: call @tribute_list_push(ptr %{}, ptr %{})", list, value);
                                    println!("      -> Amortized O(1) append with automatic resize");
                                }
                                MlirListOperation::Pop { list } => {
                                    println!("    List pop: {}.pop()", list);
                                    let result_name = format!("list_pop_{}", i);
                                    println!("      -> MLIR: %{} = call @tribute_list_pop(ptr %{})", result_name, list);
                                    println!("      -> O(1) removal from end");
                                    last_boxed_value = Some(result_name);
                                }
                                MlirListOperation::Length { list } => {
                                    println!("    List length: {}.length", list);
                                    let result_name = format!("list_len_{}", i);
                                    println!("      -> MLIR: %{} = call @tribute_list_length(ptr %{})", result_name, list);
                                    println!("      -> O(1) length access");
                                    last_boxed_value = Some(result_name);
                                }
                            }
                        }
                        MlirOperation::Call { func: func_name, args } => {
                            println!("    Creating function call: {} with {} arguments", func_name, args.len());
                            
                            // For builtin functions, we need special handling
                            if func_name.starts_with("builtin_") {
                                let builtin_name = &func_name[8..]; // Remove "builtin_" prefix
                                println!("      Builtin function: {}", builtin_name);
                                
                                match builtin_name {
                                    "print_line" => {
                                        println!("      Would generate call to print_line builtin");
                                        // In actual implementation, this would generate appropriate LLVM/MLIR calls
                                        // to printf or similar system functions
                                    }
                                    "input_line" => {
                                        println!("      Would generate call to input_line builtin");
                                        // Would generate calls to scanf or similar input functions
                                    }
                                    // Arithmetic operations - now work with boxed values
                                    "+" => {
                                        println!("      Generating boxed arithmetic addition");
                                        // Direct call to runtime function that handles:
                                        // - Type checking and unboxing
                                        // - Arithmetic operation  
                                        // - Result boxing
                                        // - Reference counting (releases inputs, returns new value with ref_count=1)
                                        let result_name = format!("add_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_add_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        println!("        -> Runtime handles: unbox → add → box → ref_count management");
                                        last_boxed_value = Some(result_name);
                                    }
                                    "-" => {
                                        println!("      Generating boxed arithmetic subtraction");
                                        let result_name = format!("sub_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_sub_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    "*" => {
                                        println!("      Generating boxed arithmetic multiplication");
                                        let result_name = format!("mul_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_mul_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    "/" => {
                                        println!("      Generating boxed arithmetic division");
                                        let result_name = format!("div_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_div_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    "%" => {
                                        println!("      Generating boxed arithmetic modulo");
                                        let result_name = format!("mod_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_mod_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    // Comparison operations - return boxed boolean values
                                    "=" | "==" => {
                                        println!("      Generating boxed equality comparison");
                                        let result_name = format!("eq_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_eq_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    "<" => {
                                        println!("      Generating boxed less than comparison");
                                        let result_name = format!("lt_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_lt_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    ">" => {
                                        println!("      Generating boxed greater than comparison");
                                        let result_name = format!("gt_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_gt_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    "<=" => {
                                        println!("      Generating boxed less than or equal comparison");
                                        let result_name = format!("le_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_le_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    ">=" => {
                                        println!("      Generating boxed greater than or equal comparison");
                                        let result_name = format!("ge_result_{}", i);
                                        println!("        -> MLIR: %{} = call @tribute_ge_boxed(ptr %{}, ptr %{})", 
                                                 result_name, 
                                                 args.get(0).unwrap_or(&"arg0".to_string()),
                                                 args.get(1).unwrap_or(&"arg1".to_string()));
                                        last_boxed_value = Some(result_name);
                                    }
                                    _ => {
                                        println!("      Unknown builtin: {}", builtin_name);
                                    }
                                }
                            } else {
                                // User-defined function call
                                println!("      User function call: {}", func_name);
                                // Would generate func.call operation in MLIR
                            }
                            
                            // Track arguments
                            for (i, arg) in args.iter().enumerate() {
                                println!("        Arg {}: {}", i, arg);
                            }
                        }
                        MlirOperation::Variable { name } => {
                            println!("    Variable reference: {}", name);
                            // Variables are always boxed values in our system
                            // This would generate a load of the boxed value pointer
                            println!("      -> {}: loaded boxed value", name);
                            last_boxed_value = Some(name.clone());
                        }
                        MlirOperation::Return { value } => {
                            println!("    Return operation with boxed value: {:?}", value);
                            // This would generate the func.return operation with a boxed value
                            if let Some(val) = value {
                                println!("      -> returning boxed value: {}", val);
                                last_boxed_value = Some(val.clone());
                            }
                        }
                        MlirOperation::Placeholder { description } => {
                            println!("    Placeholder: {}", description);
                        }
                    }
                }
                
                // Add return operation for boxed values
                let return_boxed_value = if let Some(boxed_val) = last_boxed_value {
                    println!("    Function would return boxed value: {}", boxed_val);
                    boxed_val
                } else {
                    println!("    Function would return default boxed value: nil");
                    "boxed_nil".to_string()
                };
                
                // Create minimal function body for demonstration
                // All functions return boxed values in our system
                println!("    Creating minimal function body (return boxed: {})", return_boxed_value);
                
                region.append_block(block);
                region
            },
            &[],
            location,
        );
        
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