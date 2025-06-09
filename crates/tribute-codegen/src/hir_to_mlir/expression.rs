//! HIR expression to MLIR conversion logic.

use crate::hir_to_mlir::types::{MlirExpressionResult, MlirFunction, MlirOperation};
use tribute_hir::{HirExpr, HirFunction, Expr};

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