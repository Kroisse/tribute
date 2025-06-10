//! HIR to MLIR lowering implementation
//! 
//! This module implements the conversion from Tribute HIR to MLIR operations.
//! It serves as the middle tier between HIR (for development/prototyping with Salsa) 
//! and MLIR (for optimization/compilation).

use crate::{dialect::TributeDialect, errors::LoweringError, ops::TributeOps};
use melior::{
    ir::{Location, Module, Operation, Value, operation::OperationLike},
    Context,
};
use std::collections::HashMap;
use tribute_hir::hir::*;
use tribute_ast::{Identifier, Spanned};

/// HIR to MLIR lowerer
pub struct HirToMLIRLowerer<'c> {
    context: &'c Context,
    dialect: TributeDialect<'c>,
    ops: TributeOps<'c>,
    module: Module<'c>,
    
    // Symbol tables
    function_symbols: HashMap<String, Operation<'c>>,
    value_map: HashMap<String, Value<'c, 'c>>,
}

impl<'c> HirToMLIRLowerer<'c> {
    /// Create a new HIR to MLIR lowerer
    pub fn new(context: &'c Context) -> Self {
        let dialect = TributeDialect::new(context);
        let ops = TributeOps::new(context);
        let location = Location::unknown(context);
        let module = Module::new(location);
        
        Self {
            context,
            dialect,
            ops,
            module,
            function_symbols: HashMap::new(),
            value_map: HashMap::new(),
        }
    }

    /// Lower a HIR program to MLIR
    pub fn lower_program<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        program: HirProgram<'db>,
    ) -> Result<&Module<'c>, LoweringError> {
        let functions = program.functions(db);

        // First pass: declare all functions
        for (name, function) in functions.iter() {
            self.declare_function(db, name, *function)?;
        }

        // Second pass: implement function bodies
        for (name, function) in functions.iter() {
            self.implement_function(db, name, *function)?;
        }

        Ok(&self.module)
    }

    /// Declare a function (first pass)
    fn declare_function<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        name: &Identifier,
        function: HirFunction<'db>,
    ) -> Result<(), LoweringError> {
        let location = self.dialect.unknown_location();
        let params = function.params(db);
        
        // For now, all functions take and return dynamic values
        let param_types = vec![self.ops.types().value_type(); params.len()];
        let result_types = vec![self.ops.types().value_type()];
        
        let func_op = self.ops.func(name, &param_types, &result_types, location)?;
        self.function_symbols.insert(name.clone(), func_op);
        
        Ok(())
    }

    /// Implement a function body (second pass)
    fn implement_function<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        name: &Identifier,
        function: HirFunction<'db>,
    ) -> Result<(), LoweringError> {
        let params = function.params(db);
        let body = function.body(db);
        
        // Get the function operation we created in the first pass
        let _func_op = self.function_symbols.get(name)
            .ok_or_else(|| LoweringError::FunctionNotFound(name.clone()))?;
        
        // Create a new scope for the function body
        self.value_map.clear(); // Clear any previous values
        
        // TODO: Create entry block for the function
        // For now, we'll assume the function operation structure allows us to add a body
        // In a real implementation, we'd need to properly create basic blocks
        
        // Map parameters to block arguments
        // In MLIR, function parameters become block arguments of the entry block
        // For now, we'll create placeholder values for parameters
        // In a real implementation, these would be actual block arguments
        for param in params.iter() {
            // Create a placeholder value for the parameter
            // In reality, this would be a block argument value
            let param_op = self.ops.constant_f64(0.0, self.dialect.unknown_location())?;
            let param_value = param_op
                .result(0)
                .map_err(|_| LoweringError::OperationCreationFailed("No result for parameter placeholder".to_string()))?
                .into();
            let runtime_op = self.ops.to_runtime(param_value, self.dialect.unknown_location())?;
            let runtime_value = runtime_op
                .result(0)
                .map_err(|_| LoweringError::OperationCreationFailed("No result for to_runtime".to_string()))?
                .into();
            self.value_map.insert(param.clone(), runtime_value);
        }
        
        // Lower all body expressions
        let mut last_value = None;
        for expr in body.iter() {
            last_value = Some(self.lower_hir_expression(db, *expr)?);
        }
        
        // Create return operation with the last expression's value
        if let Some(value) = last_value {
            let _return_op = self.ops.return_op(Some(value), self.dialect.unknown_location())?;
        } else {
            // Empty function body - return void/unit
            let _return_op = self.ops.return_op(None, self.dialect.unknown_location())?;
        }
        
        Ok(())
    }

    /// Lower a HIR expression to MLIR operations
    fn lower_hir_expression<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        hir_expr: HirExpr<'db>,
    ) -> Result<Value<'c, 'c>, LoweringError> {
        let expr = hir_expr.expr(db);
        self.lower_expr(db, &expr)
    }
    
    /// Lower an Expr directly to MLIR operations
    fn lower_expr<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        expr: &Expr,
    ) -> Result<Value<'c, 'c>, LoweringError> {
        let location = self.dialect.unknown_location();
        
        match expr {
            Expr::Number(n) => {
                // Create f64 constant, then convert to runtime value
                let const_op = self.ops.constant_f64(*n as f64, location)?;
                let const_value = const_op
                    .result(0)
                    .map_err(|_| LoweringError::OperationCreationFailed("No result for constant".to_string()))?
                    .into();
                let runtime_op = self.ops.to_runtime(const_value, location)?;
                let runtime_value = runtime_op
                    .result(0)
                    .map_err(|_| LoweringError::OperationCreationFailed("No result for to_runtime".to_string()))?
                    .into();
                Ok(runtime_value)
            }

            Expr::StringInterpolation(s) => {
                // Handle string interpolation with segments
                if s.segments.is_empty() {
                    // Just a simple string constant
                    let const_op = self.ops.constant_string(&s.leading_text, location)?;
                    let const_value = const_op
                        .result(0)
                        .map_err(|_| LoweringError::OperationCreationFailed("No result for string constant".to_string()))?
                        .into();
                    let runtime_op = self.ops.to_runtime(const_value, location)?;
                    let runtime_value = runtime_op
                        .result(0)
                        .map_err(|_| LoweringError::OperationCreationFailed("No result for to_runtime".to_string()))?
                        .into();
                    Ok(runtime_value)
                } else {
                    // Complex interpolation - build string by concatenating parts
                    let mut result = {
                        let const_op = self.ops.constant_string(&s.leading_text, location)?;
                        let const_value = const_op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for string constant".to_string()))?
                            .into();
                        let runtime_op = self.ops.to_runtime(const_value, location)?;
                        runtime_op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for to_runtime".to_string()))?
                            .into()
                    };
                    
                    for segment in &s.segments {
                        // Lower the interpolated expression
                        let expr_value = self.lower_spanned_expression(db, &segment.interpolation)?;
                        
                        // Concatenate with the current result
                        let concat_op = self.ops.add(result, expr_value, location)?;
                        result = concat_op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for string concatenation".to_string()))?
                            .into();
                        
                        // Add the trailing text
                        if !segment.trailing_text.is_empty() {
                            let text_const = self.ops.constant_string(&segment.trailing_text, location)?;
                            let text_value = text_const
                                .result(0)
                                .map_err(|_| LoweringError::OperationCreationFailed("No result for string constant".to_string()))?
                                .into();
                            let text_runtime = self.ops.to_runtime(text_value, location)?;
                            let text_runtime_value = text_runtime
                                .result(0)
                                .map_err(|_| LoweringError::OperationCreationFailed("No result for to_runtime".to_string()))?
                                .into();
                            
                            let concat_op = self.ops.add(result, text_runtime_value, location)?;
                            result = concat_op
                                .result(0)
                                .map_err(|_| LoweringError::OperationCreationFailed("No result for string concatenation".to_string()))?
                                .into();
                        }
                    }
                    
                    Ok(result)
                }
            }

            Expr::Variable(id) => {
                self.value_map
                    .get(id)
                    .copied()
                    .ok_or_else(|| LoweringError::SymbolTableError(format!("Variable not found: {}", id)))
            }

            Expr::Call { func, args } => {
                // For HIR calls, func is a boxed spanned expression
                // Extract function name from the expression
                let func_name = self.extract_function_name(db, func)?;
                
                let arg_values: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| self.lower_spanned_expression(db, arg))
                    .collect();
                let arg_values = arg_values?;
                
                // Handle built-in arithmetic operations specially
                match func_name.as_str() {
                    "+" => {
                        if arg_values.len() != 2 {
                            return Err(LoweringError::InvalidArguments(
                                format!("+ requires exactly 2 arguments, got {}", arg_values.len())
                            ));
                        }
                        let op = self.ops.add(arg_values[0], arg_values[1], location)?;
                        let result = op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for add".to_string()))?
                            .into();
                        Ok(result)
                    }
                    "-" => {
                        if arg_values.len() != 2 {
                            return Err(LoweringError::InvalidArguments(
                                format!("- requires exactly 2 arguments, got {}", arg_values.len())
                            ));
                        }
                        let op = self.ops.sub(arg_values[0], arg_values[1], location)?;
                        let result = op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for sub".to_string()))?
                            .into();
                        Ok(result)
                    }
                    "*" => {
                        if arg_values.len() != 2 {
                            return Err(LoweringError::InvalidArguments(
                                format!("* requires exactly 2 arguments, got {}", arg_values.len())
                            ));
                        }
                        let op = self.ops.mul(arg_values[0], arg_values[1], location)?;
                        let result = op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for mul".to_string()))?
                            .into();
                        Ok(result)
                    }
                    "/" => {
                        if arg_values.len() != 2 {
                            return Err(LoweringError::InvalidArguments(
                                format!("/ requires exactly 2 arguments, got {}", arg_values.len())
                            ));
                        }
                        let op = self.ops.div(arg_values[0], arg_values[1], location)?;
                        let result = op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for div".to_string()))?
                            .into();
                        Ok(result)
                    }
                    // Regular function call
                    _ => {
                        let call_op = self.ops.call(&func_name, &arg_values, location)?;
                        let call_value = call_op
                            .result(0)
                            .map_err(|_| LoweringError::OperationCreationFailed("No result for call".to_string()))?
                            .into();
                        Ok(call_value)
                    }
                }
            }

            Expr::Let { var, value } => {
                // Lower the value expression
                let value_result = self.lower_spanned_expression(db, value)?;
                
                // Store in symbol table for future references
                self.value_map.insert(var.clone(), value_result);
                
                // Return the value (let expressions evaluate to their bound value)
                Ok(value_result)
            }

            Expr::Block(exprs) => {
                let mut last_value = None;
                
                // Lower all expressions in the block
                for expr in exprs.iter() {
                    last_value = Some(self.lower_spanned_expression(db, expr)?);
                }
                
                // Return the last expression's value, or unit if empty
                match last_value {
                    Some(value) => Ok(value),
                    None => {
                        // Empty block - return unit/void
                        // TODO: Define a unit/void constant operation
                        Err(LoweringError::UnsupportedHir("Empty blocks not yet supported".to_string()))
                    }
                }
            }

            Expr::Match { expr: _, cases: _ } => {
                // TODO: Implement pattern matching with MLIR scf operations
                Err(LoweringError::UnsupportedHir("Pattern matching not yet implemented".to_string()))
            }
        }
    }

    /// Lower a spanned HIR expression (helper method)
    fn lower_spanned_expression<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        spanned: &Spanned<Expr>,
    ) -> Result<Value<'c, 'c>, LoweringError> {
        let (expr, _span) = spanned;
        // TODO: Use span for location information
        
        // Lower the expression directly without creating tracked structs
        self.lower_expr(db, expr)
    }

    /// Extract function name from a function expression
    fn extract_function_name(
        &mut self,
        _db: &dyn salsa::Database,
        func_expr: &Spanned<Expr>,
    ) -> Result<String, LoweringError> {
        match &func_expr.0 {
            Expr::Variable(name) => Ok(name.clone()),
            _ => Err(LoweringError::UnsupportedHir(
                "Only variable function references supported".to_string()
            )),
        }
    }

    /// Get the context
    pub fn context(&self) -> &'c Context {
        self.context
    }

    /// Get the module
    pub fn module(&self) -> &Module<'c> {
        &self.module
    }
}