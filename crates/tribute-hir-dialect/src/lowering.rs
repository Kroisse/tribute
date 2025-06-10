//! AST to MLIR lowering implementation

use crate::{dialect::TributeDialect, errors::LoweringError, ops::TributeOps};
use melior::{
    ir::{Location, Module, Operation, Value, operation::OperationLike},
    Context,
};
use std::collections::HashMap;
use tribute_ast::{
    ast::*,
    database::Db,
};

/// AST to MLIR lowerer (currently unused, prefer HIR lowering)
#[allow(dead_code)]
pub struct AstToMLIRLowerer<'c> {
    context: &'c Context,
    dialect: TributeDialect<'c>,
    ops: TributeOps<'c>,
    module: Module<'c>,
    
    // Symbol tables
    function_symbols: HashMap<String, Operation<'c>>,
    value_map: HashMap<String, Value<'c, 'c>>,
}

impl<'c> AstToMLIRLowerer<'c> {
    /// Create a new AST to MLIR lowerer
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

    /// Lower a program from AST to MLIR
    pub fn lower_program<'db>(
        &mut self,
        db: &'db dyn Db,
        program: Program<'db>,
    ) -> Result<&Module<'c>, LoweringError> {
        // First pass: declare all functions
        for item in program.items(db) {
            match item.kind(db) {
                ItemKind::Function(func) => {
                    self.declare_function(db, *func)?;
                }
                _ => {
                    // Skip non-function items in first pass
                }
            }
        }

        // Second pass: implement function bodies
        for item in program.items(db) {
            match item.kind(db) {
                ItemKind::Function(func) => {
                    self.implement_function(db, *func)?;
                }
                _ => {
                    // Skip non-function items in second pass for now
                }
            }
        }

        Ok(&self.module)
    }

    /// Declare a function (first pass)
    fn declare_function<'db>(
        &mut self,
        db: &'db dyn Db,
        func: FunctionDefinition<'db>,
    ) -> Result<(), LoweringError> {
        let name = func.name(db);
        let location = self.dialect.unknown_location();
        
        // For now, all functions take and return dynamic values
        let param_types = vec![self.ops.types().value_type(); func.parameters(db).len()];
        let result_types = vec![self.ops.types().value_type()];
        
        let func_op = self.ops.func(&name, &param_types, &result_types, location)?;
        self.function_symbols.insert(name, func_op);
        
        Ok(())
    }

    /// Implement a function body (second pass)
    fn implement_function<'db>(
        &mut self,
        db: &'db dyn Db,
        func: FunctionDefinition<'db>,
    ) -> Result<(), LoweringError> {
        let _name = func.name(db);
        let _params = func.parameters(db);
        let _body = func.body(db);
        
        // TODO: Implement function body lowering
        // This would involve creating a basic block and lowering the body expression
        
        Ok(())
    }

    /// Lower an expression to MLIR operations
    #[allow(dead_code)]
    fn lower_expression(
        &mut self,
        expr: Spanned<Expr>,
    ) -> Result<Value<'c, 'c>, LoweringError> {
        let location = self.dialect.unknown_location();
        let (expr, _span) = expr;
        
        match expr {
            Expr::Number(n) => {
                // Create f64 constant, then convert to runtime value
                let const_op = self.ops.constant_f64(n as f64, location)?;
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
                // For now, just use the leading text as a simple string
                let text = &s.leading_text;
                let const_op = self.ops.constant_string(text, location)?;
                let const_value = const_op
                    .result(0)
                    .map_err(|_| LoweringError::OperationCreationFailed("No result for string constant".to_string()))?
                    .into();
                Ok(const_value)
            }

            Expr::Binary(binary_expr) => {
                let lhs = self.lower_expression(*binary_expr.left)?;
                let rhs = self.lower_expression(*binary_expr.right)?;
                
                let result_op = match binary_expr.operator {
                    BinaryOperator::Add => self.ops.add(lhs, rhs, location)?,
                    BinaryOperator::Subtract => self.ops.sub(lhs, rhs, location)?,
                    BinaryOperator::Multiply => self.ops.mul(lhs, rhs, location)?,
                    BinaryOperator::Divide => self.ops.div(lhs, rhs, location)?,
                };
                
                let result_value = result_op
                    .result(0)
                    .map_err(|_| LoweringError::OperationCreationFailed("No result for binary op".to_string()))?
                    .into();
                Ok(result_value)
            }

            Expr::Call(call_expr) => {
                let func_name = &call_expr.function;
                
                let arg_values: Result<Vec<_>, _> = call_expr.arguments
                    .into_iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect();
                let arg_values = arg_values?;
                
                let call_op = self.ops.call(func_name, &arg_values, location)?;
                let call_value = call_op
                    .result(0)
                    .map_err(|_| LoweringError::OperationCreationFailed("No result for call".to_string()))?
                    .into();
                Ok(call_value)
            }

            Expr::Identifier(id) => {
                self.value_map
                    .get(&id)
                    .copied()
                    .ok_or_else(|| LoweringError::SymbolTableError(format!("Variable not found: {}", id)))
            }

            _ => Err(LoweringError::UnsupportedAst(format!("Expression kind not supported yet: {:?}", expr))),
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