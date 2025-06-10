//! MLIR-based evaluator for Tribute

use tribute_hir_dialect::errors::EvaluationError;
use melior::{ir::Module, Context};
use std::{collections::HashMap, fmt};

impl fmt::Display for TributeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TributeValue::Number(n) => write!(f, "{}", n),
            TributeValue::String(s) => write!(f, "{}", s),
            TributeValue::Boolean(b) => write!(f, "{}", b),
            TributeValue::Unit => write!(f, "()"),
            TributeValue::List(items) => {
                let item_strings: Vec<_> = items.iter().map(|item| format!("{}", item)).collect();
                write!(f, "[{}]", item_strings.join(", "))
            },
            TributeValue::Function { name, .. } => write!(f, "<function {}>", name),
        }
    }
}

/// Runtime value for Tribute evaluation
#[derive(Debug, Clone)]
pub enum TributeValue {
    Number(f64),
    String(String),
    Boolean(bool),
    Unit,
    List(Vec<TributeValue>),
    Function {
        name: String,
        params: Vec<String>,
        // In a full implementation, this would store MLIR function reference
    },
}

impl TributeValue {

    /// Check if value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            TributeValue::Boolean(b) => *b,
            TributeValue::Unit => false,
            TributeValue::Number(n) => *n != 0.0,
            TributeValue::String(s) => !s.is_empty(),
            TributeValue::List(items) => !items.is_empty(),
            TributeValue::Function { .. } => true,
        }
    }
}

/// MLIR-based evaluator
pub struct MLIREvaluator<'c> {
    context: &'c Context,
    module: Module<'c>,
    
    // Runtime environment
    globals: HashMap<String, TributeValue>,
    locals: Vec<HashMap<String, TributeValue>>, // Stack of local scopes
}

impl<'c> MLIREvaluator<'c> {
    /// Create a new MLIR evaluator
    pub fn new(context: &'c Context, module: Module<'c>) -> Self {
        let mut globals = HashMap::new();
        
        // Add built-in functions
        globals.insert(
            "print_line".to_string(),
            TributeValue::Function {
                name: "print_line".to_string(),
                params: vec!["value".to_string()],
            },
        );
        
        globals.insert(
            "input_line".to_string(),
            TributeValue::Function {
                name: "input_line".to_string(),
                params: vec![],
            },
        );

        Self {
            context,
            module,
            globals,
            locals: Vec::new(),
        }
    }

    /// Evaluate the MLIR module
    pub fn evaluate(&mut self) -> Result<TributeValue, EvaluationError> {
        // For now, we'll implement a simple interpreter that walks the MLIR
        // In a full implementation, this would either:
        // 1. Use MLIR's ExecutionEngine for JIT compilation, or
        // 2. Implement a custom MLIR interpreter
        
        // TODO: Implement actual MLIR evaluation
        // For now, return a placeholder
        Ok(TributeValue::Unit)
    }

    /// Evaluate a function call
    pub fn call_function(
        &mut self,
        name: &str,
        args: Vec<TributeValue>,
    ) -> Result<TributeValue, EvaluationError> {
        match name {
            "print_line" => {
                if args.len() != 1 {
                    return Err(EvaluationError::FunctionCallError(
                        "print_line expects 1 argument".to_string(),
                    ));
                }
                println!("{}", args[0]);
                Ok(TributeValue::Unit)
            }

            "input_line" => {
                if !args.is_empty() {
                    return Err(EvaluationError::FunctionCallError(
                        "input_line expects 0 arguments".to_string(),
                    ));
                }
                
                use std::io::{self, Write};
                print!("> ");
                io::stdout().flush().unwrap();
                
                let mut input = String::new();
                io::stdin()
                    .read_line(&mut input)
                    .map_err(|e| EvaluationError::MLIRExecutionError(format!("Input error: {}", e)))?;
                
                // Remove trailing newline
                if input.ends_with('\n') {
                    input.pop();
                    if input.ends_with('\r') {
                        input.pop();
                    }
                }
                
                Ok(TributeValue::String(input))
            }

            _ => {
                // Look up user-defined function
                if let Some(TributeValue::Function { .. }) = self.globals.get(name) {
                    // TODO: Implement user-defined function calls via MLIR
                    Err(EvaluationError::FunctionCallError(
                        "User-defined functions not implemented yet".to_string(),
                    ))
                } else {
                    Err(EvaluationError::FunctionCallError(format!(
                        "Function not found: {}",
                        name
                    )))
                }
            }
        }
    }

    /// Perform binary operation
    pub fn binary_op(
        &self,
        op: &str,
        left: TributeValue,
        right: TributeValue,
    ) -> Result<TributeValue, EvaluationError> {
        match (op, left, right) {
            ("add", TributeValue::Number(l), TributeValue::Number(r)) => {
                Ok(TributeValue::Number(l + r))
            }
            ("sub", TributeValue::Number(l), TributeValue::Number(r)) => {
                Ok(TributeValue::Number(l - r))
            }
            ("mul", TributeValue::Number(l), TributeValue::Number(r)) => {
                Ok(TributeValue::Number(l * r))
            }
            ("div", TributeValue::Number(l), TributeValue::Number(r)) => {
                if r == 0.0 {
                    Err(EvaluationError::DivisionByZero)
                } else {
                    Ok(TributeValue::Number(l / r))
                }
            }
            ("add", TributeValue::String(l), TributeValue::String(r)) => {
                Ok(TributeValue::String(l + &r))
            }
            _ => Err(EvaluationError::RuntimeTypeError(format!(
                "Invalid binary operation: {} with operand types",
                op
            ))),
        }
    }

    /// Push a new local scope
    pub fn push_scope(&mut self) {
        self.locals.push(HashMap::new());
    }

    /// Pop the current local scope
    pub fn pop_scope(&mut self) {
        self.locals.pop();
    }

    /// Set a variable in the current scope
    pub fn set_variable(&mut self, name: String, value: TributeValue) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, value);
        } else {
            self.globals.insert(name, value);
        }
    }

    /// Get a variable from the current scope
    pub fn get_variable(&self, name: &str) -> Result<TributeValue, EvaluationError> {
        // Search local scopes from innermost to outermost
        for scope in self.locals.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Ok(value.clone());
            }
        }

        // Search globals
        if let Some(value) = self.globals.get(name) {
            Ok(value.clone())
        } else {
            Err(EvaluationError::VariableNotFound(name.to_string()))
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