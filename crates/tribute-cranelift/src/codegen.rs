//! Code generation from HIR to Cranelift IR
//!
//! This module implements the core translation from Tribute's HIR
//! to Cranelift's intermediate representation.

use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, FuncRef, InstBuilder, Signature, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Linkage, Module};

use salsa::Database;
use tribute_ast::{Identifier, Spanned};
use tribute_hir::hir::{Expr, HirExpr, HirFunction, HirProgram};

use crate::errors::{BoxError, CompilationError, CompilationResult};
use crate::runtime::RuntimeFunctions;
use crate::types::TributeTypes;

/// Code generator for Tribute → Cranelift IR
pub struct CodeGenerator<'m, M: Module> {
    module: &'m mut M,
    runtime: &'m RuntimeFunctions,
    /// Map from function names to their IDs
    function_map: HashMap<String, FuncId>,
}

impl<'m, M: Module> CodeGenerator<'m, M> {
    /// Create a new code generator
    pub fn new(module: &'m mut M, runtime: &'m RuntimeFunctions) -> Self {
        Self {
            module,
            runtime,
            function_map: HashMap::new(),
        }
    }
    
    /// Compile a HIR program
    pub fn compile_program<'db>(
        &mut self,
        db: &'db dyn Database,
        program: HirProgram<'db>,
    ) -> CompilationResult<()> {
        let functions = program.functions(db);
        
        // First pass: declare all functions
        for (name, func) in functions.iter() {
            self.declare_function(db, name, *func)?;
        }
        
        // Second pass: compile function bodies
        for (name, func) in functions.iter() {
            self.compile_function(db, name, *func)?;
        }
        
        // Create main function that evaluates top-level expressions
        self.compile_main(db, program)?;
        
        Ok(())
    }
    
    /// Declare a function (first pass)
    fn declare_function<'db>(
        &mut self,
        db: &'db dyn Database,
        name: &Identifier,
        func: HirFunction<'db>,
    ) -> CompilationResult<()> {
        let params = func.params(db);
        
        // All Tribute functions take and return pointer-sized values
        let mut sig = Signature::new(CallConv::SystemV);
        for _ in 0..params.len() {
            sig.params.push(TributeTypes::value_param());
        }
        sig.returns.push(TributeTypes::value_param());
        
        let func_id = self.module
            .declare_function(name, Linkage::Local, &sig).box_err()?;
        
        self.function_map.insert(name.clone(), func_id);
        Ok(())
    }
    
    /// Compile a function body (second pass)
    fn compile_function<'db>(
        &mut self,
        db: &'db dyn Database,
        name: &Identifier,
        func: HirFunction<'db>,
    ) -> CompilationResult<()> {
        let func_id = self.function_map[name];
        let params = func.params(db);
        let body = func.body(db);
        
        // Create a new function context
        let mut ctx = self.module.make_context();
        ctx.func.signature = self.module.declarations().get_function_decl(func_id).signature.clone();
        
        // Build the function
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        
        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);
        
        // Map parameters to variables
        let param_values: Vec<Value> = builder.block_params(entry_block).to_vec();
        
        // Create a new lowering context for this function
        let mut lowerer = FunctionLowerer::new(
            &mut builder,
            self.module,
            self.runtime,
            &self.function_map,
        );
        
        // Define parameters
        for (i, param_name) in params.iter().enumerate() {
            lowerer.define_variable(param_name, param_values[i]);
        }
        
        // Compile function body
        let mut last_value = None;
        for expr in body.iter() {
            last_value = Some(lowerer.lower_hir_expr(db, *expr)?);
        }
        
        // Return the last value or create a unit value
        let return_value = match last_value {
            Some(val) => val,
            None => lowerer.create_unit_value()?,
        };
        builder.ins().return_(&[return_value]);
        
        // Finalize the function
        builder.finalize();
        
        // Define the function in the module
        self.module.define_function(func_id, &mut ctx).box_err()?;
        
        Ok(())
    }
    
    /// Compile the main function
    fn compile_main<'db>(
        &mut self,
        _db: &'db dyn Database,
        _program: HirProgram<'db>,
    ) -> CompilationResult<()> {
        // Create main function signature
        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns.push(AbiParam::new(cranelift_codegen::ir::types::I32));
        
        let main_id = self.module
            .declare_function("main", Linkage::Export, &sig).box_err()?;
        
        // Create context for main function
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        
        // Build the main function
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        
        // Create entry block
        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);
        
        // TODO: Evaluate top-level expressions
        // For now, just return 0
        let zero = builder.ins().iconst(cranelift_codegen::ir::types::I32, 0);
        builder.ins().return_(&[zero]);
        
        // Finalize the function
        builder.finalize();
        
        // Define the function in the module
        self.module.define_function(main_id, &mut ctx).box_err()?;
        
        Ok(())
    }
}

/// Per-function lowering context
struct FunctionLowerer<'a, 'b, M: Module> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut M,
    runtime: &'a RuntimeFunctions,
    function_map: &'a HashMap<String, FuncId>,
    /// Map from variable names to their current values
    variables: HashMap<String, Variable>,
    /// Counter for generating unique variables
    var_counter: usize,
}

impl<'a, 'b, M: Module> FunctionLowerer<'a, 'b, M> {
    fn new(
        builder: &'a mut FunctionBuilder<'b>,
        module: &'a mut M,
        runtime: &'a RuntimeFunctions,
        function_map: &'a HashMap<String, FuncId>,
    ) -> Self {
        Self {
            builder,
            module,
            runtime,
            function_map,
            variables: HashMap::new(),
            var_counter: 0,
        }
    }
    
    /// Create a new variable
    fn create_variable(&mut self) -> Variable {
        let var = Variable::from_u32(self.var_counter as u32);
        self.var_counter += 1;
        self.builder.declare_var(var, TributeTypes::pointer_type());
        var
    }
    
    /// Define a variable with a value
    fn define_variable(&mut self, name: &str, value: Value) {
        let var = self.create_variable();
        self.builder.def_var(var, value);
        self.variables.insert(name.to_string(), var);
    }
    
    /// Get the value of a variable
    fn use_variable(&mut self, name: &str) -> CompilationResult<Value> {
        let var = self.variables.get(name)
            .ok_or_else(|| CompilationError::TypeError(format!("Undefined variable: {}", name)))?;
        Ok(self.builder.use_var(*var))
    }
    
    /// Create a unit value
    fn create_unit_value(&mut self) -> CompilationResult<Value> {
        // For now, use null pointer as unit
        // TODO: Create proper unit value in runtime
        Ok(self.builder.ins().null(TributeTypes::pointer_type()))
    }
    
    /// Lower a HIR expression
    fn lower_hir_expr<'db>(
        &mut self,
        db: &'db dyn Database,
        expr: HirExpr<'db>,
    ) -> CompilationResult<Value> {
        let expr_data = expr.expr(db);
        self.lower_expr(db, &expr_data)
    }
    
    /// Lower an expression
    fn lower_expr(
        &mut self,
        db: &dyn Database,
        expr: &Expr,
    ) -> CompilationResult<Value> {
        match expr {
            Expr::Number(n) => self.lower_number(*n),
            Expr::StringInterpolation(s) => self.lower_string_interpolation(db, s),
            Expr::Variable(name) => self.use_variable(name),
            Expr::Call { func, args } => self.lower_call(db, func, args),
            Expr::Let { var, value } => self.lower_let(db, var, value),
            Expr::Block(exprs) => self.lower_block(db, exprs),
            Expr::Match { .. } => Err(CompilationError::UnsupportedFeature(
                "Pattern matching not yet implemented".to_string()
            )),
        }
    }
    
    /// Lower a number literal
    fn lower_number(&mut self, n: i64) -> CompilationResult<Value> {
        // Create f64 constant
        let float_val = self.builder.ins().f64const(n as f64);
        
        // Call runtime to create number value
        let value_from_number = self.import_runtime_func(self.runtime.value_from_number)?;
        let value = self.builder.ins().call(value_from_number, &[float_val]);
        
        Ok(self.builder.inst_results(value)[0])
    }
    
    /// Lower string interpolation
    fn lower_string_interpolation(
        &mut self,
        _db: &dyn Database,
        s: &tribute_hir::hir::StringInterpolation,
    ) -> CompilationResult<Value> {
        if s.segments.is_empty() {
            // Simple string constant
            self.lower_string_literal(&s.leading_text)
        } else {
            // Complex interpolation - not yet implemented
            Err(CompilationError::UnsupportedFeature(
                "String interpolation not yet implemented".to_string()
            ))
        }
    }
    
    /// Lower a string literal
    fn lower_string_literal(&mut self, text: &str) -> CompilationResult<Value> {
        // For now, create a dummy string
        // TODO: Emit string data and get pointer
        let null = self.builder.ins().null(TributeTypes::pointer_type());
        let len = self.builder.ins().iconst(TributeTypes::size_type(), text.len() as i64);
        
        let value_from_string = self.import_runtime_func(self.runtime.value_from_string)?;
        let value = self.builder.ins().call(value_from_string, &[null, len]);
        
        Ok(self.builder.inst_results(value)[0])
    }
    
    /// Lower a function call
    fn lower_call(
        &mut self,
        db: &dyn Database,
        func: &Spanned<Expr>,
        args: &[Spanned<Expr>],
    ) -> CompilationResult<Value> {
        // Extract function name
        let func_name = match &func.0 {
            Expr::Variable(name) => name,
            _ => return Err(CompilationError::UnsupportedFeature(
                "Only direct function calls supported".to_string()
            )),
        };
        
        // Lower arguments
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.lower_expr(db, &arg.0)?);
        }
        
        // Handle built-in operations
        match func_name.as_str() {
            "+" => self.lower_binary_op(self.runtime.value_add, &arg_values),
            "-" => self.lower_binary_op(self.runtime.value_sub, &arg_values),
            "*" => self.lower_binary_op(self.runtime.value_mul, &arg_values),
            "/" => self.lower_binary_op(self.runtime.value_div, &arg_values),
            "print_line" => self.lower_print_line(&arg_values),
            "input_line" => self.lower_input_line(&arg_values),
            _ => {
                // User-defined function
                let func_id = self.function_map.get(func_name)
                    .ok_or_else(|| CompilationError::FunctionNotFound(func_name.clone()))?;
                
                let func_ref = self.import_user_func(*func_id)?;
                let result = self.builder.ins().call(func_ref, &arg_values);
                Ok(self.builder.inst_results(result)[0])
            }
        }
    }
    
    /// Lower a binary operation
    fn lower_binary_op(&mut self, runtime_func: FuncId, args: &[Value]) -> CompilationResult<Value> {
        if args.len() != 2 {
            return Err(CompilationError::TypeError(
                format!("Binary operation requires 2 arguments, got {}", args.len())
            ));
        }
        
        let func_ref = self.import_runtime_func(runtime_func)?;
        let result = self.builder.ins().call(func_ref, &[args[0], args[1]]);
        Ok(self.builder.inst_results(result)[0])
    }
    
    /// Lower print_line built-in
    fn lower_print_line(&mut self, args: &[Value]) -> CompilationResult<Value> {
        if args.len() != 1 {
            return Err(CompilationError::TypeError(
                format!("print_line requires 1 argument, got {}", args.len())
            ));
        }
        
        let func_ref = self.import_runtime_func(self.runtime.builtin_print_line)?;
        self.builder.ins().call(func_ref, &[args[0]]);
        
        // Return unit
        self.create_unit_value()
    }
    
    /// Lower input_line built-in
    fn lower_input_line(&mut self, args: &[Value]) -> CompilationResult<Value> {
        if !args.is_empty() {
            return Err(CompilationError::TypeError(
                format!("input_line requires 0 arguments, got {}", args.len())
            ));
        }
        
        let func_ref = self.import_runtime_func(self.runtime.builtin_input_line)?;
        let result = self.builder.ins().call(func_ref, &[]);
        Ok(self.builder.inst_results(result)[0])
    }
    
    /// Lower a let binding
    fn lower_let(
        &mut self,
        db: &dyn Database,
        var: &str,
        value: &Spanned<Expr>,
    ) -> CompilationResult<Value> {
        let val = self.lower_expr(db, &value.0)?;
        self.define_variable(var, val);
        Ok(val)
    }
    
    /// Lower a block expression
    fn lower_block(
        &mut self,
        db: &dyn Database,
        exprs: &[Spanned<Expr>],
    ) -> CompilationResult<Value> {
        let mut last_value = None;
        
        for expr in exprs {
            last_value = Some(self.lower_expr(db, &expr.0)?);
        }
        
        match last_value {
            Some(val) => Ok(val),
            None => self.create_unit_value(),
        }
    }
    
    /// Import a runtime function
    fn import_runtime_func(&mut self, func_id: FuncId) -> CompilationResult<FuncRef> {
        let func_ref = self.module
            .declare_func_in_func(func_id, self.builder.func);
        Ok(func_ref)
    }
    
    /// Import a user-defined function
    fn import_user_func(&mut self, func_id: FuncId) -> CompilationResult<FuncRef> {
        let func_ref = self.module
            .declare_func_in_func(func_id, self.builder.func);
        Ok(func_ref)
    }
}