//! Code generation from HIR to Cranelift IR
//!
//! This module implements the core translation from Tribute's HIR
//! to Cranelift's intermediate representation.

use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, FuncRef, InstBuilder, Signature, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};

use tribute_ast::{Identifier, Spanned};
use tribute_hir::hir::{Expr, HirExpr, HirFunction, HirProgram, Literal, MatchCase, Pattern};

use crate::CompilationError;
use crate::errors::{CompilationErrorKind, CompilationResult};
use crate::runtime::RuntimeFunctions;
use crate::types::TributeTypes;
use tribute_core::{Db, TargetInfo};

/// String constant table for managing compile-time strings
#[derive(Debug)]
pub struct StringConstantTable {
    /// Map from string content to its offset in the data section
    strings: HashMap<String, u32>,
    /// Raw string data that will be placed in .rodata
    data: Vec<u8>,
    /// Data ID for the string table in Cranelift
    data_id: Option<DataId>,
}

impl Default for StringConstantTable {
    fn default() -> Self {
        Self::new()
    }
}

impl StringConstantTable {
    /// Create a new empty string constant table
    pub fn new() -> Self {
        Self {
            strings: HashMap::new(),
            data: Vec::new(),
            data_id: None,
        }
    }

    /// Add a string constant and return its offset
    pub fn add_string(&mut self, text: &str) -> u32 {
        if let Some(&offset) = self.strings.get(text) {
            return offset;
        }

        let offset = self.data.len() as u32;

        // Add string data (length-based, no null terminator needed)
        let bytes = text.as_bytes();
        self.data.extend_from_slice(bytes);

        self.strings.insert(text.to_string(), offset);
        offset
    }

    /// Get the total size of the string data
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Get the raw data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get or create the data ID for this table
    pub fn get_or_create_data_id<M: Module>(
        &mut self,
        module: &mut M,
    ) -> Result<DataId, Box<cranelift_module::ModuleError>> {
        if let Some(data_id) = self.data_id {
            return Ok(data_id);
        }

        // Create a data section for the string constants
        let data_id =
            module.declare_data("__tribute_string_constants", Linkage::Local, true, false)?;
        self.data_id = Some(data_id);
        Ok(data_id)
    }

    /// Finalize the string table in the module
    pub fn finalize<M: Module>(
        &mut self,
        module: &mut M,
    ) -> Result<(), Box<cranelift_module::ModuleError>> {
        if self.data.is_empty() {
            return Ok(());
        }

        let data_id = self.get_or_create_data_id(module)?;

        // Create the data description
        let mut data_desc = DataDescription::new();
        data_desc.define(self.data.clone().into_boxed_slice());

        // Define the data in the module
        module.define_data(data_id, &data_desc)?;

        Ok(())
    }
}

/// Code generator for Tribute → Cranelift IR
pub struct CodeGenerator<'m, M: Module> {
    module: &'m mut M,
    runtime: &'m RuntimeFunctions,
    target: &'m TargetInfo,
    /// Map from function names to their IDs
    function_map: HashMap<String, FuncId>,
    /// String constant table for compile-time strings
    string_table: StringConstantTable,
}

impl<'m, M: Module> CodeGenerator<'m, M> {
    /// Create a new code generator
    pub fn new(module: &'m mut M, runtime: &'m RuntimeFunctions, target: &'m TargetInfo) -> Self {
        Self {
            module,
            runtime,
            target,
            function_map: HashMap::new(),
            string_table: StringConstantTable::new(),
        }
    }

    /// Add a string literal to the constant table and return its offset
    pub fn add_string_literal(&mut self, text: &str) -> u32 {
        self.string_table.add_string(text)
    }

    /// Get the data ID for the string constant table
    pub fn get_string_table_data_id(&mut self) -> CompilationResult<DataId> {
        Ok(self.string_table.get_or_create_data_id(self.module)?)
    }

    /// Collect all string literals from a function
    fn collect_string_literals<'db>(
        &mut self,
        db: &'db dyn Db,
        func: HirFunction<'db>,
    ) -> CompilationResult<HashMap<String, u32>> {
        let mut literals = HashMap::new();
        let body = func.body(db);

        for expr in body.iter() {
            self.collect_string_literals_from_expr(db, *expr, &mut literals)?;
        }

        Ok(literals)
    }

    /// Recursively collect string literals from an expression
    fn collect_string_literals_from_expr<'db>(
        &mut self,
        db: &'db dyn Db,
        expr: HirExpr<'db>,
        literals: &mut HashMap<String, u32>,
    ) -> CompilationResult<()> {
        let expr_data = expr.expr(db);
        match &expr_data {
            Expr::StringInterpolation(s) => {
                if s.segments.is_empty() {
                    // Simple string literal
                    let offset = self.add_string_literal(&s.leading_text);
                    literals.insert(s.leading_text.clone(), offset);
                }
                // TODO: Handle complex interpolation
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    // arg is Spanned<Expr>, need to extract the expression
                    self.collect_string_literals_from_expr_data(&arg.0, literals)?;
                }
            }
            Expr::Let { value, .. } => {
                // value is Box<Spanned<Expr>>
                self.collect_string_literals_from_expr_data(&value.0, literals)?;
            }
            Expr::Block(exprs) => {
                for expr in exprs {
                    // expr is Spanned<Expr>
                    self.collect_string_literals_from_expr_data(&expr.0, literals)?;
                }
            }
            _ => {
                // Other expressions don't contain string literals
            }
        }
        Ok(())
    }

    /// Recursively collect string literals from an expression (raw Expr)
    fn collect_string_literals_from_expr_data(
        &mut self,
        expr: &Expr,
        literals: &mut HashMap<String, u32>,
    ) -> CompilationResult<()> {
        match expr {
            Expr::StringInterpolation(s) => {
                if s.segments.is_empty() {
                    // Simple string literal
                    let offset = self.add_string_literal(&s.leading_text);
                    literals.insert(s.leading_text.clone(), offset);
                }
                // TODO: Handle complex interpolation
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    // arg is Spanned<Expr>, need to extract the expression
                    self.collect_string_literals_from_expr_data(&arg.0, literals)?;
                }
            }
            Expr::Let { value, .. } => {
                // value is Box<Spanned<Expr>>
                self.collect_string_literals_from_expr_data(&value.0, literals)?;
            }
            Expr::Block(exprs) => {
                for expr in exprs {
                    // expr is Spanned<Expr>
                    self.collect_string_literals_from_expr_data(&expr.0, literals)?;
                }
            }
            _ => {
                // Other expressions don't contain string literals
            }
        }
        Ok(())
    }

    /// Compile a HIR program
    pub fn compile_program<'db>(
        &mut self,
        db: &'db dyn Db,
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

        // Finalize string constant table
        self.string_table.finalize(self.module)?;

        Ok(())
    }

    /// Declare a function (first pass)
    fn declare_function<'db>(
        &mut self,
        db: &'db dyn Db,
        name: &Identifier,
        func: HirFunction<'db>,
    ) -> CompilationResult<()> {
        let params = func.params(db);

        // All Tribute functions take and return pointer-sized values
        let mut sig = Signature::new(CallConv::SystemV);
        for _ in 0..params.len() {
            sig.params.push(TributeTypes::value_param(db, self.target));
        }
        sig.returns.push(TributeTypes::value_param(db, self.target));

        // Rename user's main function to avoid conflict with C main
        let func_name = if name == "main" {
            "_tribute_main"
        } else {
            name
        };

        let func_id = self
            .module
            .declare_function(func_name, Linkage::Local, &sig)?;

        self.function_map.insert(name.clone(), func_id);
        Ok(())
    }

    /// Compile a function body (second pass)
    fn compile_function<'db>(
        &mut self,
        db: &'db dyn Db,
        name: &Identifier,
        func: HirFunction<'db>,
    ) -> CompilationResult<()> {
        let func_id = self.function_map[name];
        let params = func.params(db);
        let body = func.body(db);

        // Create a new function context
        let mut ctx = self.module.make_context();
        ctx.func.signature = self
            .module
            .declarations()
            .get_function_decl(func_id)
            .signature
            .clone();

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

        // Collect string literals from this function
        let string_literals = self.collect_string_literals(db, func)?;
        let string_table_data_id = if !string_literals.is_empty() {
            Some(self.get_string_table_data_id()?)
        } else {
            None
        };

        // Create a new lowering context for this function
        let mut lowerer = FunctionLowerer::new(
            &mut builder,
            self.module,
            self.runtime,
            self.target,
            db,
            &self.function_map,
            string_literals,
            string_table_data_id,
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
        self.module.define_function(func_id, &mut ctx)?;

        Ok(())
    }

    /// Compile the main function
    fn compile_main<'db>(
        &mut self,
        db: &'db dyn Db,
        program: HirProgram<'db>,
    ) -> CompilationResult<()> {
        // Create main function signature
        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns
            .push(AbiParam::new(cranelift_codegen::ir::types::I32));

        let main_id = self
            .module
            .declare_function("main", Linkage::Export, &sig)?;

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

        // Check if there's a main function defined by the user
        let functions = program.functions(db);
        if let Some(_main_func) = functions.get("main") {
            // Call the user-defined main function
            if let Some(main_func_id) = self.function_map.get("main") {
                let main_func_ref = self
                    .module
                    .declare_func_in_func(*main_func_id, builder.func);
                let _result = builder.ins().call(main_func_ref, &[]);

                // Return 0 for successful execution (ignore the main function's return value)
                let zero = builder.ins().iconst(cranelift_codegen::ir::types::I32, 0);
                builder.ins().return_(&[zero]);
            } else {
                // Main function not yet compiled - return error code
                let error_code = builder.ins().iconst(cranelift_codegen::ir::types::I32, 1);
                builder.ins().return_(&[error_code]);
            }
        } else {
            // No main function - just return 0
            let zero = builder.ins().iconst(cranelift_codegen::ir::types::I32, 0);
            builder.ins().return_(&[zero]);
        }

        // Finalize the function
        builder.finalize();

        // Define the function in the module
        self.module.define_function(main_id, &mut ctx)?;

        Ok(())
    }
}

/// Per-function lowering context
struct FunctionLowerer<'a, 'b, 'db, M: Module> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut M,
    runtime: &'a RuntimeFunctions,
    target: &'a TargetInfo,
    db: &'db dyn Db,
    function_map: &'a HashMap<String, FuncId>,
    /// Map from variable names to their current values
    variables: HashMap<String, Variable>,
    /// Counter for generating unique variables
    var_counter: usize,
    /// Map from string literals to their offsets in the constant table
    string_literals: HashMap<String, u32>,
    /// Data ID for the string constant table
    string_table_data_id: Option<DataId>,
}

impl<'a, 'b, 'db, M: Module> FunctionLowerer<'a, 'b, 'db, M> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        builder: &'a mut FunctionBuilder<'b>,
        module: &'a mut M,
        runtime: &'a RuntimeFunctions,
        target: &'a TargetInfo,
        db: &'db dyn Db,
        function_map: &'a HashMap<String, FuncId>,
        string_literals: HashMap<String, u32>,
        string_table_data_id: Option<DataId>,
    ) -> Self {
        Self {
            builder,
            module,
            runtime,
            target,
            db,
            function_map,
            variables: HashMap::new(),
            var_counter: 0,
            string_literals,
            string_table_data_id,
        }
    }

    /// Create a new variable
    fn create_variable(&mut self) -> Variable {
        let var = Variable::from_u32(self.var_counter as u32);
        self.var_counter += 1;
        self.builder
            .declare_var(var, TributeTypes::pointer_type(self.db, self.target));
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
        let var = self.variables.get(name).ok_or_else(|| {
            CompilationErrorKind::TypeError(format!("Undefined variable: {}", name))
        })?;
        Ok(self.builder.use_var(*var))
    }

    /// Create a unit value
    fn create_unit_value(&mut self) -> CompilationResult<Value> {
        // For now, use null pointer as unit
        // TODO: Create proper unit value in runtime
        Ok(self
            .builder
            .ins()
            .iconst(TributeTypes::pointer_type(self.db, self.target), 0))
    }

    /// Lower a HIR expression
    fn lower_hir_expr(&mut self, db: &dyn Db, expr: HirExpr) -> CompilationResult<Value> {
        let expr_data = expr.expr(db);
        self.lower_expr(db, &expr_data)
    }

    /// Lower an expression
    fn lower_expr(&mut self, db: &dyn Db, expr: &Expr) -> CompilationResult<Value> {
        match expr {
            Expr::Number(n) => self.lower_number(*n),
            Expr::Bool(_) => Err(CompilationError::unsupported_feature("Bool literals")),
            Expr::Nil => Err(CompilationError::unsupported_feature("Nil literal")),
            Expr::StringInterpolation(s) => self.lower_string_interpolation(db, s),
            Expr::Variable(name) => self.use_variable(name),
            Expr::Call { func, args } => self.lower_call(db, func, args),
            Expr::Let { var, value } => self.lower_let(db, var, value),
            Expr::Block(exprs) => self.lower_block(db, exprs),
            Expr::Match { expr, cases } => self.lower_match(db, expr, cases),
            Expr::Lambda { .. } => Err(CompilationError::unsupported_feature("Lambda expressions")),
            Expr::List(_) => Err(CompilationError::unsupported_feature("List literals")),
            Expr::Tuple(_) => Err(CompilationError::unsupported_feature("Tuple literals")),
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
        _db: &dyn Db,
        s: &tribute_hir::hir::StringInterpolation,
    ) -> CompilationResult<Value> {
        if s.segments.is_empty() {
            // Simple string constant
            self.lower_string_literal(&s.leading_text)
        } else {
            // Complex interpolation - not yet implemented
            Err(CompilationError::unsupported_feature(
                "string interpolation",
            ))
        }
    }

    /// Lower a string literal
    fn lower_string_literal(&mut self, text: &str) -> CompilationResult<Value> {
        if let Some(&offset) = self.string_literals.get(text) {
            // String is in the constant table, use static string function
            if let Some(_data_id) = self.string_table_data_id {
                let offset_val = self
                    .builder
                    .ins()
                    .iconst(cranelift_codegen::ir::types::I32, offset as i64);
                let len_val = self
                    .builder
                    .ins()
                    .iconst(cranelift_codegen::ir::types::I32, text.len() as i64);

                let value_from_static_string =
                    self.import_runtime_func(self.runtime.value_from_static_string)?;
                let value = self
                    .builder
                    .ins()
                    .call(value_from_static_string, &[offset_val, len_val]);

                return Ok(self.builder.inst_results(value)[0]);
            }
        }

        // Fallback: create runtime string (for strings that are too short for static storage)
        // This automatically uses inline or heap mode based on length
        if text.len() <= 15 {
            // For short strings, we can embed them directly
            // TODO: Implement inline string creation in runtime
            let null = self
                .builder
                .ins()
                .iconst(TributeTypes::pointer_type(self.db, self.target), 0);
            let len = self.builder.ins().iconst(
                TributeTypes::size_type(self.db, self.target),
                text.len() as i64,
            );

            let value_from_string = self.import_runtime_func(self.runtime.value_from_string)?;
            let value = self.builder.ins().call(value_from_string, &[null, len]);

            Ok(self.builder.inst_results(value)[0])
        } else {
            // Long strings should go to the constant table, but if not found, use runtime allocation
            let null = self
                .builder
                .ins()
                .iconst(TributeTypes::pointer_type(self.db, self.target), 0);
            let len = self.builder.ins().iconst(
                TributeTypes::size_type(self.db, self.target),
                text.len() as i64,
            );

            let value_from_string = self.import_runtime_func(self.runtime.value_from_string)?;
            let value = self.builder.ins().call(value_from_string, &[null, len]);

            Ok(self.builder.inst_results(value)[0])
        }
    }

    /// Lower a function call
    fn lower_call(
        &mut self,
        db: &dyn Db,
        func: &Spanned<Expr>,
        args: &[Spanned<Expr>],
    ) -> CompilationResult<Value> {
        // Extract function name
        let func_name = match &func.0 {
            Expr::Variable(name) => name,
            _ => {
                return Err(CompilationError::unsupported_feature(
                    "calling via expression",
                ));
            }
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
                let func_id = self
                    .function_map
                    .get(func_name)
                    .ok_or_else(|| CompilationError::function_not_found(func_name.clone()))?;

                let func_ref = self.import_user_func(*func_id)?;
                let result = self.builder.ins().call(func_ref, &arg_values);
                Ok(self.builder.inst_results(result)[0])
            }
        }
    }

    /// Lower a binary operation
    fn lower_binary_op(
        &mut self,
        runtime_func: FuncId,
        args: &[Value],
    ) -> CompilationResult<Value> {
        if args.len() != 2 {
            return Err(CompilationError::type_error(format!(
                "Binary operation requires 2 arguments, got {}",
                args.len()
            )));
        }

        let func_ref = self.import_runtime_func(runtime_func)?;
        let result = self.builder.ins().call(func_ref, &[args[0], args[1]]);
        Ok(self.builder.inst_results(result)[0])
    }

    /// Lower print_line built-in
    fn lower_print_line(&mut self, args: &[Value]) -> CompilationResult<Value> {
        if args.len() != 1 {
            return Err(CompilationError::type_error(format!(
                "print_line requires 1 argument, got {}",
                args.len()
            )));
        }

        let func_ref = self.import_runtime_func(self.runtime.builtin_print_line)?;
        self.builder.ins().call(func_ref, &[args[0]]);

        // Return unit
        self.create_unit_value()
    }

    /// Lower input_line built-in
    fn lower_input_line(&mut self, args: &[Value]) -> CompilationResult<Value> {
        if !args.is_empty() {
            return Err(CompilationError::type_error(format!(
                "input_line requires 0 arguments, got {}",
                args.len()
            )));
        }

        let func_ref = self.import_runtime_func(self.runtime.builtin_input_line)?;
        let result = self.builder.ins().call(func_ref, &[]);
        Ok(self.builder.inst_results(result)[0])
    }

    /// Lower a let binding
    fn lower_let(
        &mut self,
        db: &dyn Db,
        var: &str,
        value: &Spanned<Expr>,
    ) -> CompilationResult<Value> {
        let val = self.lower_expr(db, &value.0)?;
        self.define_variable(var, val);
        Ok(val)
    }

    /// Lower a block expression
    fn lower_block(&mut self, db: &dyn Db, exprs: &[Spanned<Expr>]) -> CompilationResult<Value> {
        let mut last_value = None;

        for expr in exprs {
            last_value = Some(self.lower_expr(db, &expr.0)?);
        }

        match last_value {
            Some(val) => Ok(val),
            None => self.create_unit_value(),
        }
    }

    /// Lower a match expression
    /// Simplified implementation for Phase 3 - basic pattern matching only
    fn lower_match(
        &mut self,
        db: &dyn Db,
        expr: &Spanned<Expr>,
        cases: &[MatchCase],
    ) -> CompilationResult<Value> {
        if cases.is_empty() {
            return Err(CompilationError::type_error(
                "Match expression must have at least one case".to_string(),
            ));
        }

        // Check for guards (not yet supported in codegen)
        for case in cases {
            if case.guard.is_some() {
                return Err(CompilationError::unsupported_feature("pattern guards"));
            }
        }

        // Evaluate the match expression value
        let match_value = self.lower_expr(db, &expr.0)?;

        // Create a variable to store the result
        let result_var = self.create_variable();

        // Create the end block where all cases will converge
        let end_block = self.builder.create_block();

        // Handle the very simple case of just one case (common)
        if cases.len() == 1 {
            let case = &cases[0];

            // For single case, just check the pattern and execute
            match &case.pattern {
                Pattern::Wildcard | Pattern::Variable(_) => {
                    // Always matches
                    self.bind_pattern_variables(&case.pattern, match_value)?;
                    let case_result = self.lower_expr(db, &case.body.0)?;
                    self.builder.def_var(result_var, case_result);
                    self.builder.ins().jump(end_block, &[]);
                }
                Pattern::Literal(literal) => {
                    // Test and branch
                    let case_body_block = self.builder.create_block();
                    let fallback_block = self.builder.create_block();

                    let condition = self.test_literal_pattern(match_value, literal)?;
                    self.builder
                        .ins()
                        .brif(condition, case_body_block, &[], fallback_block, &[]);

                    // Case body
                    self.builder.switch_to_block(case_body_block);
                    self.bind_pattern_variables(&case.pattern, match_value)?;
                    let case_result = self.lower_expr(db, &case.body.0)?;
                    self.builder.def_var(result_var, case_result);
                    self.builder.ins().jump(end_block, &[]);
                    self.builder.seal_block(case_body_block);

                    // Fallback (no match)
                    self.builder.switch_to_block(fallback_block);
                    let unit_val = self.create_unit_value()?;
                    self.builder.def_var(result_var, unit_val);
                    self.builder.ins().jump(end_block, &[]);
                    self.builder.seal_block(fallback_block);
                }
                _ => {
                    return Err(CompilationError::unsupported_feature("complex patterns"));
                }
            }
        } else {
            // Handle multiple cases with if-else chain
            let mut blocks = Vec::new();

            // Create all blocks first
            for _ in 0..cases.len() {
                blocks.push((
                    self.builder.create_block(), // test block
                    self.builder.create_block(), // body block
                ));
            }
            let fallback_block = self.builder.create_block();

            // Jump to the first test block
            self.builder.ins().jump(blocks[0].0, &[]);

            // Generate each case
            for (i, case) in cases.iter().enumerate() {
                let (test_block, body_block) = blocks[i];
                let next_test_block = if i + 1 < blocks.len() {
                    blocks[i + 1].0
                } else {
                    fallback_block
                };

                // Generate test
                self.builder.switch_to_block(test_block);

                match &case.pattern {
                    Pattern::Wildcard | Pattern::Variable(_) => {
                        // Always match
                        self.builder.ins().jump(body_block, &[]);
                    }
                    Pattern::Literal(literal) => {
                        let condition = self.test_literal_pattern(match_value, literal)?;
                        self.builder
                            .ins()
                            .brif(condition, body_block, &[], next_test_block, &[]);
                    }
                    _ => {
                        // Unsupported pattern - skip to next
                        self.builder.ins().jump(next_test_block, &[]);
                    }
                }
                self.builder.seal_block(test_block);

                // Generate body
                self.builder.switch_to_block(body_block);
                self.bind_pattern_variables(&case.pattern, match_value)?;
                let case_result = self.lower_expr(db, &case.body.0)?;
                self.builder.def_var(result_var, case_result);
                self.builder.ins().jump(end_block, &[]);
                self.builder.seal_block(body_block);
            }

            // Generate fallback (no match found)
            self.builder.switch_to_block(fallback_block);
            let unit_val = self.create_unit_value()?;
            self.builder.def_var(result_var, unit_val);
            self.builder.ins().jump(end_block, &[]);
            self.builder.seal_block(fallback_block);
        }

        // Switch to end block and return result
        self.builder.switch_to_block(end_block);
        self.builder.seal_block(end_block);

        Ok(self.builder.use_var(result_var))
    }

    /// Test a literal pattern for matching (returns boolean)
    fn test_literal_pattern(
        &mut self,
        match_value: Value,
        literal: &Literal,
    ) -> CompilationResult<Value> {
        match literal {
            Literal::Number(n) => {
                let literal_value = self.lower_number(*n)?;
                let equals_func = self.import_runtime_func(self.runtime.value_equals)?;
                let comparison = self
                    .builder
                    .ins()
                    .call(equals_func, &[match_value, literal_value]);
                Ok(self.builder.inst_results(comparison)[0])
            }
            Literal::Bool(_) => Err(CompilationError::unsupported_feature("Bool patterns")),
            Literal::Nil => Err(CompilationError::unsupported_feature("Nil patterns")),
            Literal::StringInterpolation(s) => {
                if s.segments.is_empty() {
                    let literal_value = self.lower_string_literal(&s.leading_text)?;
                    let equals_func = self.import_runtime_func(self.runtime.value_equals)?;
                    let comparison = self
                        .builder
                        .ins()
                        .call(equals_func, &[match_value, literal_value]);
                    Ok(self.builder.inst_results(comparison)[0])
                } else {
                    Err(CompilationError::unsupported_feature(
                        "string interpolation in patterns",
                    ))
                }
            }
        }
    }

    /// Bind variables from pattern matching
    fn bind_pattern_variables(&mut self, pattern: &Pattern, value: Value) -> CompilationResult<()> {
        match pattern {
            Pattern::Variable(name) => {
                // Bind the variable to the matched value
                self.define_variable(name, value);
                Ok(())
            }
            Pattern::Wildcard | Pattern::Literal(_) => {
                // No variables to bind
                Ok(())
            }
            Pattern::List(_) | Pattern::Rest(_) | Pattern::Constructor { .. } => {
                // TODO: Implement in future iterations
                Err(CompilationError::unsupported_feature(
                    "complex pattern variable binding",
                ))
            }
        }
    }

    /// Import a runtime function
    fn import_runtime_func(&mut self, func_id: FuncId) -> CompilationResult<FuncRef> {
        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
        Ok(func_ref)
    }

    /// Import a user-defined function
    fn import_user_func(&mut self, func_id: FuncId) -> CompilationResult<FuncRef> {
        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
        Ok(func_ref)
    }
}
