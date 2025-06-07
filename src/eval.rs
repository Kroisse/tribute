use crate::builtins;
use tribute_ast::{ast::Identifier, Spanned};
use tribute_hir::hir::{Expr, HirExpr, HirFunction, HirProgram, Literal, Pattern};
use std::collections::HashMap;

type Error = Box<dyn std::error::Error + 'static>;

pub type BuiltinFn = for<'a> fn(&'a [Value]) -> Result<Value, Error>;

#[derive(Clone, Debug)]
pub enum Value {
    Unit,
    Number(i64),
    String(String),
    List(Vec<Value>),
    Fn(Identifier, Vec<Identifier>, Vec<Spanned<tribute_ast::ast::Expr>>),
    BuiltinFn(&'static str, BuiltinFn),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Unit => f.write_str("()"),
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::List(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                f.write_str("]")
            },
            Value::Fn(name, _, _) => write!(f, "<fn '{}'>", name),
            Value::BuiltinFn(name, _) => write!(f, "<builtin fn '{}'>", name),
        }
    }
}

pub struct Environment<'parent> {
    parent: Option<&'parent Self>,
    bindings: HashMap<Identifier, Value>,
}

impl Environment<'_> {
    pub fn toplevel() -> Self {
        Environment {
            parent: None,
            bindings: builtins::BUILTINS.clone(),
        }
    }

    pub fn lookup(&self, ident: &Identifier) -> Result<&Value, Error> {
        for (id, value) in &self.bindings {
            if id == ident {
                return Ok(value);
            }
        }
        if let Some(parent) = &self.parent {
            parent.lookup(ident)
        } else {
            Err(format!("identifier not found: {}", ident).into())
        }
    }

    pub fn bind(&mut self, ident: Identifier, value: Value) {
        self.bindings.insert(ident, value);
    }

    pub fn child(&self, bindings: impl IntoIterator<Item = (Identifier, Value)>) -> Environment<'_> {
        Environment {
            parent: Some(self),
            bindings: bindings.into_iter().collect(),
        }
    }
}

// Context for HIR evaluation with access to the program
pub struct HirEvalContext<'db> {
    pub db: &'db dyn salsa::Database,
    pub program: HirProgram<'db>,
}

// Legacy AST-based evaluation function (DEPRECATED)
// This function violates the HIR-first evaluation principle and should not be used.
// Use eval_with_hir() or eval_hir_expr() instead.
#[deprecated(since = "0.1.0", note = "Use eval_with_hir() for HIR-based evaluation instead")]
pub fn eval_expr<'db>(db: &'db dyn salsa::Database, env: &mut Environment<'_>, expr: &tribute_ast::ast::Expr) -> Result<Value, Error> {
    // For compatibility with existing AST-based code, we need to parse and lower to HIR
    // This is a transitional function and should be phased out in favor of direct HIR evaluation
    use tribute_ast::ast::Expr;
    
    match expr {
        // Handle function definitions by storing them in the environment
        Expr::List(exprs) if !exprs.is_empty() => {
            if let (Expr::Identifier(fn_name), _) = &exprs[0] {
                if fn_name == "fn" && exprs.len() >= 3 {
                    // This is a function definition - store it in the environment
                    if let (Expr::List(sig_parts), _) = &exprs[1] {
                        if !sig_parts.is_empty() {
                            if let (Expr::Identifier(name), _) = &sig_parts[0] {
                                let param_names: Result<Vec<String>, Error> = sig_parts
                                    .iter()
                                    .skip(1)
                                    .map(|(expr, _)| match expr {
                                        Expr::Identifier(id) => Ok(id.clone()),
                                        _ => Err("function parameter must be an identifier".into()),
                                    })
                                    .collect();
                                
                                let param_names = param_names?;
                                let body_exprs = exprs[2..].to_vec();
                                let function_value = Value::Fn(name.clone(), param_names, body_exprs);
                                env.bind(name.clone(), function_value);
                                return Ok(Value::Unit);
                            }
                        }
                    }
                }
                
                // Handle let bindings
                if fn_name == "let" && exprs.len() == 3 {
                    if let (Expr::Identifier(ident), _) = &exprs[1] {
                        let value = eval_expr(db, env, &exprs[2].0)?;
                        env.bind(ident.clone(), value.clone());
                        return Ok(value);
                    }
                }
            }
        }
        _ => {}
    }
    
    // For other expressions, try direct evaluation based on expression type
    match expr {
        Expr::Number(n) => Ok(Value::Number(*n)),
        Expr::String(s) => Ok(Value::String(s.clone())),
        Expr::Identifier(id) => env.lookup(id).cloned(),
        Expr::List(exprs) if !exprs.is_empty() => {
            // Function call or special form
            let func_name = match &exprs[0].0 {
                Expr::Identifier(name) => name,
                _ => return Err("First element of list must be an identifier".into()),
            };
            
            // Handle special forms first
            match func_name.as_str() {
                "match" => {
                    if exprs.len() < 3 {
                        return Err("match requires at least expression and one case".into());
                    }
                    
                    let value = eval_expr(db, env, &exprs[1].0)?;
                    
                    // Try each case
                    for case_expr in &exprs[2..] {
                        if let Expr::List(case_parts) = &case_expr.0 {
                            if case_parts.len() == 3 {
                                if let Expr::Identifier(case_kw) = &case_parts[0].0 {
                                    if case_kw == "case" {
                                        // Check if pattern matches
                                        if let Ok(pattern_value) = eval_expr(db, env, &case_parts[1].0) {
                                            if values_match(&value, &pattern_value) {
                                                return eval_expr(db, env, &case_parts[2].0);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return Err("no matching case found".into());
                }
                _ => {}
            }
            
            // Look up the function in the environment
            let func_value = env.lookup(func_name)?.clone();
            
            match func_value {
                Value::BuiltinFn(_, f) => {
                    // Evaluate arguments for builtin functions
                    let arg_values: Result<Vec<Value>, _> = exprs[1..]
                        .iter()
                        .map(|(arg_expr, _)| eval_expr(db, env, arg_expr))
                        .collect();
                    f(&arg_values?)
                }
                Value::Fn(name, params, body) => {
                    // User-defined function call
                    let arg_values: Result<Vec<Value>, _> = exprs[1..]
                        .iter()
                        .map(|(arg_expr, _)| eval_expr(db, env, arg_expr))
                        .collect();
                    let arg_values = arg_values?;
                    
                    if params.len() != arg_values.len() {
                        return Err(format!(
                            "function {} expects {} arguments, got {}",
                            name,
                            params.len(),
                            arg_values.len()
                        ).into());
                    }
                    
                    // Create bindings for function parameters
                    let bindings: Vec<(String, Value)> = params
                        .iter()
                        .zip(arg_values)
                        .map(|(param, value)| (param.clone(), value))
                        .collect();
                    
                    let mut child_env = env.child(bindings);
                    
                    // Evaluate function body
                    let mut result = Value::Unit;
                    for (body_expr, _) in body {
                        result = eval_expr(db, &mut child_env, &body_expr)?;
                    }
                    Ok(result)
                }
                _ => Err(format!("'{}' is not a function", func_name).into()),
            }
        }
        _ => Err("Unsupported expression type".into()),
    }
}


pub fn eval_hir_program<'db>(
    db: &'db dyn salsa::Database,
    env: &mut Environment<'_>,
    program: HirProgram<'db>,
) -> Result<Value, Error> {
    let functions = program.functions(db);

    // Bind all functions to environment first
    for (name, func) in &functions {
        let func_value = Value::Fn(
            name.clone(),
            func.params(db).to_vec(),
            vec![], // HIR functions store their body differently
        );
        env.bind(name.clone(), func_value);
    }

    let context = HirEvalContext { db, program };

    // If there's a main function, call it
    if let Some(main_name) = program.main(db) {
        if let Some(main_func) = functions.get(&main_name) {
            eval_hir_function_body_with_context(&context, *main_func, vec![])
        } else {
            Err(format!("main function '{}' not found", main_name).into())
        }
    } else {
        // No main function, just return unit
        Ok(Value::Unit)
    }
}

pub fn eval_hir_function_body<'db>(
    db: &'db dyn salsa::Database,
    program: HirProgram<'db>,
    env: &mut Environment<'_>,
    func: HirFunction<'db>,
    args: Vec<Value>,
) -> Result<Value, Error> {
    let params = func.params(db);
    if args.len() != params.len() {
        return Err(format!(
            "function {} expects {} arguments, got {}",
            func.name(db),
            params.len(),
            args.len()
        )
        .into());
    }

    // Create bindings for function parameters
    let bindings: Vec<(std::string::String, Value)> = params
        .iter()
        .zip(args)
        .map(|(param, value)| (param.clone(), value))
        .collect();

    let mut child_env = env.child(bindings);

    // Create context for evaluation
    let context = HirEvalContext { db, program };

    // Evaluate all expressions in the function body
    let body = func.body(db);
    let mut result = Value::Unit;
    for expr in body {
        result = eval_hir_expr(&context, &mut child_env, expr)?;
    }

    Ok(result)
}

pub fn eval_hir_function_body_with_context<'db>(
    context: &HirEvalContext<'db>,
    func: HirFunction<'db>,
    args: Vec<Value>,
) -> Result<Value, Error> {
    let params = func.params(context.db);
    if args.len() != params.len() {
        return Err(format!(
            "function {} expects {} arguments, got {}",
            func.name(context.db),
            params.len(),
            args.len()
        )
        .into());
    }

    // Create bindings for function parameters
    let bindings: Vec<(Identifier, Value)> = params
        .iter()
        .zip(args)
        .map(|(param, value)| (param.clone(), value))
        .collect();

    // Create new environment with all functions bound
    let mut env = Environment::toplevel();

    // Bind all functions in the program
    let functions = context.program.functions(context.db);
    for (name, hir_func) in &functions {
        let func_value = Value::Fn(
            name.clone(),
            hir_func.params(context.db),
            vec![], // HIR doesn't store the original AST body
        );
        env.bind(name.clone(), func_value);
    }

    let mut child_env = env.child(bindings);

    // Evaluate all expressions in the function body
    let body = func.body(context.db);
    let mut result = Value::Unit;
    for expr in body {
        result = eval_hir_expr(context, &mut child_env, expr)?;
    }

    Ok(result)
}

pub fn eval_hir_expr<'db>(
    context: &HirEvalContext<'db>,
    env: &mut Environment<'_>,
    expr: HirExpr<'db>,
) -> Result<Value, Error> {
    eval_spanned_expr(
        context,
        env,
        (expr.expr(context.db).clone(), expr.span(context.db)),
    )
}

fn eval_spanned_expr<'db>(
    context: &HirEvalContext<'db>,
    env: &mut Environment<'_>,
    (expr, _span): Spanned<Expr>,
) -> Result<Value, Error> {
    use Expr::*;

    match expr {
        Number(n) => Ok(Value::Number(n)),
        String(s) => Ok(Value::String(s)),
        Variable(name) => env.lookup(&name).cloned(),
        Call { func, args } => {
            let func_expr = eval_spanned_expr(context, env, *func)?;
            match func_expr {
                Value::Fn(name, params, _) => {
                    let arg_values: Result<Vec<Value>, Error> = args
                        .into_iter()
                        .map(|arg| eval_spanned_expr(context, env, arg))
                        .collect();
                    let arg_values = arg_values?;

                    // Create child environment with parameter bindings
                    if params.len() != arg_values.len() {
                        return Err(format!(
                            "function {} expects {} arguments, got {}",
                            name,
                            params.len(),
                            arg_values.len()
                        )
                        .into());
                    }

                    let bindings: Vec<(Identifier, Value)> =
                        params.into_iter().zip(arg_values.iter().cloned()).collect();

                    let _child_env = env.child(bindings);

                    // Find the function in the program and evaluate its body
                    let functions = context.program.functions(context.db);
                    if let Some((_, func)) = functions.iter().find(|(fname, _)| **fname == name) {
                        eval_hir_function_body_with_context(context, *func, arg_values)
                    } else {
                        Err(format!("Function '{}' not found", name).into())
                    }
                }
                Value::BuiltinFn(_, f) => {
                    let arg_values: Result<Vec<Value>, Error> = args
                        .into_iter()
                        .map(|arg| eval_spanned_expr(context, env, arg))
                        .collect();
                    f(&arg_values?)
                }
                _ => Err(format!("not a function: {:?}", func_expr).into()),
            }
        }
        Let { var, value } => {
            let val = eval_spanned_expr(context, env, *value)?;
            env.bind(var, val.clone());
            Ok(val)
        }
        Match { expr, cases } => {
            let value = eval_spanned_expr(context, env, *expr)?;
            for case in cases {
                if let Some(bindings) = match_pattern(&value, &case.pattern) {
                    let mut child_env = env.child(bindings);
                    return eval_spanned_expr(context, &mut child_env, case.body);
                }
            }
            Err("no matching case found".into())
        }
        Block(exprs) => {
            let mut result = Value::Unit;
            for expr in exprs {
                result = eval_spanned_expr(context, env, expr)?;
            }
            Ok(result)
        }
    }
}


fn match_pattern(value: &Value, pattern: &Pattern) -> Option<Vec<(std::string::String, Value)>> {
    match pattern {
        Pattern::Literal(lit) => {
            // Compare literal values
            match (lit, value) {
                (Literal::Number(a), Value::Number(b)) if a == b => Some(vec![]),
                (Literal::String(a), Value::String(b)) if a == b => Some(vec![]),
                _ => None,
            }
        }
        Pattern::Variable(name) => {
            // Variable patterns always match and bind
            Some(vec![(name.clone(), value.clone())])
        }
        Pattern::Wildcard => {
            // Wildcard always matches without binding
            Some(vec![])
        }
        Pattern::List(patterns) => {
            // List pattern matching
            match value {
                Value::List(values) => {
                    // Check if we have a rest pattern
                    if let Some(last_pattern) = patterns.last() {
                        if matches!(last_pattern, Pattern::Rest(_)) {
                            return match_list_with_rest(values, patterns);
                        }
                    }

                    // Exact length matching
                    if values.len() != patterns.len() {
                        return None;
                    }

                    let mut bindings = Vec::new();
                    for (value, pattern) in values.iter().zip(patterns.iter()) {
                        if let Some(mut pattern_bindings) = match_pattern(value, pattern) {
                            bindings.append(&mut pattern_bindings);
                        } else {
                            return None;
                        }
                    }
                    Some(bindings)
                }
                _ => None,
            }
        }
        Pattern::Rest(_) => {
            // Rest patterns should only appear in list contexts
            None
        }
    }
}

// Helper function to check if two values match (for simple pattern matching)
fn values_match(value1: &Value, value2: &Value) -> bool {
    match (value1, value2) {
        (Value::Number(a), Value::Number(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        _ => false,
    }
}

fn match_list_with_rest(
    values: &[Value],
    patterns: &[Pattern],
) -> Option<Vec<(std::string::String, Value)>> {
    if patterns.is_empty() {
        return None;
    }

    let (rest_pattern, head_patterns) = patterns.split_last().unwrap();

    if let Pattern::Rest(rest_name) = rest_pattern {
        // Must have at least as many values as head patterns
        if values.len() < head_patterns.len() {
            return None;
        }

        let mut bindings = Vec::new();

        // Match head patterns
        for (value, pattern) in values
            .iter()
            .take(head_patterns.len())
            .zip(head_patterns.iter())
        {
            if let Some(mut pattern_bindings) = match_pattern(value, pattern) {
                bindings.append(&mut pattern_bindings);
            } else {
                return None;
            }
        }

        // Bind rest values
        let rest_values = values[head_patterns.len()..].to_vec();
        bindings.push((rest_name.clone(), Value::List(rest_values)));

        Some(bindings)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tribute_ast::TributeDatabaseImpl;
    use salsa::Database;

    #[test]
    fn test_hir_arithmetic() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test simple arithmetic expression wrapped in a main function
            let source = r#"(fn (main) (+ 1 2))"#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Number(3)) => {}
                Ok(other) => panic!("Expected Number(3), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_print_line() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test print_line builtin wrapped in a main function
            let source = r#"(fn (main) (print_line "Hello HIR"))"#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Unit) => {}
                Ok(other) => panic!("Expected Unit, got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_nested_arithmetic() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test nested arithmetic wrapped in a main function
            let source = r#"(fn (main) (+ (* 2 3) (/ 8 2)))"#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Number(10)) => {}
                Ok(other) => panic!("Expected Number(10), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_let_binding() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test let binding without body
            let source = r#"(fn (main) (let x 42) x)"#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Number(42)) => {}
                Ok(other) => panic!("Expected Number(42), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_pattern_matching() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test pattern matching
            let source = r#"
                (fn (test_number n)
                  (match n
                    (case 0 "zero")
                    (case 1 "one")
                    (case _ "other")))
                (fn (main) (test_number 0))
            "#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::String(s)) if s == "zero" => {}
                Ok(other) => panic!("Expected String(\"zero\"), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_user_defined_function() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test user-defined function call
            let source = r#"
                (fn (add x y) (+ x y))
                (fn (main) (add 10 20))
            "#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Number(30)) => {}
                Ok(other) => panic!("Expected Number(30), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_functions_calling_each_other() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test functions calling each other
            let source = r#"
                (fn (double x) (* x 2))
                (fn (add_and_double x y) (double (+ x y)))
                (fn (main) (add_and_double 5 10))
            "#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Number(30)) => {}
                Ok(other) => panic!("Expected Number(30), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_recursive_function() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test recursive function (factorial)
            let source = r#"
                (fn (factorial n)
                  (match n
                    (case 0 1)
                    (case _ (* n (factorial (- n 1))))))
                (fn (main) (factorial 5))
            "#;
            match crate::eval_with_hir(db, "test.trb", source) {
                Ok(Value::Number(120)) => {}
                Ok(other) => panic!("Expected Number(120), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }
}
