use crate::builtins;
use std::collections::HashMap;
use tribute_ast::{Spanned, ast::Identifier};
use tribute_hir::hir::{Expr, HirExpr, HirFunction, HirProgram, Literal, Pattern};

type Error = Box<dyn std::error::Error + 'static>;

pub type BuiltinFn = for<'a> fn(&'a [Value]) -> Result<Value, Error>;

#[derive(Clone, Debug)]
pub enum Value {
    Unit,
    Number(i64),
    String(String),
    List(Vec<Value>),
    Fn(
        Identifier,
        Vec<Identifier>,
        Vec<Spanned<tribute_ast::ast::Expr>>,
    ),
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
            }
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

    pub fn child(
        &self,
        bindings: impl IntoIterator<Item = (Identifier, Value)>,
    ) -> Environment<'_> {
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
        StringInterpolation(interp) => {
            if interp.segments.is_empty() {
                // Simple string without interpolation
                Ok(Value::String(interp.leading_text.clone()))
            } else {
                // String with interpolation
                let mut result = std::string::String::new();
                result.push_str(&interp.leading_text);
                for segment in &interp.segments {
                    // Evaluate the interpolation expression
                    let value = eval_spanned_expr(context, env, (*segment.interpolation).clone())?;
                    let value_str = value_to_string(&value)?;
                    result.push_str(&value_str);
                    result.push_str(&segment.trailing_text);
                }
                Ok(Value::String(result))
            }
        }
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

/// Convert a Value to its string representation for interpolation
fn value_to_string(value: &Value) -> Result<String, Error> {
    match value {
        Value::Unit => Ok("()".to_string()),
        Value::Number(n) => Ok(n.to_string()),
        Value::String(s) => Ok(s.clone()),
        Value::List(items) => {
            let item_strings: Result<Vec<String>, Error> =
                items.iter().map(value_to_string).collect();
            let item_strings = item_strings?;
            Ok(format!("[{}]", item_strings.join(", ")))
        }
        Value::Fn(name, _, _) => Ok(format!("<fn '{}'>", name)),
        Value::BuiltinFn(name, _) => Ok(format!("<builtin fn '{}'>", name)),
    }
}

fn match_pattern(value: &Value, pattern: &Pattern) -> Option<Vec<(std::string::String, Value)>> {
    match pattern {
        Pattern::Literal(lit) => {
            // Compare literal values
            match (lit, value) {
                (Literal::Number(a), Value::Number(b)) if a == b => Some(vec![]),
                (Literal::StringInterpolation(interp), Value::String(s)) => {
                    if interp.segments.is_empty() {
                        // Simple string pattern
                        if &interp.leading_text == s {
                            Some(vec![])
                        } else {
                            None
                        }
                    } else {
                        // For now, string interpolation in patterns is not supported
                        // This would require complex pattern matching logic
                        None
                    }
                }
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
    use salsa::Database;
    use tribute_ast::TributeDatabaseImpl;

    #[test]
    fn test_hir_arithmetic() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test simple arithmetic expression wrapped in a main function
            let source = r#"fn main() { 1 + 2 }"#;
            match crate::eval_str(db, "test.trb", source) {
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
            let source = r#"fn main() { print_line("Hello HIR") }"#;
            match crate::eval_str(db, "test.trb", source) {
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
            let source = r#"fn main() { (2 * 3) + (8 / 2) }"#;
            match crate::eval_str(db, "test.trb", source) {
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
            let source = r#"fn main() { let x = 42; x }"#;
            match crate::eval_str(db, "test.trb", source) {
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
                fn test_number(n) {
                  match n {
                    0 => "zero",
                    1 => "one",
                    _ => "other"
                  }
                }
                fn main() { test_number(0) }
            "#;
            match crate::eval_str(db, "test.trb", source) {
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
                fn add(x, y) { x + y }
                fn main() { add(10, 20) }
            "#;
            match crate::eval_str(db, "test.trb", source) {
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
                fn double(x) { x * 2 }
                fn add_and_double(x, y) { double(x + y) }
                fn main() { add_and_double(5, 10) }
            "#;
            match crate::eval_str(db, "test.trb", source) {
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
                fn factorial(n) {
                  match n {
                    0 => 1,
                    _ => n * factorial(n - 1)
                  }
                }
                fn main() { factorial(5) }
            "#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Number(120)) => {}
                Ok(other) => panic!("Expected Number(120), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }
}
