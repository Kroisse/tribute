use crate::{
    builtins,
    eval::{Environment, Value},
};
use tribute_ast::{ast::Identifier, Spanned};
use tribute_hir::hir::{Expr, HirExpr, HirFunction, HirProgram, Literal, Pattern};

type Error = Box<dyn std::error::Error + 'static>;

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
        Builtin { name, args } => eval_builtin_by_name(context, env, &name, args),
        Block(exprs) => {
            let mut result = Value::Unit;
            for expr in exprs {
                result = eval_spanned_expr(context, env, expr)?;
            }
            Ok(result)
        }
    }
}

fn eval_builtin_by_name<'db>(
    context: &HirEvalContext<'db>,
    env: &mut Environment<'_>,
    name: &str,
    args: Vec<Spanned<Expr>>,
) -> Result<Value, Error> {
    let arg_values: Result<Vec<Value>, Error> = args
        .into_iter()
        .map(|arg| eval_spanned_expr(context, env, arg))
        .collect();
    let arg_values = arg_values?;

    match name {
        "+" => {
            if arg_values.len() != 2 {
                return Err("+ requires exactly 2 arguments".into());
            }
            match (&arg_values[0], &arg_values[1]) {
                (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
                _ => Err("+ requires numbers".into()),
            }
        }
        "-" => {
            if arg_values.len() != 2 {
                return Err("- requires exactly 2 arguments".into());
            }
            match (&arg_values[0], &arg_values[1]) {
                (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
                _ => Err("- requires numbers".into()),
            }
        }
        "*" => {
            if arg_values.len() != 2 {
                return Err("* requires exactly 2 arguments".into());
            }
            match (&arg_values[0], &arg_values[1]) {
                (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),
                _ => Err("* requires numbers".into()),
            }
        }
        "/" => {
            if arg_values.len() != 2 {
                return Err("/ requires exactly 2 arguments".into());
            }
            match (&arg_values[0], &arg_values[1]) {
                (Value::Number(a), Value::Number(b)) => {
                    if *b == 0 {
                        Err("division by zero".into())
                    } else {
                        Ok(Value::Number(a / b))
                    }
                }
                _ => Err("/ requires numbers".into()),
            }
        }
        "print_line" => {
            if let Some(builtin_fn) = builtins::BUILTINS.get("print_line") {
                if let Value::BuiltinFn(_, f) = builtin_fn {
                    f(&arg_values)
                } else {
                    Err("print_line is not a builtin function".into())
                }
            } else {
                Err("print_line not found".into())
            }
        }
        "input_line" => {
            if let Some(builtin_fn) = builtins::BUILTINS.get("input_line") {
                if let Value::BuiltinFn(_, f) = builtin_fn {
                    f(&arg_values)
                } else {
                    Err("input_line is not a builtin function".into())
                }
            } else {
                Err("input_line not found".into())
            }
        }
        _ => Err(format!("unknown builtin function: {}", name).into()),
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
