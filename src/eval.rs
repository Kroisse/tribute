//! DEPRECATED: legacy HIR evaluator.
//!
//! This module is kept for backwards compatibility and is not compiled by
//! default. Enable the Cargo feature `legacy-eval` to build and use it.

use crate::builtins;
use std::collections::HashMap;
use tribute_ast::{Spanned, ast::Identifier};
use tribute_hir::hir::{Expr, HirExpr, HirFunction, HirProgram, Literal, Pattern};

type Error = Box<dyn std::error::Error + 'static>;

pub type BuiltinFn = for<'a> fn(&'a [Value]) -> Result<Value, Error>;

#[derive(Clone, Debug)]
pub enum Value {
    Unit,
    Bool(bool),
    /// Natural number (non-negative): 0, 42, 0b1010
    Nat(u64),
    /// Integer (signed): +1, -1
    Int(i64),
    /// Float: 1.0, -3.14
    Float(f64),
    /// Rune (Unicode codepoint): ?a, ?\n, ?\x41
    Rune(char),
    String(String),
    /// Bytes: b"hello", rb"raw"
    Bytes(Vec<u8>),
    List(Vec<Value>),
    /// Tuple value: #(a, b, c)
    Tuple(Vec<Value>),
    /// Record value: type name, fields (name -> value)
    Record(Identifier, HashMap<Identifier, Value>),
    Fn(
        Identifier,
        Vec<Identifier>,
        Vec<Spanned<tribute_ast::ast::Expr>>,
    ),
    /// Lambda closure: params, body, captured environment
    Lambda(Vec<Identifier>, Box<Spanned<Expr>>),
    BuiltinFn(&'static str, BuiltinFn),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Unit => f.write_str("Nil"),
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
            Value::Nat(n) => write!(f, "{}", n),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Rune(ch) => {
                // Display rune with ? prefix, using escape for special chars
                match ch {
                    '\n' => write!(f, "?\\n"),
                    '\r' => write!(f, "?\\r"),
                    '\t' => write!(f, "?\\t"),
                    '\0' => write!(f, "?\\0"),
                    '\\' => write!(f, "?\\\\"),
                    c if c.is_ascii_graphic() || *c == ' ' => write!(f, "?{}", c),
                    c => write!(f, "?\\u{:04X}", *c as u32),
                }
            }
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Bytes(bytes) => {
                // Display bytes as b"..." with hex escapes for non-printable bytes
                f.write_str("b\"")?;
                for &byte in bytes {
                    if byte.is_ascii_graphic() || byte == b' ' {
                        write!(f, "{}", byte as char)?;
                    } else {
                        write!(f, "\\x{:02x}", byte)?;
                    }
                }
                f.write_str("\"")
            }
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
            Value::Tuple(items) => {
                f.write_str("#(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                f.write_str(")")
            }
            Value::Record(type_name, fields) => {
                write!(f, "{} {{ ", type_name)?;
                for (i, (name, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}: {}", name, value)?;
                }
                f.write_str(" }")
            }
            Value::Fn(name, _, _) => write!(f, "<fn '{}'>", name),
            Value::Lambda(params, _) => write!(f, "<lambda({})>", params.join(", ")),
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
        Nat(n) => Ok(Value::Nat(n)),
        Int(n) => Ok(Value::Int(n)),
        Float(n) => Ok(Value::Float(n)),
        Rune(ch) => Ok(Value::Rune(ch)),
        Bool(b) => Ok(Value::Bool(b)),
        Nil => Ok(Value::Unit),
        StringLit(s) => Ok(Value::String(s)),
        BytesLit(b) => Ok(Value::Bytes(b)),
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
                Value::Lambda(params, body) => {
                    let arg_values: Result<Vec<Value>, Error> = args
                        .into_iter()
                        .map(|arg| eval_spanned_expr(context, env, arg))
                        .collect();
                    let arg_values = arg_values?;

                    if params.len() != arg_values.len() {
                        return Err(format!(
                            "lambda expects {} arguments, got {}",
                            params.len(),
                            arg_values.len()
                        )
                        .into());
                    }

                    let bindings: Vec<(Identifier, Value)> =
                        params.into_iter().zip(arg_values).collect();

                    let mut child_env = env.child(bindings);
                    eval_spanned_expr(context, &mut child_env, *body)
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
        Let { pattern, value } => {
            let val = eval_spanned_expr(context, env, *value)?;
            if let Some(bindings) = match_pattern(&val, &pattern) {
                for (name, bound_value) in bindings {
                    env.bind(name, bound_value);
                }
                Ok(val)
            } else {
                Err("pattern match failed in let binding".to_string().into())
            }
        }
        Match { expr, cases } => {
            let value = eval_spanned_expr(context, env, *expr)?;
            for case in cases {
                if let Some(bindings) = match_pattern(&value, &case.pattern) {
                    let mut child_env = env.child(bindings);
                    // Check guard condition if present
                    if let Some(guard) = case.guard {
                        let guard_result = eval_spanned_expr(context, &mut child_env, guard)?;
                        match guard_result {
                            Value::Bool(true) => {}         // Guard passed
                            Value::Bool(false) => continue, // Guard failed, try next case
                            _ => return Err("guard expression must evaluate to Bool".into()),
                        }
                    }
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
        List(elements) => {
            let values: Result<Vec<Value>, Error> = elements
                .into_iter()
                .map(|elem| eval_spanned_expr(context, env, elem))
                .collect();
            Ok(Value::List(values?))
        }
        Tuple(first, rest) => {
            let first_value = eval_spanned_expr(context, env, *first)?;
            let rest_values: Result<Vec<Value>, Error> = rest
                .into_iter()
                .map(|elem| eval_spanned_expr(context, env, elem))
                .collect();
            let mut values = vec![first_value];
            values.extend(rest_values?);
            Ok(Value::Tuple(values))
        }
        Lambda { params, body } => {
            // Create a lambda value with the body
            Ok(Value::Lambda(params, body))
        }
        Record { type_name, fields } => {
            use tribute_hir::hir::RecordField;
            let mut field_map = HashMap::new();
            for field in fields {
                match field {
                    RecordField::Spread(spread_expr) => {
                        // Spread: ..expr - merge fields from another record
                        let spread_value = eval_spanned_expr(context, env, spread_expr)?;
                        match spread_value {
                            Value::Record(_, spread_fields) => {
                                for (name, value) in spread_fields {
                                    field_map.insert(name, value);
                                }
                            }
                            _ => return Err("spread must be a record".into()),
                        }
                    }
                    RecordField::Field { name, value } => {
                        let field_value = eval_spanned_expr(context, env, value)?;
                        field_map.insert(name, field_value);
                    }
                    RecordField::Shorthand(name) => {
                        // Shorthand: name - get value from variable with same name
                        let value = env.lookup(&name).cloned()?;
                        field_map.insert(name, value);
                    }
                }
            }
            Ok(Value::Record(type_name, field_map))
        }
    }
}

fn match_pattern(value: &Value, pattern: &Pattern) -> Option<Vec<(std::string::String, Value)>> {
    match pattern {
        Pattern::Literal(lit) => {
            // Compare literal values
            match (lit, value) {
                (Literal::Nat(a), Value::Nat(b)) if a == b => Some(vec![]),
                (Literal::Int(a), Value::Int(b)) if a == b => Some(vec![]),
                // Allow Nat patterns to match Int values when the value is non-negative
                (Literal::Nat(a), Value::Int(b)) if *b >= 0 && *a == (*b as u64) => Some(vec![]),
                // Allow Float pattern matching with tolerance for equality
                (Literal::Float(a), Value::Float(b)) if (a - b).abs() < f64::EPSILON => {
                    Some(vec![])
                }
                (Literal::Rune(a), Value::Rune(b)) if a == b => Some(vec![]),
                (Literal::Bool(a), Value::Bool(b)) if a == b => Some(vec![]),
                (Literal::Nil, Value::Unit) => Some(vec![]),
                (Literal::StringPat(pattern), Value::String(s)) if pattern == s => Some(vec![]),
                (Literal::BytesPat(pattern), Value::Bytes(b)) if pattern == b => Some(vec![]),
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
        Pattern::List { elements, rest } => {
            // List pattern matching
            match value {
                Value::List(values) => {
                    // Handle rest pattern
                    match rest {
                        Some(rest_binding) => {
                            // [a, b, ..tail] or [a, b, ..]
                            if values.len() < elements.len() {
                                return None;
                            }
                            let mut bindings = Vec::new();
                            // Match element patterns
                            for (value, pattern) in
                                values.iter().take(elements.len()).zip(elements.iter())
                            {
                                if let Some(mut pattern_bindings) = match_pattern(value, pattern) {
                                    bindings.append(&mut pattern_bindings);
                                } else {
                                    return None;
                                }
                            }
                            // Bind rest if named
                            if let Some(name) = rest_binding {
                                let rest_values = values[elements.len()..].to_vec();
                                bindings.push((name.clone(), Value::List(rest_values)));
                            }
                            Some(bindings)
                        }
                        None => {
                            // Exact length matching: [a, b, c]
                            if values.len() != elements.len() {
                                return None;
                            }
                            let mut bindings = Vec::new();
                            for (value, pattern) in values.iter().zip(elements.iter()) {
                                if let Some(mut pattern_bindings) = match_pattern(value, pattern) {
                                    bindings.append(&mut pattern_bindings);
                                } else {
                                    return None;
                                }
                            }
                            Some(bindings)
                        }
                    }
                }
                _ => None,
            }
        }
        Pattern::Rest(_) => {
            // Rest patterns should only appear in list contexts
            None
        }
        Pattern::Constructor { name, args } => {
            // Constructor patterns can match Record values
            match value {
                Value::Record(type_name, fields) => {
                    // Type name must match
                    if name != type_name {
                        return None;
                    }
                    match args {
                        tribute_hir::hir::ConstructorArgs::None => {
                            // Match only if record has no fields
                            if fields.is_empty() {
                                Some(Vec::new())
                            } else {
                                None
                            }
                        }
                        tribute_hir::hir::ConstructorArgs::Positional(_) => {
                            // Positional args don't make sense for records
                            None
                        }
                        tribute_hir::hir::ConstructorArgs::Named {
                            fields: pattern_fields,
                            rest,
                        } => {
                            // Match each pattern field against the record field
                            let mut bindings = Vec::new();
                            for pattern_field in pattern_fields {
                                if let Some(field_value) = fields.get(&pattern_field.name) {
                                    if let Some(mut field_bindings) =
                                        match_pattern(field_value, &pattern_field.pattern)
                                    {
                                        bindings.append(&mut field_bindings);
                                    } else {
                                        return None;
                                    }
                                } else {
                                    // Field not found in record
                                    return None;
                                }
                            }
                            // If rest is false, ensure all fields are accounted for
                            if !rest {
                                let matched_fields: std::collections::HashSet<_> =
                                    pattern_fields.iter().map(|f| &f.name).collect();
                                for field_name in fields.keys() {
                                    if !matched_fields.contains(field_name) {
                                        return None;
                                    }
                                }
                            }
                            Some(bindings)
                        }
                    }
                }
                _ => None,
            }
        }
        Pattern::Tuple(first, rest) => {
            // Tuple patterns match against tuple values
            match value {
                Value::Tuple(values) => {
                    let pattern_count = 1 + rest.len();
                    if values.len() != pattern_count {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    // Match first pattern
                    if let Some(mut first_bindings) = match_pattern(&values[0], first) {
                        bindings.append(&mut first_bindings);
                    } else {
                        return None;
                    }
                    // Match rest patterns
                    for (value, pattern) in values[1..].iter().zip(rest.iter()) {
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
        Pattern::As(inner_pattern, binding) => {
            // As pattern: match inner and also bind the whole value
            if let Some(mut bindings) = match_pattern(value, inner_pattern) {
                bindings.push((binding.clone(), value.clone()));
                Some(bindings)
            } else {
                None
            }
        }
        Pattern::Handler(_) => {
            // TODO: Handler pattern matching requires effect handling infrastructure
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::Database;
    use tribute_core::TributeDatabaseImpl;

    #[test]
    fn test_hir_arithmetic() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test simple arithmetic expression wrapped in a main function
            let source = r#"fn main() { 1 + 2 }"#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Nat(3)) => {}
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
            // Note: Use {} for grouping, () is reserved for operator functions
            let source = r#"fn main() { {2 * 3} + {8 / 2} }"#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Nat(10)) => {}
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
                Ok(Value::Nat(42)) => {}
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
                  case n {
                    0 -> "zero",
                    1 -> "one",
                    _ -> "other"
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
                Ok(Value::Nat(30)) => {}
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
                Ok(Value::Nat(30)) => {}
                Ok(other) => panic!("Expected Number(30), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_recursive_function() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test recursive function (factorial)
            // Note: Result is Int because subtraction (n - 1) returns Int
            let source = r#"
                fn factorial(n) {
                  case n {
                    0 -> 1,
                    _ -> n * factorial(n - 1)
                  }
                }
                fn main() { factorial(5) }
            "#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Nat(120)) | Ok(Value::Int(120)) => {}
                Ok(other) => panic!("Expected 120, got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_lambda_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test simple lambda expression
            let source = r#"
                fn main() {
                    let add = fn(x, y) x + y
                    add(3, 4)
                }
            "#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Nat(7)) => {}
                Ok(other) => panic!("Expected Number(7), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_lambda_with_block() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test lambda with block body
            let source = r#"
                fn main() {
                    let double_plus_one = fn(x) {
                        let doubled = x * 2
                        doubled + 1
                    }
                    double_plus_one(5)
                }
            "#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Nat(11)) => {}
                Ok(other) => panic!("Expected Number(11), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_lambda_no_params() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test lambda without parameters
            let source = r#"
                fn main() {
                    let constant = fn() 42
                    constant()
                }
            "#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Nat(42)) => {}
                Ok(other) => panic!("Expected Number(42), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_rune_literals() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test simple rune literal
            let source = r#"fn main() { ?A }"#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Rune('A')) => {}
                Ok(other) => panic!("Expected Rune('A'), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_rune_escape_sequences() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test rune with escape sequence
            let source = r#"fn main() { ?\n }"#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::Rune('\n')) => {}
                Ok(other) => panic!("Expected Rune('\\n'), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }

    #[test]
    fn test_hir_rune_pattern_matching() {
        TributeDatabaseImpl::default().attach(|db| {
            // Test rune in pattern matching
            let source = r#"
                fn classify(ch) {
                    case ch {
                        ?A -> "letter A",
                        ?B -> "letter B",
                        _ -> "other"
                    }
                }
                fn main() { classify(?A) }
            "#;
            match crate::eval_str(db, "test.trb", source) {
                Ok(Value::String(s)) if s == "letter A" => {}
                Ok(other) => panic!("Expected String(\"letter A\"), got {:?}", other),
                Err(e) => panic!("HIR evaluation failed: {}", e),
            }
        });
    }
}
