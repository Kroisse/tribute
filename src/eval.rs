use crate::ast::{Expr, Identifier, Spanned};
use std::collections::HashMap;

type Error = Box<dyn std::error::Error + 'static>;

type BuiltinFn = for<'a> fn(&'a [Value]) -> Result<Value, Error>;

#[derive(Clone)]
pub enum Value {
    Unit,
    Number(i64),
    String(String),
    Fn(Identifier, Vec<Identifier>, Vec<Spanned<Expr>>),
    BuiltinFn(&'static str, BuiltinFn),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Unit => f.write_str("()"),
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
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

    fn bind(&mut self, ident: Identifier, value: Value) {
        self.bindings.insert(ident, value);
    }

    pub fn child(&self, bindings: impl IntoIterator<Item = (Identifier, Value)>) -> Environment<'_> {
        Environment {
            parent: Some(self),
            bindings: bindings.into_iter().collect(),
        }
    }
}

pub fn eval_expr(env: &mut Environment<'_>, expr: &Expr) -> Result<Value, Error> {
    use Expr::*;
    Ok(match expr {
        Number(n) => Value::Number(*n),
        String(s) => Value::String(s.clone()),
        Identifier(ident) => env.lookup(ident)?.clone(),
        List(exprs) => {
            let Some((head, rest)) = exprs.split_first() else {
                return Ok(Value::Unit);
            };
            let head = match &head.0 {
                // TODO: this must have been checked in the compile time
                Number(_) => return Err("expected function, got number".into()),
                String(_) => return Err("expected function, got string".into()),
                Identifier(ident) => match ident.as_str() {
                    "let" => {
                        return handle_let(env, rest);
                    }
                    "fn" => {
                        return handle_fn(env, rest);
                    }
                    _ => env.lookup(ident)?.clone(),
                },
                _ => eval_expr(env, &head.0)?,
            };
            match head {
                Value::Fn(_, params, body) => {
                    let bindings = params
                        .iter()
                        .zip(rest.iter())
                        .map(|(param, arg)| Ok((param.clone(), eval_expr(env, &arg.0)?)))
                        .collect::<Result<Vec<_>, Error>>()?;
                    let mut child_env = env.child(bindings);
                    let mut result = Value::Unit;
                    for expr in body {
                        result = eval_expr(&mut child_env, &expr.0)?;
                    }
                    result
                }
                Value::BuiltinFn(_, f) => {
                    let args = rest
                        .iter()
                        .map(|expr| eval_expr(env, &expr.0))
                        .collect::<Result<Vec<_>, _>>()?;
                    f(&args)?
                }
                _ => unimplemented!(),
            }
        }
    })
}

fn handle_let(env: &mut Environment<'_>, rest: &[Spanned<Expr>]) -> Result<Value, Error> {
    let [(Expr::Identifier(ident), _), value] = rest else {
        return Err("expected identifier and value".into());
    };
    let value = eval_expr(env, &value.0)?;
    env.bind(ident.clone(), value);
    Ok(Value::Unit)
}

fn handle_fn(env: &mut Environment<'_>, rest: &[Spanned<Expr>]) -> Result<Value, Error> {
    if rest.len() < 2 {
        return Err("fn requires signature and body".into());
    }
    
    let (sig, body_exprs) = rest.split_first().unwrap();
    let Expr::List(sig_parts) = &sig.0 else {
        return Err("expected signature list".into());
    };
    
    let Some(((Expr::Identifier(name), _), params)) = sig_parts.split_first() else {
        return Err("expected name and parameters".into());
    };

    let param_names: Result<Vec<Identifier>, Error> = params
        .iter()
        .map(|(expr, _)| match expr {
            Expr::Identifier(id) => Ok(id.clone()),
            _ => Err("function parameter must be an identifier".into()),
        })
        .collect();

    let param_names = param_names?;
    let function_value = Value::Fn(name.clone(), param_names, body_exprs.to_vec());
    env.bind(name.clone(), function_value);

    Ok(Value::Unit)
}

mod builtins {
    use std::sync::LazyLock;

    use super::*;

    pub static BUILTINS: LazyLock<HashMap<String, Value>> = LazyLock::new(|| {
        let temp: &[(&str, BuiltinFn)] = &[
            ("print_line", |args| {
                for arg in args {
                    print!("{}", arg);
                }
                println!();
                Ok(Value::Unit)
            }),
            ("input_line", |_| {
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                Ok(Value::String(input))
            }),
        ];
        temp.iter()
            .map(|(name, f)| (name.to_string(), Value::BuiltinFn(name, *f)))
            .collect()
    });
}
