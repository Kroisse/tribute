use crate::hir::*;
use std::collections::BTreeMap;
use tribute_ast::{Expr as AstExpr, SimpleSpan, Spanned};

/// Error type for HIR lowering
#[derive(Debug, Clone, PartialEq)]
pub enum LowerError {
    InvalidFunctionDefinition(String),
    InvalidLetBinding(String),
    InvalidMatchExpression(String),
    UnknownForm(String),
    InvalidPattern(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LowerError::InvalidFunctionDefinition(msg) => {
                write!(f, "Invalid function definition: {}", msg)
            }
            LowerError::InvalidLetBinding(msg) => write!(f, "Invalid let binding: {}", msg),
            LowerError::InvalidMatchExpression(msg) => {
                write!(f, "Invalid match expression: {}", msg)
            }
            LowerError::UnknownForm(form) => write!(f, "Unknown form: {}", form),
            LowerError::InvalidPattern(msg) => write!(f, "Invalid pattern: {}", msg),
        }
    }
}

impl std::error::Error for LowerError {}

pub type LowerResult<T> = Result<T, LowerError>;

/// Intermediate representation for function during lowering
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: tribute_ast::Identifier,
    pub params: Vec<tribute_ast::Identifier>,
    pub body: Vec<Spanned<Expr>>,
    pub span: SimpleSpan,
}

/// Convert AST to HIR function definitions
pub fn lower_to_hir(expressions: Vec<Spanned<AstExpr>>) -> LowerResult<(BTreeMap<tribute_ast::Identifier, FunctionDef>, Option<tribute_ast::Identifier>)> {
    let mut functions = BTreeMap::new();
    let mut main_function = None;

    for (expr, span) in expressions {
        match expr {
            AstExpr::List(list) if !list.is_empty() => {
                if let AstExpr::Identifier(ref name) = list[0].0 {
                    if name == "fn" {
                        let function = lower_function_def(list, span)?;
                        if function.name == "main" {
                            main_function = Some(function.name.clone());
                        }
                        functions.insert(function.name.clone(), function);
                    } else {
                        return Err(LowerError::UnknownForm(format!(
                            "Top-level form must be function definition, found: {}",
                            name
                        )));
                    }
                } else {
                    return Err(LowerError::UnknownForm(
                        "Top-level expressions must start with identifier".to_string(),
                    ));
                }
            }
            _ => {
                return Err(LowerError::UnknownForm(
                    "Top-level expressions must be lists".to_string(),
                ));
            }
        }
    }

    Ok((functions, main_function))
}

fn lower_function_def(list: Vec<Spanned<AstExpr>>, span: SimpleSpan) -> LowerResult<FunctionDef> {
    if list.len() < 3 {
        return Err(LowerError::InvalidFunctionDefinition(
            "Function definition requires at least name and body".to_string(),
        ));
    }

    // Parse function signature: (fn (name param1 param2 ...) body...)
    let sig = match &list[1].0 {
        AstExpr::List(sig_list) if !sig_list.is_empty() => sig_list,
        _ => {
            return Err(LowerError::InvalidFunctionDefinition(
                "Function signature must be a list".to_string(),
            ));
        }
    };

    let name = match &sig[0].0 {
        AstExpr::Identifier(name) => name.clone(),
        _ => {
            return Err(LowerError::InvalidFunctionDefinition(
                "Function name must be an identifier".to_string(),
            ));
        }
    };

    let params: Result<Vec<_>, _> = sig[1..]
        .iter()
        .map(|(expr, _)| match expr {
            AstExpr::Identifier(param) => Ok(param.clone()),
            _ => Err(LowerError::InvalidFunctionDefinition(
                "Function parameters must be identifiers".to_string(),
            )),
        })
        .collect();
    let params = params?;

    // Convert body expressions
    let body: LowerResult<Vec<_>> = list[2..].iter().map(lower_expr).collect();
    let body = body?;

    Ok(FunctionDef {
        name,
        params,
        body,
        span,
    })
}

fn lower_expr(expr: &Spanned<AstExpr>) -> LowerResult<Spanned<Expr>> {
    let (ast_expr, span) = expr;
    let span = *span;

    let hir_expr = match ast_expr {
        AstExpr::Number(n) => Expr::Number(*n),
        AstExpr::String(s) => Expr::String(s.clone()),
        AstExpr::Identifier(id) => Expr::Variable(id.clone()),
        AstExpr::List(list) if list.is_empty() => {
            return Err(LowerError::UnknownForm(
                "Empty list not allowed".to_string(),
            ));
        }
        AstExpr::List(list) => {
            match &list[0].0 {
                AstExpr::Identifier(name) => {
                    match name.as_str() {
                        "let" => lower_let_binding(list)?,
                        "match" => lower_match_expr(list)?,
                        "case" => {
                            return Err(LowerError::UnknownForm(
                                "'case' can only appear inside 'match'".to_string(),
                            ));
                        }
                        _ => {
                            // Function call
                            let func = Box::new(lower_expr(&list[0])?);
                            let args: LowerResult<Vec<_>> =
                                list[1..].iter().map(lower_expr).collect();
                            Expr::Call { func, args: args? }
                        }
                    }
                }
                _ => {
                    // Function call with expression as function
                    let func = Box::new(lower_expr(&list[0])?);
                    let args: LowerResult<Vec<_>> = list[1..].iter().map(lower_expr).collect();
                    Expr::Call { func, args: args? }
                }
            }
        }
    };

    Ok((hir_expr, span))
}

fn lower_let_binding(list: &[Spanned<AstExpr>]) -> LowerResult<Expr> {
    if list.len() != 3 {
        return Err(LowerError::InvalidLetBinding(
            "Let binding requires exactly variable and value: (let var value)".to_string(),
        ));
    }

    let var = match &list[1].0 {
        AstExpr::Identifier(var) => var.clone(),
        _ => {
            return Err(LowerError::InvalidLetBinding(
                "Let variable must be an identifier".to_string(),
            ));
        }
    };

    let value = Box::new(lower_expr(&list[2])?);

    Ok(Expr::Let { var, value })
}

fn lower_match_expr(list: &[Spanned<AstExpr>]) -> LowerResult<Expr> {
    if list.len() < 3 {
        return Err(LowerError::InvalidMatchExpression(
            "Match requires expression and at least one case".to_string(),
        ));
    }

    let expr = Box::new(lower_expr(&list[1])?);

    let mut cases = Vec::new();
    for case_expr in &list[2..] {
        match &case_expr.0 {
            AstExpr::List(case_list) if case_list.len() == 3 => {
                if let AstExpr::Identifier(case_kw) = &case_list[0].0 {
                    if case_kw == "case" {
                        let pattern = lower_pattern(&case_list[1])?;
                        let body = lower_expr(&case_list[2])?;
                        cases.push(MatchCase { pattern, body });
                    } else {
                        return Err(LowerError::InvalidMatchExpression(
                            "Match cases must start with 'case'".to_string(),
                        ));
                    }
                } else {
                    return Err(LowerError::InvalidMatchExpression(
                        "Match cases must start with 'case'".to_string(),
                    ));
                }
            }
            _ => {
                return Err(LowerError::InvalidMatchExpression(
                    "Invalid case format".to_string(),
                ));
            }
        }
    }

    Ok(Expr::Match { expr, cases })
}

fn lower_pattern(expr: &Spanned<AstExpr>) -> LowerResult<Pattern> {
    match &expr.0 {
        AstExpr::Number(n) => Ok(Pattern::Literal(Literal::Number(*n))),
        AstExpr::String(s) => Ok(Pattern::Literal(Literal::String(s.clone()))),
        AstExpr::Identifier(id) if id == "_" => Ok(Pattern::Wildcard),
        AstExpr::Identifier(id) if id.starts_with("...") => {
            // Rest pattern: ...rest
            let rest_name = id.strip_prefix("...").unwrap_or(id);
            Ok(Pattern::Rest(rest_name.to_string()))
        }
        AstExpr::Identifier(id) => Ok(Pattern::Variable(id.clone())),
        AstExpr::List(list) => {
            // List pattern: [pattern1 pattern2 ...]
            let patterns: LowerResult<Vec<_>> = list.iter().map(lower_pattern).collect();
            Ok(Pattern::List(patterns?))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tribute_ast::{Expr as AstExpr, SimpleSpan};

    fn make_span() -> SimpleSpan {
        SimpleSpan::new(0, 0)
    }

    fn test_identifier(name: &str) -> AstExpr {
        AstExpr::Identifier(name.to_string())
    }

    fn test_string(s: &str) -> AstExpr {
        AstExpr::String(s.to_string())
    }

    #[test]
    fn test_lower_simple_function() {
        let input = vec![(
            AstExpr::List(vec![
                (test_identifier("fn"), make_span()),
                (
                    AstExpr::List(vec![(test_identifier("main"), make_span())]),
                    make_span(),
                ),
                (
                    AstExpr::List(vec![
                        (test_identifier("print_line"), make_span()),
                        (test_string("Hello, world!"), make_span()),
                    ]),
                    make_span(),
                ),
            ]),
            make_span(),
        )];

        let (functions, main) = lower_to_hir(input).unwrap();
        assert_eq!(main, Some("main".to_string()));
        assert!(functions.contains_key("main"));
    }
}
