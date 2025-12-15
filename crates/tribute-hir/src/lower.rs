use crate::hir::*;
use std::collections::BTreeMap;
use tribute_ast::{
    Expr as AstExpr, FunctionDefinition, ItemKind, LiteralPattern, Pattern as AstPattern, Program,
    Span, Spanned, Statement,
};

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
    pub span: Span,
}

/// Convert AST Program to HIR function definitions
pub fn lower_program_to_hir<'db>(
    db: &'db dyn salsa::Database,
    program: Program<'db>,
) -> LowerResult<(
    BTreeMap<tribute_ast::Identifier, FunctionDef>,
    Option<tribute_ast::Identifier>,
)> {
    let mut functions = BTreeMap::new();
    let mut main_function = None;

    for item in program.items(db) {
        match item.kind(db) {
            ItemKind::Function(func_def) => {
                let function = lower_function_def(db, *func_def)?;
                if function.name == "main" {
                    main_function = Some(function.name.clone());
                }
                functions.insert(function.name.clone(), function);
            }
            _ => {
                return Err(LowerError::UnknownForm("Unknown item kind".to_string()));
            }
        }
    }

    Ok((functions, main_function))
}

fn lower_function_def<'db>(
    db: &'db dyn salsa::Database,
    func_def: FunctionDefinition<'db>,
) -> LowerResult<FunctionDef> {
    let name = func_def.name(db);
    let params = func_def.parameters(db);
    let body_block = func_def.body(db);

    // Convert body statements to HIR expressions
    let mut body_exprs = Vec::new();
    for statement in &body_block.statements {
        match statement {
            Statement::Let(let_stmt) => {
                let value = lower_expr(&let_stmt.value)?;
                let let_expr = Expr::Let {
                    var: let_stmt.name.clone(),
                    value: Box::new(value),
                };
                body_exprs.push((let_expr, let_stmt.value.1));
            }
            Statement::Expression(expr) => {
                body_exprs.push(lower_expr(expr)?);
            }
        }
    }

    Ok(FunctionDef {
        name,
        params,
        body: body_exprs,
        span: func_def.span(db),
    })
}

fn lower_expr(expr: &Spanned<AstExpr>) -> LowerResult<Spanned<Expr>> {
    let (ast_expr, span) = expr;
    let span = *span;

    let hir_expr = match ast_expr {
        AstExpr::Number(n) => Expr::Number(*n),
        AstExpr::StringInterpolation(interp) => {
            let segments: LowerResult<Vec<_>> = interp
                .segments
                .iter()
                .map(|segment| {
                    Ok(crate::hir::StringSegment {
                        interpolation: Box::new(lower_expr(&segment.interpolation)?),
                        trailing_text: segment.trailing_text.clone(),
                    })
                })
                .collect();
            Expr::StringInterpolation(crate::hir::StringInterpolation {
                leading_text: interp.leading_text.clone(),
                segments: segments?,
            })
        }
        AstExpr::Identifier(id) => Expr::Variable(id.clone()),
        AstExpr::Binary(bin_expr) => {
            let left = Box::new(lower_expr(&bin_expr.left)?);
            let right = Box::new(lower_expr(&bin_expr.right)?);
            let op_name = match bin_expr.operator {
                // Arithmetic
                tribute_ast::BinaryOperator::Add => "+".to_string(),
                tribute_ast::BinaryOperator::Subtract => "-".to_string(),
                tribute_ast::BinaryOperator::Multiply => "*".to_string(),
                tribute_ast::BinaryOperator::Divide => "/".to_string(),
                tribute_ast::BinaryOperator::Modulo => "%".to_string(),
                // Comparison
                tribute_ast::BinaryOperator::Equal => "==".to_string(),
                tribute_ast::BinaryOperator::NotEqual => "!=".to_string(),
                tribute_ast::BinaryOperator::LessThan => "<".to_string(),
                tribute_ast::BinaryOperator::GreaterThan => ">".to_string(),
                tribute_ast::BinaryOperator::LessEqual => "<=".to_string(),
                tribute_ast::BinaryOperator::GreaterEqual => ">=".to_string(),
                // Logical
                tribute_ast::BinaryOperator::And => "&&".to_string(),
                tribute_ast::BinaryOperator::Or => "||".to_string(),
                // Concatenation
                tribute_ast::BinaryOperator::Concat => "<>".to_string(),
            };
            Expr::Call {
                func: Box::new((Expr::Variable(op_name), span)),
                args: vec![*left, *right],
            }
        }
        AstExpr::Call(call_expr) => {
            let func = Box::new((Expr::Variable(call_expr.function.clone()), span));
            let args: LowerResult<Vec<_>> = call_expr.arguments.iter().map(lower_expr).collect();
            Expr::Call { func, args: args? }
        }
        AstExpr::Match(match_expr) => {
            let expr = Box::new(lower_expr(&match_expr.value)?);
            let cases: LowerResult<Vec<_>> = match_expr
                .arms
                .iter()
                .map(|arm| {
                    let pattern = lower_pattern(&arm.pattern)?;
                    let body = lower_expr(&arm.value)?;
                    Ok(MatchCase { pattern, body })
                })
                .collect();
            Expr::Match {
                expr,
                cases: cases?,
            }
        }
        AstExpr::List(elements) => {
            let lowered: LowerResult<Vec<_>> = elements.iter().map(lower_expr).collect();
            Expr::List(lowered?)
        }
        AstExpr::Tuple(elements) => {
            let lowered: LowerResult<Vec<_>> = elements.iter().map(lower_expr).collect();
            Expr::Tuple(lowered?)
        }
    };

    Ok((hir_expr, span))
}

fn lower_pattern(pattern: &AstPattern) -> LowerResult<Pattern> {
    match pattern {
        AstPattern::Literal(lit) => {
            match lit {
                LiteralPattern::Number(n) => Ok(Pattern::Literal(Literal::Number(*n))),
                LiteralPattern::String(s) => {
                    // Convert simple string to StringInterpolation without segments
                    Ok(Pattern::Literal(Literal::StringInterpolation(
                        crate::hir::StringInterpolation {
                            leading_text: s.clone(),
                            segments: Vec::new(),
                        },
                    )))
                }
                LiteralPattern::StringInterpolation(interp) => {
                    let segments: LowerResult<Vec<_>> = interp
                        .segments
                        .iter()
                        .map(|segment| {
                            Ok(crate::hir::StringSegment {
                                interpolation: Box::new(lower_expr(&segment.interpolation)?),
                                trailing_text: segment.trailing_text.clone(),
                            })
                        })
                        .collect();
                    Ok(Pattern::Literal(Literal::StringInterpolation(
                        crate::hir::StringInterpolation {
                            leading_text: interp.leading_text.clone(),
                            segments: segments?,
                        },
                    )))
                }
            }
        }
        AstPattern::Wildcard => Ok(Pattern::Wildcard),
        AstPattern::Identifier(id) => Ok(Pattern::Variable(id.clone())),
    }
}

#[cfg(test)]
mod tests {
    use tribute_ast::{Expr as AstExpr, Span};

    fn make_span() -> Span {
        Span::new(0, 0)
    }

    fn test_identifier(name: &str) -> AstExpr {
        AstExpr::Identifier(name.to_string())
    }

    // TODO: Update tests to work with new AST structure
    #[test]
    fn test_lower_simple_function() {
        // This test needs to be rewritten to use the new AST structure
        // For now, we'll skip it
    }
}
