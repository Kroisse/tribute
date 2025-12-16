use crate::hir::*;
use std::collections::BTreeMap;
use tribute_ast::{
    ConstructorArgs as AstConstructorArgs, Expr as AstExpr, FunctionDefinition, ItemKind,
    LiteralPattern, Pattern as AstPattern, PatternField as AstPatternField, Program, Span, Spanned,
    Statement,
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
            ItemKind::Use(_)
            | ItemKind::Mod(_)
            | ItemKind::Struct(_)
            | ItemKind::Enum(_)
            | ItemKind::Const(_) => {
                // Use/mod/type/const declarations are not evaluated code
                // They will be handled by the module system and type checker (future work)
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
    // Extract just the parameter names (type annotations are handled by type checker)
    let params: Vec<_> = func_def
        .parameters(db)
        .iter()
        .map(|p| p.name.clone())
        .collect();
    let body_block = func_def.body(db);

    // Convert body statements to HIR expressions
    let mut body_exprs = Vec::new();
    for statement in &body_block.statements {
        match statement {
            Statement::Let(let_stmt) => {
                let pattern = lower_pattern(&let_stmt.pattern)?;
                let value = lower_expr(&let_stmt.value)?;
                let let_expr = Expr::Let {
                    pattern,
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
        AstExpr::Nat(n) => Expr::Nat(*n),
        AstExpr::Int(n) => Expr::Int(*n),
        AstExpr::Float(n) => Expr::Float(*n),
        AstExpr::Rune(ch) => Expr::Rune(*ch),
        AstExpr::Bool(b) => Expr::Bool(*b),
        AstExpr::Nil => Expr::Nil,
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
            let op_name = operator_to_string(&bin_expr.operator);
            // If there's a qualifier, create a qualified name like "Int::+"
            let func_name = match &bin_expr.qualifier {
                Some(q) => format!("{}::{}", q, op_name),
                None => op_name,
            };
            Expr::Call {
                func: Box::new((Expr::Variable(func_name), span)),
                args: vec![*left, *right],
            }
        }
        AstExpr::Call(call_expr) => {
            let func = Box::new((Expr::Variable(call_expr.function.clone()), span));
            let args: LowerResult<Vec<_>> = call_expr.arguments.iter().map(lower_expr).collect();
            Expr::Call { func, args: args? }
        }
        // UFCS: x.f(y, z) -> f(x, y, z)
        AstExpr::MethodCall(method_call) => {
            let func = Box::new((Expr::Variable(method_call.method.clone()), span));
            let receiver = lower_expr(&method_call.receiver)?;
            let mut args = vec![receiver];
            for arg in &method_call.arguments {
                args.push(lower_expr(arg)?);
            }
            Expr::Call { func, args }
        }
        AstExpr::Match(match_expr) => {
            let expr = Box::new(lower_expr(&match_expr.value)?);
            let mut cases = Vec::new();
            for arm in &match_expr.arms {
                let pattern = lower_pattern(&arm.pattern)?;
                for branch in &arm.branches {
                    let guard = branch.guard.as_ref().map(lower_expr).transpose()?;
                    let body = lower_expr(&branch.value)?;
                    cases.push(MatchCase {
                        pattern: pattern.clone(),
                        guard,
                        body,
                    });
                }
            }
            Expr::Match { expr, cases }
        }
        AstExpr::Lambda(lambda_expr) => {
            let body = Box::new(lower_expr(&lambda_expr.body)?);
            // Extract just the parameter names (type annotations are handled by type checker)
            let params: Vec<_> = lambda_expr
                .parameters
                .iter()
                .map(|p| p.name.clone())
                .collect();
            Expr::Lambda { params, body }
        }
        AstExpr::Block(statements) => {
            let mut exprs = Vec::new();
            for statement in statements {
                match statement {
                    Statement::Let(let_stmt) => {
                        let pattern = lower_pattern(&let_stmt.pattern)?;
                        let value = lower_expr(&let_stmt.value)?;
                        let let_expr = Expr::Let {
                            pattern,
                            value: Box::new(value),
                        };
                        exprs.push((let_expr, let_stmt.value.1));
                    }
                    Statement::Expression(expr) => {
                        exprs.push(lower_expr(expr)?);
                    }
                }
            }
            Expr::Block(exprs)
        }
        AstExpr::List(elements) => {
            let lowered: LowerResult<Vec<_>> = elements.iter().map(lower_expr).collect();
            Expr::List(lowered?)
        }
        AstExpr::Tuple(first, rest) => {
            let lowered_first = Box::new(lower_expr(first)?);
            let lowered_rest: LowerResult<Vec<_>> = rest.iter().map(lower_expr).collect();
            Expr::Tuple(lowered_first, lowered_rest?)
        }
        AstExpr::Record(record_expr) => {
            let fields: LowerResult<Vec<_>> =
                record_expr.fields.iter().map(lower_record_field).collect();
            Expr::Record {
                type_name: record_expr.type_name.clone(),
                fields: fields?,
            }
        }
        AstExpr::OperatorFn(op_fn) => {
            // Operator functions are just references to the operator as a variable
            // e.g., (+) becomes Variable("+"), (Int::+) becomes Variable("Int::+")
            let func_name = match &op_fn.qualifier {
                Some(q) => format!("{}::{}", q, op_fn.op),
                None => op_fn.op.clone(),
            };
            Expr::Variable(func_name)
        }
    };

    Ok((hir_expr, span))
}

fn lower_record_field(field: &tribute_ast::RecordField) -> LowerResult<crate::hir::RecordField> {
    match field {
        tribute_ast::RecordField::Spread(expr) => {
            Ok(crate::hir::RecordField::Spread(lower_expr(expr)?))
        }
        tribute_ast::RecordField::Field { name, value } => Ok(crate::hir::RecordField::Field {
            name: name.clone(),
            value: lower_expr(value)?,
        }),
        tribute_ast::RecordField::Shorthand(name) => {
            Ok(crate::hir::RecordField::Shorthand(name.clone()))
        }
    }
}

fn lower_pattern(pattern: &AstPattern) -> LowerResult<Pattern> {
    match pattern {
        AstPattern::Literal(lit) => {
            match lit {
                LiteralPattern::Nat(n) => Ok(Pattern::Literal(Literal::Nat(*n))),
                LiteralPattern::Int(n) => Ok(Pattern::Literal(Literal::Int(*n))),
                LiteralPattern::Float(n) => Ok(Pattern::Literal(Literal::Float(*n))),
                LiteralPattern::Rune(ch) => Ok(Pattern::Literal(Literal::Rune(*ch))),
                LiteralPattern::Bool(b) => Ok(Pattern::Literal(Literal::Bool(*b))),
                LiteralPattern::Nil => Ok(Pattern::Literal(Literal::Nil)),
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
        AstPattern::Constructor(ctor) => {
            let args = lower_constructor_args(&ctor.args)?;
            Ok(Pattern::Constructor {
                name: ctor.name.clone(),
                args,
            })
        }
        AstPattern::Tuple(first, rest) => {
            let lowered_first = Box::new(lower_pattern(first)?);
            let lowered_rest: LowerResult<Vec<_>> = rest.iter().map(lower_pattern).collect();
            Ok(Pattern::Tuple(lowered_first, lowered_rest?))
        }
        AstPattern::List(list_pat) => {
            let elements: LowerResult<Vec<_>> =
                list_pat.elements.iter().map(lower_pattern).collect();
            Ok(Pattern::List {
                elements: elements?,
                rest: list_pat.rest.clone(),
            })
        }
        AstPattern::As(inner, binding) => {
            let lowered_inner = Box::new(lower_pattern(inner)?);
            Ok(Pattern::As(lowered_inner, binding.clone()))
        }
        AstPattern::Handler(handler_pat) => {
            Ok(Pattern::Handler(lower_handler_pattern(handler_pat)?))
        }
    }
}

fn lower_constructor_args(args: &AstConstructorArgs) -> LowerResult<ConstructorArgs> {
    match args {
        AstConstructorArgs::None => Ok(ConstructorArgs::None),
        AstConstructorArgs::Positional(patterns) => {
            let lowered: LowerResult<Vec<_>> = patterns.iter().map(lower_pattern).collect();
            Ok(ConstructorArgs::Positional(lowered?))
        }
        AstConstructorArgs::Named { fields, rest } => {
            let lowered: LowerResult<Vec<_>> = fields.iter().map(lower_pattern_field).collect();
            Ok(ConstructorArgs::Named {
                fields: lowered?,
                rest: *rest,
            })
        }
    }
}

fn lower_pattern_field(field: &AstPatternField) -> LowerResult<PatternField> {
    Ok(PatternField {
        name: field.name.clone(),
        pattern: lower_pattern(&field.pattern)?,
    })
}

fn lower_handler_pattern(
    handler_pat: &tribute_ast::HandlerPattern,
) -> LowerResult<crate::hir::HandlerPattern> {
    match handler_pat {
        tribute_ast::HandlerPattern::Done(id) => Ok(crate::hir::HandlerPattern::Done(id.clone())),
        tribute_ast::HandlerPattern::Suspend {
            operation,
            args,
            continuation,
        } => {
            let lowered_args: LowerResult<Vec<_>> = args.iter().map(lower_pattern).collect();
            Ok(crate::hir::HandlerPattern::Suspend {
                operation: operation.clone(),
                args: lowered_args?,
                continuation: continuation.clone(),
            })
        }
    }
}

/// Convert a BinaryOperator to its string representation
fn operator_to_string(op: &tribute_ast::BinaryOperator) -> String {
    match op {
        tribute_ast::BinaryOperator::Add => "+".to_string(),
        tribute_ast::BinaryOperator::Subtract => "-".to_string(),
        tribute_ast::BinaryOperator::Multiply => "*".to_string(),
        tribute_ast::BinaryOperator::Divide => "/".to_string(),
        tribute_ast::BinaryOperator::Modulo => "%".to_string(),
        tribute_ast::BinaryOperator::Equal => "==".to_string(),
        tribute_ast::BinaryOperator::NotEqual => "!=".to_string(),
        tribute_ast::BinaryOperator::LessThan => "<".to_string(),
        tribute_ast::BinaryOperator::GreaterThan => ">".to_string(),
        tribute_ast::BinaryOperator::LessEqual => "<=".to_string(),
        tribute_ast::BinaryOperator::GreaterEqual => ">=".to_string(),
        tribute_ast::BinaryOperator::And => "&&".to_string(),
        tribute_ast::BinaryOperator::Or => "||".to_string(),
        tribute_ast::BinaryOperator::Concat => "<>".to_string(),
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
