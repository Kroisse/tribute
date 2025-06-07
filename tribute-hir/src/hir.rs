use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tribute_ast::{Identifier, SimpleSpan, Spanned};

pub type Span = SimpleSpan;

/// High-level intermediate representation for Tribute programs (tracked by Salsa)
#[salsa::tracked]
pub struct HirProgram<'db> {
    #[return_ref]
    pub functions: BTreeMap<Identifier, HirFunction<'db>>,
    pub main: Option<Identifier>,
}

/// Function definition in HIR (tracked by Salsa)
#[salsa::tracked]
pub struct HirFunction<'db> {
    #[return_ref]
    pub name: Identifier,
    #[return_ref]
    pub params: Vec<Identifier>,
    #[return_ref]
    pub body: Vec<HirExpr<'db>>,
    pub span: Span,
}

/// HIR Expression (tracked by Salsa)
#[salsa::tracked]
pub struct HirExpr<'db> {
    pub expr: Expr,
    pub span: Span,
}

/// HIR expressions with structured language constructs
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Literal values
    Number(i64),
    String(String),

    /// Variable reference
    Variable(Identifier),

    /// Function call
    Call {
        func: Box<Spanned<Expr>>,
        args: Vec<Spanned<Expr>>,
    },

    /// Local variable binding
    Let {
        var: Identifier,
        value: Box<Spanned<Expr>>,
    },

    /// Pattern matching
    Match {
        expr: Box<Spanned<Expr>>,
        cases: Vec<MatchCase>,
    },

    /// Built-in operations
    Builtin {
        name: Identifier,
        args: Vec<Spanned<Expr>>,
    },

    /// Block expression (sequence of expressions)
    Block(Vec<Spanned<Expr>>),
}

/// Pattern matching case
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub body: Spanned<Expr>,
}

/// Patterns for matching
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pattern {
    /// Literal pattern (exact match)
    Literal(Literal),
    /// Variable pattern (binds value)
    Variable(Identifier),
    /// Wildcard pattern (matches anything)
    Wildcard,
    /// List pattern (matches list structure)
    List(Vec<Pattern>),
    /// Rest pattern for matching remaining elements
    Rest(Identifier),
}

/// Literal values for patterns
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Literal {
    Number(i64),
    String(String),
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{n}"),
            Expr::String(s) => write!(f, "\"{s}\""),
            Expr::Variable(v) => write!(f, "{v}"),
            Expr::Call { func, args } => {
                write!(f, "({}", func.0)?;
                for arg in args {
                    write!(f, " {}", arg.0)?;
                }
                write!(f, ")")
            }
            Expr::Let { var, value } => {
                write!(f, "(let {var} {})", value.0)
            }
            Expr::Match { expr, cases } => {
                write!(f, "(match {}", expr.0)?;
                for case in cases {
                    write!(f, " (case {:?} {})", case.pattern, case.body.0)?;
                }
                write!(f, ")")
            }
            Expr::Builtin { name, args } => {
                write!(f, "({name}")?;
                for arg in args {
                    write!(f, " {}", arg.0)?;
                }
                write!(f, ")")
            }
            Expr::Block(exprs) => {
                write!(f, "(block")?;
                for expr in exprs {
                    write!(f, " {}", expr.0)?;
                }
                write!(f, ")")
            }
        }
    }
}
