use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tribute_ast::{Identifier, Span, Spanned};

/// High-level intermediate representation for Tribute programs (tracked by Salsa)
#[salsa::tracked(debug)]
pub struct HirProgram<'db> {
    #[return_ref]
    pub functions: BTreeMap<Identifier, HirFunction<'db>>,
    pub main: Option<Identifier>,
}

/// Function definition in HIR (tracked by Salsa)
#[salsa::tracked(debug)]
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
#[salsa::tracked(debug)]
pub struct HirExpr<'db> {
    pub expr: Expr,
    pub span: Span,
}

/// HIR expressions with structured language constructs
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expr {
    /// Literal values
    Number(i64),
    Bool(bool),
    Nil,
    StringInterpolation(StringInterpolation),

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

    /// Lambda expression: fn(x) x + 1
    Lambda {
        params: Vec<Identifier>,
        body: Box<Spanned<Expr>>,
    },

    /// Block expression (sequence of expressions)
    Block(Vec<Spanned<Expr>>),

    /// List literal [a, b, c]
    List(Vec<Spanned<Expr>>),

    /// Tuple literal #(a, b, c)
    Tuple(Vec<Spanned<Expr>>),
}

/// Pattern matching case
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub guard: Option<Spanned<Expr>>,
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

/// String interpolation in HIR
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringInterpolation {
    pub leading_text: String,
    pub segments: Vec<StringSegment>,
}

/// String segment for interpolation
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringSegment {
    pub interpolation: Box<Spanned<Expr>>,
    pub trailing_text: String,
}

/// Literal values for patterns
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Literal {
    Number(i64),
    Bool(bool),
    Nil,
    StringInterpolation(StringInterpolation),
}
