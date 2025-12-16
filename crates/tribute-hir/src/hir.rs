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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Natural number literal: 0, 42, 0b1010, 0o777, 0xc0ffee
    Nat(u64),
    /// Integer literal: +1, -1, +0b1010, -0xff
    Int(i64),
    /// Float literal: 1.0, -3.14
    Float(f64),
    /// Rune (Unicode codepoint): ?a, ?\n, ?\x41, ?\u0041
    Rune(char),
    Bool(bool),
    Nil,
    /// String literal: "hello"
    StringLit(String),
    /// Bytes literal: b"hello", rb"raw"
    BytesLit(Vec<u8>),

    /// Variable reference
    Variable(Identifier),

    /// Function call
    Call {
        func: Box<Spanned<Expr>>,
        args: Vec<Spanned<Expr>>,
    },

    /// Local variable binding with pattern destructuring
    Let {
        pattern: Pattern,
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

    /// Tuple literal #(a, b, c) - first element + rest (non-empty)
    Tuple(Box<Spanned<Expr>>, Vec<Spanned<Expr>>),

    /// Record expression: User { name: "Alice", age: 30 }
    Record {
        type_name: Identifier,
        fields: Vec<RecordField>,
    },
}

// Manual Hash implementation for Expr because f64 doesn't implement Hash
impl std::hash::Hash for Expr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Expr::Nat(n) => n.hash(state),
            Expr::Int(n) => n.hash(state),
            Expr::Float(f) => f.to_bits().hash(state),
            Expr::Rune(c) => c.hash(state),
            Expr::Bool(b) => b.hash(state),
            Expr::Nil => {}
            Expr::StringLit(s) => s.hash(state),
            Expr::BytesLit(b) => b.hash(state),
            Expr::Variable(id) => id.hash(state),
            Expr::Call { func, args } => {
                func.hash(state);
                args.hash(state);
            }
            Expr::Let { pattern, value } => {
                pattern.hash(state);
                value.hash(state);
            }
            Expr::Match { expr, cases } => {
                expr.hash(state);
                cases.hash(state);
            }
            Expr::Lambda { params, body } => {
                params.hash(state);
                body.hash(state);
            }
            Expr::Block(b) => b.hash(state),
            Expr::List(l) => l.hash(state),
            Expr::Tuple(first, rest) => {
                first.hash(state);
                rest.hash(state);
            }
            Expr::Record { type_name, fields } => {
                type_name.hash(state);
                fields.hash(state);
            }
        }
    }
}

/// A field in a record expression
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub enum RecordField {
    /// Spread: ..expr
    Spread(Spanned<Expr>),
    /// Full form: name: value
    Field {
        name: Identifier,
        value: Spanned<Expr>,
    },
    /// Shorthand: name (equivalent to name: name)
    Shorthand(Identifier),
}

/// Pattern matching case
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub guard: Option<Spanned<Expr>>,
    pub body: Spanned<Expr>,
}

/// Patterns for matching
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub enum Pattern {
    /// Literal pattern (exact match)
    Literal(Literal),
    /// Variable pattern (binds value)
    Variable(Identifier),
    /// Wildcard pattern (matches anything)
    Wildcard,
    /// Constructor pattern: Some(x), None, Pair(a, b)
    Constructor {
        name: Identifier,
        args: ConstructorArgs,
    },
    /// Tuple pattern: #(a, b), #(x, y, z) - first element + rest
    Tuple(Box<Pattern>, Vec<Pattern>),
    /// List pattern (matches list structure): [], [a, b], [head, ..tail]
    List {
        elements: Vec<Pattern>,
        rest: Option<Option<Identifier>>,
    },
    /// Rest pattern for matching remaining elements
    Rest(Identifier),
    /// As pattern: Some(x) as opt
    As(Box<Pattern>, Identifier),
    /// Handler pattern for effect handling
    Handler(HandlerPattern),
}

/// Arguments for a constructor pattern
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub enum ConstructorArgs {
    /// No arguments: None
    None,
    /// Tuple-style arguments: Some(x), Pair(a, b)
    Positional(Vec<Pattern>),
    /// Struct-style fields: Ok { value: x }, User { name, .. }
    Named {
        fields: Vec<PatternField>,
        rest: bool,
    },
}

/// A named field in a struct-style constructor pattern
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub struct PatternField {
    pub name: Identifier,
    /// For shorthand `{ name }`, pattern is `Variable(name)` (same as field name)
    pub pattern: Pattern,
}

/// Handler pattern for matching effect requests
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub enum HandlerPattern {
    /// Completion pattern: { result }
    Done(Identifier),
    /// Suspend pattern: { Path::op(args) -> k }
    Suspend {
        operation: Vec<Identifier>,
        args: Vec<Pattern>,
        continuation: Identifier,
    },
}

/// Literal values for patterns
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Natural number: 0, 42, 0b1010
    Nat(u64),
    /// Integer: +1, -1
    Int(i64),
    /// Float: 1.0, -3.14
    Float(f64),
    /// Rune (Unicode codepoint)
    Rune(char),
    Bool(bool),
    Nil,
    /// String pattern: "hello"
    StringPat(String),
    /// Bytes pattern: b"hello", rb"raw"
    BytesPat(Vec<u8>),
}

// Manual Hash implementation for Literal because f64 doesn't implement Hash
impl std::hash::Hash for Literal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Literal::Nat(n) => n.hash(state),
            Literal::Int(n) => n.hash(state),
            Literal::Float(f) => f.to_bits().hash(state),
            Literal::Rune(c) => c.hash(state),
            Literal::Bool(b) => b.hash(state),
            Literal::Nil => {}
            Literal::StringPat(s) => s.hash(state),
            Literal::BytesPat(b) => b.hash(state),
        }
    }
}
