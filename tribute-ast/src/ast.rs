use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
        }
    }
}
pub type Spanned<T> = (T, Span);

pub type Identifier = String;

#[salsa::tracked(debug)]
pub struct Program<'db> {
    #[tracked]
    #[returns(ref)]
    pub items: Vec<Item<'db>>,
}

#[salsa::tracked(debug)]
pub struct Item<'db> {
    #[tracked]
    #[returns(ref)]
    pub kind: ItemKind<'db>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
#[non_exhaustive]
pub enum ItemKind<'db> {
    Function(FunctionDefinition<'db>),
}

#[salsa::tracked(debug)]
pub struct FunctionDefinition<'db> {
    pub name: Identifier,
    pub parameters: Vec<Identifier>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Statement {
    Let(LetStatement),
    Expression(Spanned<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LetStatement {
    pub name: Identifier,
    pub value: Spanned<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(i64),
    String(String),
    StringInterpolation(StringInterpolation),
    Identifier(Identifier),
    Binary(BinaryExpression),
    Call(CallExpression),
    Match(MatchExpression),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BinaryExpression {
    pub left: Box<Spanned<Expr>>,
    pub operator: BinaryOperator,
    pub right: Box<Spanned<Expr>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StringInterpolation {
    pub segments: Vec<StringSegment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StringSegment {
    Text(String),
    Interpolation(Box<Spanned<Expr>>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CallExpression {
    pub function: Identifier,
    pub arguments: Vec<Spanned<Expr>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatchExpression {
    pub value: Box<Spanned<Expr>>,
    pub arms: Vec<MatchArm>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub value: Spanned<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    Literal(LiteralPattern),
    Wildcard,
    Identifier(Identifier),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralPattern {
    Number(i64),
    String(String),
    StringInterpolation(StringInterpolation),
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::String(s) => write!(f, "\"{}\"", s),
            Expr::StringInterpolation(interp) => {
                write!(f, "\"")?;
                for segment in &interp.segments {
                    match segment {
                        StringSegment::Text(text) => write!(f, "{}", text)?,
                        StringSegment::Interpolation(expr) => write!(f, "{{{}}}", expr.0)?,
                    }
                }
                write!(f, "\"")
            }
            Expr::Identifier(s) => f.write_str(s),
            Expr::Binary(bin) => write!(f, "({} {} {})", bin.left.0, bin.operator, bin.right.0),
            Expr::Call(call) => {
                write!(f, "{}(", call.function)?;
                for (i, (arg, _)) in call.arguments.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expr::Match(match_expr) => {
                write!(f, "match {} {{ ", match_expr.value.0)?;
                for arm in &match_expr.arms {
                    write!(f, "{} => {}, ", arm.pattern, arm.value.0)?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl std::fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
        }
    }
}

impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Pattern::Literal(lit) => write!(f, "{}", lit),
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Identifier(id) => write!(f, "{}", id),
        }
    }
}

impl std::fmt::Display for LiteralPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LiteralPattern::Number(n) => write!(f, "{}", n),
            LiteralPattern::String(s) => write!(f, "\"{}\"", s),
            LiteralPattern::StringInterpolation(interp) => {
                write!(f, "\"")?;
                for segment in &interp.segments {
                    match segment {
                        StringSegment::Text(text) => write!(f, "{}", text)?,
                        StringSegment::Interpolation(expr) => write!(f, "{{{}}}", expr.0)?,
                    }
                }
                write!(f, "\"")
            }
        }
    }
}
