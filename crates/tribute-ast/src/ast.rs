use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
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
    pub leading_text: String,
    pub segments: Vec<StringSegment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StringSegment {
    pub interpolation: Box<Spanned<Expr>>,
    pub trailing_text: String,
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

