pub use tribute_core::{Span, Spanned};

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
    Struct(StructDefinition<'db>),
    Enum(EnumDefinition<'db>),
}

#[salsa::tracked(debug)]
pub struct FunctionDefinition<'db> {
    pub name: Identifier,
    pub parameters: Vec<Identifier>,
    pub body: Block,
    pub span: Span,
}

/// Struct type definition: struct User { name: String, age: Nat }
#[salsa::tracked(debug)]
pub struct StructDefinition<'db> {
    pub name: Identifier,
    pub type_params: Vec<Identifier>,
    pub fields: Vec<StructField>,
    pub is_pub: bool,
    pub span: Span,
}

/// Field in a struct definition
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructField {
    pub name: Identifier,
    pub ty: TypeRef,
}

/// Enum type definition: enum Option(a) { None, Some(a) }
#[salsa::tracked(debug)]
pub struct EnumDefinition<'db> {
    pub name: Identifier,
    pub type_params: Vec<Identifier>,
    pub variants: Vec<EnumVariant>,
    pub is_pub: bool,
    pub span: Span,
}

/// Variant in an enum definition
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub name: Identifier,
    pub fields: Option<VariantFields>,
}

/// Fields for an enum variant
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VariantFields {
    /// Tuple fields: Some(a), Lit(Int)
    Tuple(Vec<TypeRef>),
    /// Struct fields: Ok { value: a }
    Struct(Vec<StructField>),
}

/// Type reference in type annotations
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeRef {
    /// Simple type name: String, Nat, User
    Named(Identifier),
    /// Type variable: a, b, elem
    Variable(Identifier),
    /// Generic type: List(a), Option(String)
    Generic {
        name: Identifier,
        args: Vec<TypeRef>,
    },
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
    MethodCall(MethodCallExpression),
    Match(MatchExpression),
    List(Vec<Spanned<Expr>>),
    Tuple(Vec<Spanned<Expr>>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BinaryExpression {
    pub left: Box<Spanned<Expr>>,
    pub operator: BinaryOperator,
    pub right: Box<Spanned<Expr>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    // Comparison
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessEqual,
    GreaterEqual,
    // Logical
    And,
    Or,
    // Concatenation
    Concat,
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

/// UFCS method call: x.f(y) -> f(x, y), x.f -> f(x)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MethodCallExpression {
    pub receiver: Box<Spanned<Expr>>,
    pub method: Identifier,
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
