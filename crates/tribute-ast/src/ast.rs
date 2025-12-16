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
    Use(UseDeclaration<'db>),
    Mod(ModDeclaration<'db>),
    Function(FunctionDefinition<'db>),
    Struct(StructDefinition<'db>),
    Enum(EnumDefinition<'db>),
    Const(ConstDefinition<'db>),
}

/// Use declaration: use std::io, use std::collections::{List, Map}
#[salsa::tracked(debug)]
pub struct UseDeclaration<'db> {
    pub path: UsePath,
    pub is_pub: bool,
    pub span: Span,
}

/// Path in a use declaration
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UsePath {
    /// Path segments: ["std", "collections"]
    pub segments: Vec<Identifier>,
    /// Optional group of items: {List, Map}
    pub group: Option<Vec<Identifier>>,
}

/// Module declaration: mod foo, mod foo { ... }
#[salsa::tracked(debug)]
pub struct ModDeclaration<'db> {
    pub name: Identifier,
    /// None for external module (mod foo), Some for inline module (mod foo { ... })
    #[tracked]
    #[returns(ref)]
    pub items: Option<Vec<Item<'db>>>,
    pub is_pub: bool,
    pub span: Span,
}

/// Parameter with optional type annotation: x or x: Int
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter {
    pub name: Identifier,
    /// Optional type annotation
    pub ty: Option<TypeRef>,
}

#[salsa::tracked(debug)]
pub struct FunctionDefinition<'db> {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    /// Optional return type annotation: -> Int
    pub return_type: Option<TypeRef>,
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

/// Const definition: const MAX_SIZE = 1000
#[salsa::tracked(debug)]
pub struct ConstDefinition<'db> {
    pub name: Identifier,
    pub ty: Option<TypeRef>,
    pub value: Spanned<Expr>,
    pub is_pub: bool,
    pub span: Span,
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

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Statement {
    Let(LetStatement),
    Expression(Spanned<Expr>),
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct LetStatement {
    pub pattern: Pattern,
    pub value: Spanned<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Natural number literal: 0, 42 (no sign)
    Nat(u64),
    /// Integer literal: +1, -1 (explicit sign)
    Int(i64),
    /// Float literal: 1.0, -3.14 (decimal point required)
    Float(f64),
    /// Rune literal: ?a, ?\n, ?\x41, ?\u0041 (Unicode codepoint)
    Rune(char),
    Bool(bool),
    Nil,
    StringInterpolation(StringInterpolation),
    /// Bytes literal: b"hello", b"\x00\x01"
    BytesInterpolation(BytesInterpolation),
    Identifier(Identifier),
    Binary(BinaryExpression),
    Call(CallExpression),
    MethodCall(MethodCallExpression),
    Match(MatchExpression),
    Lambda(LambdaExpression),
    /// Block expression: { expr1; expr2; ... }
    Block(Vec<Statement>),
    List(Vec<Spanned<Expr>>),
    /// Tuple expression: #(a, b, c) - first element + rest (non-empty)
    Tuple(Box<Spanned<Expr>>, Vec<Spanned<Expr>>),
    /// Record expression: User { name: "Alice", age: 30 }
    Record(RecordExpression),
    /// Operator as function: (+), (-), (<>), (Int::+), (String::<>), etc.
    OperatorFn(OperatorFnExpression),
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
            Expr::StringInterpolation(s) => s.hash(state),
            Expr::BytesInterpolation(b) => b.hash(state),
            Expr::Identifier(id) => id.hash(state),
            Expr::Binary(b) => b.hash(state),
            Expr::Call(c) => c.hash(state),
            Expr::MethodCall(m) => m.hash(state),
            Expr::Match(m) => m.hash(state),
            Expr::Lambda(l) => l.hash(state),
            Expr::Block(b) => b.hash(state),
            Expr::List(l) => l.hash(state),
            Expr::Tuple(first, rest) => {
                first.hash(state);
                rest.hash(state);
            }
            Expr::Record(r) => r.hash(state),
            Expr::OperatorFn(o) => o.hash(state),
        }
    }
}

/// Operator function expression: (+), (Int::+), (String::<>)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperatorFnExpression {
    /// The operator: "+", "-", "<>", etc.
    pub op: String,
    /// Optional type qualifier: Some("Int") for (Int::+), None for (+)
    pub qualifier: Option<Identifier>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct BinaryExpression {
    pub left: Box<Spanned<Expr>>,
    pub operator: BinaryOperator,
    /// Optional type qualifier: Some("Int") for `a Int::+ b`, None for `a + b`
    pub qualifier: Option<Identifier>,
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

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct StringInterpolation {
    pub leading_text: String,
    pub segments: Vec<StringSegment>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct StringSegment {
    pub interpolation: Box<Spanned<Expr>>,
    pub trailing_text: String,
}

/// Bytes with interpolation support: b"hello \{name}"
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct BytesInterpolation {
    pub leading_bytes: Vec<u8>,
    pub segments: Vec<BytesSegment>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct BytesSegment {
    pub interpolation: Box<Spanned<Expr>>,
    pub trailing_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct CallExpression {
    pub function: Identifier,
    pub arguments: Vec<Spanned<Expr>>,
}

/// UFCS method call: x.f(y) -> f(x, y), x.f -> f(x)
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct MethodCallExpression {
    pub receiver: Box<Spanned<Expr>>,
    pub method: Identifier,
    pub arguments: Vec<Spanned<Expr>>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct MatchExpression {
    pub value: Box<Spanned<Expr>>,
    pub arms: Vec<MatchArm>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub branches: Vec<GuardedBranch>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct GuardedBranch {
    pub guard: Option<Spanned<Expr>>,
    pub value: Spanned<Expr>,
}

/// Lambda expression: fn(x) x + 1, fn(x: Int) -> Int x + 1
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct LambdaExpression {
    pub parameters: Vec<Parameter>,
    /// Optional return type annotation: -> Int
    pub return_type: Option<TypeRef>,
    pub body: Box<Spanned<Expr>>,
}

/// Record expression: User { name: "Alice", age: 30 }
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct RecordExpression {
    /// Type name (e.g., "User", "Point")
    pub type_name: Identifier,
    /// Record fields
    pub fields: Vec<RecordField>,
}

/// A field in a record expression
#[derive(Clone, Debug, PartialEq, Hash)]
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

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Pattern {
    Literal(LiteralPattern),
    Wildcard,
    Identifier(Identifier),
    /// Constructor pattern: Some(x), None, Pair(a, b), Ok { value: x }
    Constructor(ConstructorPattern),
    /// Tuple pattern: #(a, b), #(x, y, z) - first element + rest (non-empty)
    Tuple(Box<Pattern>, Vec<Pattern>),
    /// List pattern: [], [a, b, c], [head, ..tail]
    List(ListPattern),
    /// As pattern: Some(x) as opt
    As(Box<Pattern>, Identifier),
    /// Handler pattern: { result } or { State::get() -> k }
    Handler(HandlerPattern),
}

/// Constructor pattern for matching enum variants
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct ConstructorPattern {
    /// Constructor name (e.g., "Some", "None", "Ok")
    pub name: Identifier,
    /// Pattern arguments
    pub args: ConstructorArgs,
}

/// Arguments for a constructor pattern
#[derive(Clone, Debug, PartialEq, Hash)]
pub enum ConstructorArgs {
    /// No arguments: None
    None,
    /// Tuple-style arguments: Some(x), Pair(a, b)
    Positional(Vec<Pattern>),
    /// Struct-style fields: Ok { value: x }, User { name, .. }
    Named {
        fields: Vec<PatternField>,
        /// True if `..` is present to ignore remaining fields
        rest: bool,
    },
}

/// A named field in a struct-style constructor pattern
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct PatternField {
    pub name: Identifier,
    /// For shorthand `{ name }`, pattern is `Identifier(name)` (same as field name)
    pub pattern: Pattern,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LiteralPattern {
    /// Natural number pattern: 0, 42
    Nat(u64),
    /// Integer pattern: +1, -1
    Int(i64),
    /// Float pattern: 1.0, -3.14
    Float(f64),
    /// Rune literal pattern: ?a, ?\n, ?\x41
    Rune(char),
    Bool(bool),
    Nil,
    String(String),
    StringInterpolation(StringInterpolation),
    /// Bytes pattern: b"hello", rb"data"
    Bytes(Vec<u8>),
    BytesInterpolation(BytesInterpolation),
}

// Manual Hash implementation for LiteralPattern because f64 doesn't implement Hash
impl std::hash::Hash for LiteralPattern {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            LiteralPattern::Nat(n) => n.hash(state),
            LiteralPattern::Int(n) => n.hash(state),
            LiteralPattern::Float(f) => f.to_bits().hash(state),
            LiteralPattern::Rune(c) => c.hash(state),
            LiteralPattern::Bool(b) => b.hash(state),
            LiteralPattern::Nil => {}
            LiteralPattern::String(s) => s.hash(state),
            LiteralPattern::StringInterpolation(s) => s.hash(state),
            LiteralPattern::Bytes(b) => b.hash(state),
            LiteralPattern::BytesInterpolation(b) => b.hash(state),
        }
    }
}

/// List pattern: [], [a, b, c], [head, ..tail]
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct ListPattern {
    /// Patterns for list elements
    pub elements: Vec<Pattern>,
    /// Optional rest pattern: ..tail or just ..
    pub rest: Option<Option<Identifier>>,
}

/// Handler pattern for matching effect requests
#[derive(Clone, Debug, PartialEq, Hash)]
pub enum HandlerPattern {
    /// Completion pattern: { result }
    Done(Identifier),
    /// Suspend pattern: { Path::op(args) -> k }
    Suspend {
        /// Operation path (e.g., "State::get")
        operation: Vec<Identifier>,
        /// Pattern arguments for the operation
        args: Vec<Pattern>,
        /// Continuation binding
        continuation: Identifier,
    },
}
