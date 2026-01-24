//! Expression types for the AST.
//!
//! Expressions are parameterized by a "phase" type `V` that represents
//! the information available about variable references at that stage:
//! - `UnresolvedName`: After parsing, before name resolution
//! - `ResolvedRef`: After name resolution
//! - `TypedRef`: After type checking

use trunk_ir::Symbol;

use super::node_id::NodeId;
use super::pattern::Pattern;
use super::types::TypeAnnotation;

/// An expression in the AST, parameterized by phase type `V`.
///
/// The phase type determines what information is available about
/// variable/function references:
/// - During parsing: `Expr<UnresolvedName>`
/// - After resolve: `Expr<ResolvedRef<'db>>`
/// - After typecheck: `Expr<TypedRef<'db>>`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Expr<V>
where
    V: salsa::Update,
{
    /// Unique identifier for span lookup.
    pub id: NodeId,
    /// The kind of expression.
    pub kind: Box<ExprKind<V>>,
}

impl<V> Expr<V>
where
    V: salsa::Update,
{
    /// Create a new expression with the given ID and kind.
    pub fn new(id: NodeId, kind: ExprKind<V>) -> Self {
        Self {
            id,
            kind: Box::new(kind),
        }
    }
}

/// The different kinds of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum ExprKind<V>
where
    V: salsa::Update,
{
    // === References ===
    /// Variable or function reference.
    /// The type `V` determines what we know about the reference.
    Var(V),

    // === Literals ===
    /// Integer literal: `42`, `-1`
    IntLit(i64),

    /// Floating point literal: `3.14`, `-0.5`
    FloatLit(FloatBits),

    /// String literal: `"hello"`
    StringLit(String),

    /// Boolean literal: `True`, `False`
    BoolLit(bool),

    /// Nil literal: `Nil`
    Nil,

    // === Calls and Construction ===
    /// Function call: `foo(a, b)`
    ///
    /// Also used for UFCS method calls after desugaring:
    /// `x.foo(a, b)` becomes `Call { callee: "foo", args: [x, a, b] }`
    Call { callee: Expr<V>, args: Vec<Expr<V>> },

    /// Constructor application: `Some(42)`, `None`, `Ok(value)`
    Cons { ctor: V, args: Vec<Expr<V>> },

    /// Record construction: `User { name: "Alice", age: 30 }`
    Record {
        type_name: V,
        fields: Vec<(Symbol, Expr<V>)>,
        /// Optional spread expression: `{ ..base, field: value }`
        spread: Option<Expr<V>>,
    },

    /// Field access: `expr.field`
    FieldAccess { expr: Expr<V>, field: Symbol },

    /// Method call: `expr.method(args)`
    ///
    /// This is kept separate from Call during parsing for TDNR.
    /// After TDNR, it may be converted to a Call.
    MethodCall {
        receiver: Expr<V>,
        method: Symbol,
        args: Vec<Expr<V>>,
    },

    // === Control Flow ===
    /// Block expression: `{ stmt1; stmt2; expr }`
    Block(Vec<Stmt<V>>),

    /// Conditional expression: `if cond { then } else { else }`
    If {
        cond: Expr<V>,
        then_branch: Expr<V>,
        else_branch: Option<Expr<V>>,
    },

    /// Pattern matching: `case expr { arms }`
    Case {
        scrutinee: Expr<V>,
        arms: Vec<Arm<V>>,
    },

    /// Lambda expression: `|x, y| expr` or `|x, y| { stmts }`
    Lambda { params: Vec<Param>, body: Expr<V> },

    /// Handle expression: `handle expr { handlers }`
    Handle {
        body: Expr<V>,
        handlers: Vec<HandlerArm<V>>,
    },

    // === Compound Expressions ===
    /// Tuple expression: `(a, b, c)`
    Tuple(Vec<Expr<V>>),

    /// List expression: `[a, b, c]`
    List(Vec<Expr<V>>),

    /// Binary operation: `a + b`, `a && b`
    BinOp {
        op: BinOpKind,
        lhs: Expr<V>,
        rhs: Expr<V>,
    },

    /// Unary operation: `-a`, `!b`
    UnaryOp { op: UnaryOpKind, expr: Expr<V> },

    /// Error expression (for error recovery in parsing).
    Error,
}

/// A statement in a block.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum Stmt<V>
where
    V: salsa::Update,
{
    /// Let binding: `let pattern = expr`
    Let {
        id: NodeId,
        pattern: Pattern<V>,
        ty: Option<TypeAnnotation>,
        value: Expr<V>,
    },

    /// Expression statement: `expr;`
    Expr { id: NodeId, expr: Expr<V> },

    /// Return from a block: the final expression without semicolon.
    Return { id: NodeId, expr: Expr<V> },
}

/// A case arm in pattern matching.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Arm<V>
where
    V: salsa::Update,
{
    /// Node ID for span lookup.
    pub id: NodeId,
    /// The pattern to match.
    pub pattern: Pattern<V>,
    /// Optional guard condition: `| guard`
    pub guard: Option<Expr<V>>,
    /// The body expression.
    pub body: Expr<V>,
}

/// A handler arm in a handle expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct HandlerArm<V>
where
    V: salsa::Update,
{
    /// Node ID for span lookup.
    pub id: NodeId,
    /// The kind of handler (result or effect).
    pub kind: HandlerKind<V>,
    /// The handler body.
    pub body: Expr<V>,
}

/// The kind of handler arm.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum HandlerKind<V>
where
    V: salsa::Update,
{
    /// Result handler: `{ result }`
    Result { binding: Pattern<V> },

    /// Effect handler: `{ Effect.op(args) -> k }`
    Effect {
        ability: V,
        op: Symbol,
        params: Vec<Pattern<V>>,
        continuation: Option<Symbol>,
    },
}

/// A function parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Param {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Parameter name.
    pub name: Symbol,
    /// Optional type annotation.
    pub ty: Option<TypeAnnotation>,
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Boolean
    And,
    Or,

    // String
    Concat, // `<>`
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    /// Arithmetic negation: `-x`
    Neg,
    /// Boolean negation: `!x`
    Not,
}

/// Wrapper for f64 that implements Eq and Hash.
///
/// This is needed because f64 doesn't implement Eq/Hash due to NaN.
/// We use the bit representation for comparison.
#[derive(Clone, Copy, Debug)]
pub struct FloatBits(f64);

impl FloatBits {
    /// Create a new FloatBits from an f64.
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Get the underlying f64 value.
    pub fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for FloatBits {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatBits {}

impl std::hash::Hash for FloatBits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for FloatBits {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

// ============================================================================
// Phase-specific type aliases
// ============================================================================

use super::phases::{ResolvedRef, TypedRef, UnresolvedName};

/// Expression after parsing, before name resolution.
pub type ParsedExpr = Expr<UnresolvedName>;

/// Expression after name resolution.
pub type ResolvedExpr<'db> = Expr<ResolvedRef<'db>>;

/// Expression after type checking.
pub type TypedExpr<'db> = Expr<TypedRef<'db>>;

/// Statement after parsing.
pub type ParsedStmt = Stmt<UnresolvedName>;

/// Statement after name resolution.
pub type ResolvedStmt<'db> = Stmt<ResolvedRef<'db>>;

/// Statement after type checking.
pub type TypedStmt<'db> = Stmt<TypedRef<'db>>;
