//! Expression types for the AST.
//!
//! Expressions are parameterized by a "phase" type `V` that represents
//! the information available about variable references at that stage:
//! - `UnresolvedName`: After parsing, before name resolution
//! - `ResolvedRef`: After name resolution
//! - `TypedRef`: After type checking

use itertools::Itertools;
use trunk_ir::{Symbol, SymbolPath};

use super::node_id::NodeId;
use super::pattern::Pattern;
use super::phases::LocalId;
use super::types::TypeAnnotation;

/// A field initializer in a record expression: `name: value`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct FieldInit<V> {
    /// Identifier whose span is the field name.
    pub id: NodeId,
    /// The field name.
    pub name: Symbol,
    /// The initializing expression.
    pub value: Expr<V>,
}

/// The functions a callee may name, by the rank of how each comes into the
/// call's scope (`new-plans/modules.md` § 후보의 순위). A function is in its
/// highest rank only.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Candidates<V> {
    /// The functions the scope names directly: its declarations and `use`s.
    #[salsa_value(unsafe(prove(V: salsa::SalsaValue)))]
    direct: Vec<V>,
    /// `m::f` for each name `m` in the scope, in a method call only.
    #[salsa_value(unsafe(prove(V: salsa::SalsaValue)))]
    in_scope: Vec<V>,
    /// The prelude's functions, whatever the scope hides.
    #[salsa_value(unsafe(prove(V: salsa::SalsaValue)))]
    prelude: Vec<V>,
}

impl<V> Default for Candidates<V> {
    fn default() -> Self {
        Self {
            direct: Vec::new(),
            in_scope: Vec::new(),
            prelude: Vec::new(),
        }
    }
}

impl<V: PartialEq> Candidates<V> {
    /// The candidates of each rank, without those a higher rank has.
    pub fn new(direct: Vec<V>, mut in_scope: Vec<V>, mut prelude: Vec<V>) -> Self {
        in_scope.retain(|function| !direct.contains(function));
        prelude.retain(|function| !direct.contains(function) && !in_scope.contains(function));
        Self {
            direct,
            in_scope,
            prelude,
        }
    }
}

impl<V> Candidates<V> {
    /// The candidates of each rank, highest first.
    pub fn ranks(&self) -> [&[V]; 3] {
        [&self.direct, &self.in_scope, &self.prelude]
    }

    /// The candidates of each rank, highest first, to reorder.
    pub fn ranks_mut(&mut self) -> [&mut Vec<V>; 3] {
        [&mut self.direct, &mut self.in_scope, &mut self.prelude]
    }

    /// Every candidate, highest rank first.
    pub fn iter(&self) -> impl Iterator<Item = &V> {
        self.ranks().into_iter().flatten()
    }

    pub fn len(&self) -> usize {
        self.ranks().iter().map(|rank| rank.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.ranks().iter().all(|rank| rank.is_empty())
    }

    /// The candidate, when there is one across all ranks.
    pub fn single(&self) -> Option<&V> {
        self.iter().exactly_one().ok()
    }

    /// Each candidate mapped by `f`, in its rank.
    pub fn map<W>(&self, mut f: impl FnMut(&V) -> W) -> Candidates<W> {
        self.filter_map(|candidate| Some(f(candidate)))
    }

    /// The candidates `f` maps to a value, in their ranks.
    pub fn filter_map<W>(&self, mut f: impl FnMut(&V) -> Option<W>) -> Candidates<W> {
        let [direct, in_scope, prelude] = self
            .ranks()
            .map(|rank| rank.iter().filter_map(&mut f).collect());
        Candidates {
            direct,
            in_scope,
            prelude,
        }
    }
}

/// An expression in the AST, parameterized by phase type `V`.
///
/// The phase type determines what information is available about
/// variable/function references:
/// - During parsing: `Expr<UnresolvedName>`
/// - After resolve: `Expr<ResolvedRef<'db>>`
/// - After typecheck: `Expr<TypedRef<'db>>`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Expr<V> {
    /// Unique identifier for span lookup.
    pub id: NodeId,
    /// The kind of expression.
    // The recursive AST owns its children; V is the only phase-dependent value.
    #[salsa_value(unsafe(prove(V: salsa::SalsaValue)))]
    pub kind: Box<ExprKind<V>>,
}

impl<V> Expr<V> {
    /// Create a new expression with the given ID and kind.
    pub fn new(id: NodeId, kind: ExprKind<V>) -> Self {
        Self {
            id,
            kind: Box::new(kind),
        }
    }
}

/// The different kinds of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum ExprKind<V> {
    // === References ===
    /// Variable or function reference.
    /// The type `V` determines what we know about the reference.
    Var(V),

    // === Literals ===
    /// Natural number literal: `42`
    NatLit(u64),

    /// Signed integer literal: `+42`, `-1`
    IntLit(i64),

    /// Floating point literal: `3.14`, `-0.5`
    FloatLit(FloatBits),

    /// String literal: `"hello"`
    StringLit(String),

    /// Bytes literal: `b"hello"`
    BytesLit(Vec<u8>),

    /// Boolean literal: `True`, `False`
    BoolLit(bool),

    /// Nil literal: `Nil`
    Nil,

    /// Rune (character) literal: `?a`, `?\n`
    RuneLit(char),

    // === Calls and Construction ===
    /// Function call: `foo(a, b)`
    ///
    /// A method call `x.foo(a, b)` is this call with `x` as the first
    /// argument and a [`Functions`](Self::Functions) callee for `foo`.
    Call {
        callee: Expr<V>,
        args: Vec<Expr<V>>,
        syntax: CallSyntax,
    },

    /// Constructor application: `Some(42)`, `None`, `Ok(value)`
    Cons { ctor: V, args: Vec<Expr<V>> },

    /// Record construction: `User { name: "Alice", age: 30 }`
    Record {
        type_name: V,
        fields: Vec<FieldInit<V>>,
        /// Optional spread expression: `{ ..base, field: value }`
        spread: Option<Expr<V>>,
    },

    /// A name that stands for one of several declared functions: `f` in
    /// `x.f(y)`, or a name several `use`s give different functions.
    ///
    /// Parsing writes the method of `x.f(y)` as the callee of a call whose
    /// first argument is `x`, without candidates. Name resolution collects
    /// the functions the name may be from the call's scope, and type checking
    /// selects one, replacing the expression with a reference to it.
    Functions {
        name: SymbolPath,
        /// Selection tries a lower rank only when no candidate of the
        /// higher ones fits.
        // V is the only phase-dependent value.
        #[salsa_value(unsafe(prove(V: salsa::SalsaValue)))]
        candidates: Candidates<V>,
    },

    // === Control Flow ===
    /// Block expression: `{ stmt1; stmt2; expr }`
    Block { stmts: Vec<Stmt<V>>, value: Expr<V> },

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

    /// Resume expression: `resume value` — only valid inside `op` handler arms.
    Resume {
        arg: Expr<V>,
        /// LocalId for the continuation binding, assigned during name resolution.
        local_id: Option<LocalId>,
    },

    /// Proper tail call: `become f(args)`. The operand must be a call in tail
    /// position; typechecking enforces both.
    Become { call: Expr<V> },

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

    /// Error expression (for error recovery in parsing).
    Error,
}

/// A statement in a block.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum Stmt<V> {
    /// Let binding: `let pattern = expr`
    Let {
        id: NodeId,
        pattern: Pattern<V>,
        ty: Option<TypeAnnotation>,
        value: Expr<V>,
    },

    /// Expression statement: `expr;`
    Expr { id: NodeId, expr: Expr<V> },
}

/// A case arm in pattern matching.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Arm<V> {
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct HandlerArm<V> {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// The kind of handler (result or effect).
    pub kind: HandlerKind<V>,
    /// The handler body.
    pub body: Expr<V>,
}

/// The kind of handler arm.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum HandlerKind<V> {
    /// Completion handler: `do result { body }`
    Do { binding: Pattern<V> },

    /// Tail-resumptive handler: `fn Op(args) { body }`
    /// Return value is used as the resume value. No continuation capture.
    Fn {
        ability: V,
        op: Symbol,
        params: Vec<Pattern<V>>,
    },

    /// General handler: `op Op(args) { body }`
    /// Uses `resume` keyword in body. Continuation is captured.
    Op {
        ability: V,
        op: Symbol,
        params: Vec<Pattern<V>>,
        /// LocalId for the synthetic `resume` binding (assigned during name resolution).
        resume_local_id: Option<LocalId>,
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
    /// Local ID assigned during name resolution (None before resolution).
    pub local_id: Option<super::phases::LocalId>,
}

/// How a call was written.
///
/// Diagnostics and tools read this to speak of a call the way the source
/// wrote it. What a call means does not depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum CallSyntax {
    /// `f(x, y)`
    Call,
    /// `x.f(y)` or `x.f`
    Method,
    /// `x + y`
    Operator,
}

/// Binary operators.
///
/// Only boolean operators remain here. Arithmetic and comparison operators
/// are desugared to calls of the operator's name in astgen and resolved via TDNR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    And,
    Or,
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
