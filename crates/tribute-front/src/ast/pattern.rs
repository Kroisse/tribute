//! Pattern types for the AST.
//!
//! Patterns are used in:
//! - Case expressions: `case expr { Some(x) => ... }`
//! - Let bindings: `let (a, b) = pair`
//! - Function parameters: `fn foo(User { name, age }) { ... }`
//!
//! Like expressions, patterns are parameterized by a phase type `V`.

use trunk_ir::Symbol;

use super::expr::FloatBits;
use super::node_id::NodeId;

/// A pattern in the AST, parameterized by phase type `V`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Pattern<V>
where
    V: salsa::Update,
{
    /// Unique identifier for span lookup.
    pub id: NodeId,
    /// The kind of pattern.
    pub kind: Box<PatternKind<V>>,
}

impl<V> Pattern<V>
where
    V: salsa::Update,
{
    /// Create a new pattern with the given ID and kind.
    pub fn new(id: NodeId, kind: PatternKind<V>) -> Self {
        Self {
            id,
            kind: Box::new(kind),
        }
    }
}

/// The different kinds of patterns.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum PatternKind<V>
where
    V: salsa::Update,
{
    /// Wildcard pattern: `_`
    ///
    /// Matches anything without binding.
    Wildcard,

    /// Identifier pattern: `x`
    ///
    /// Matches anything and binds it to the name.
    Bind { name: Symbol },

    /// Literal pattern: `42`, `"hello"`, `true`
    Literal(LiteralPattern),

    /// Constructor/Variant pattern: `Some(x)`, `None`, `Ok(value)`
    ///
    /// The `ctor` is the constructor reference (phase-dependent).
    Variant { ctor: V, fields: Vec<Pattern<V>> },

    /// Record pattern: `{ name, age: a }`
    Record {
        type_name: Option<V>,
        fields: Vec<FieldPattern<V>>,
        /// Whether there's a `..` to ignore remaining fields.
        rest: bool,
    },

    /// Tuple pattern: `(a, b, c)`
    Tuple(Vec<Pattern<V>>),

    /// List pattern: `[a, b, c]`
    List(Vec<Pattern<V>>),

    /// List pattern with rest: `[head, ..tail]` or `[a, b, ..]`
    ListRest {
        /// Patterns for the head elements.
        head: Vec<Pattern<V>>,
        /// Optional binding for the rest (None means `..` without binding).
        rest: Option<Symbol>,
    },

    /// As pattern: `pattern as name`
    ///
    /// Matches the inner pattern and also binds the whole value.
    As { pattern: Pattern<V>, name: Symbol },

    /// Or pattern: `None | Some(0)`
    ///
    /// Matches if any of the alternatives match.
    /// All alternatives must bind the same names.
    Or(Vec<Pattern<V>>),

    /// Error pattern (for error recovery).
    Error,
}

/// A field in a record pattern.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct FieldPattern<V>
where
    V: salsa::Update,
{
    /// The field name.
    pub name: Symbol,
    /// The pattern to match the field value.
    /// If None, uses a bind pattern with the same name.
    pub pattern: Option<Pattern<V>>,
}

/// Literal values in patterns.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum LiteralPattern {
    /// Integer literal: `42`, `-1`
    Int(i64),
    /// Float literal: `3.14`
    Float(FloatBits),
    /// String literal: `"hello"`
    String(String),
    /// Boolean literal: `true`, `false`
    Bool(bool),
    /// Unit literal: `()`
    Unit,
}

// ============================================================================
// Phase-specific type aliases
// ============================================================================

use super::phases::{ResolvedRef, TypedRef, UnresolvedName};

/// Pattern after parsing, before name resolution.
pub type ParsedPattern = Pattern<UnresolvedName>;

/// Pattern after name resolution.
pub type ResolvedPattern<'db> = Pattern<ResolvedRef<'db>>;

/// Pattern after type checking.
pub type TypedPattern<'db> = Pattern<TypedRef<'db>>;
