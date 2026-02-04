//! Compilation phase markers for the AST.
//!
//! This module defines types that represent different stages of name resolution:
//! - `UnresolvedName`: A name as parsed from source (phase 1)
//! - `ResolvedRef`: A reference to a resolved definition (phase 2)
//! - `TypedRef`: A resolved reference with type information (phase 3)
//!
//! The AST uses a generic parameter `V` that varies by phase, allowing
//! the same structure to represent the AST at different compilation stages.

use std::fmt::{self, Display, Formatter};

use trunk_ir::{Symbol, SymbolVec};

use super::node_id::NodeId;
use super::types::Type;

// ============================================================================
// Helper types
// ============================================================================

/// A qualified name that can be displayed efficiently without intermediate allocations.
struct QualifiedName<'a> {
    module_path: &'a SymbolVec,
    name: Symbol,
}

impl Display for QualifiedName<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for segment in self.module_path.iter() {
            if !first {
                f.write_str("::")?;
            }
            first = false;
            segment.with_str(|s| f.write_str(s))?;
        }
        if !first {
            f.write_str("::")?;
        }
        self.name.with_str(|s| f.write_str(s))
    }
}

// ============================================================================
// Phase 1: Unresolved (after parsing)
// ============================================================================

/// A name reference as it appears in source code, before resolution.
///
/// At this stage, we don't know whether `foo` refers to a local variable,
/// a function, a constructor, or something else.
///
/// # Examples
///
/// - `foo` → `{ module_path: [], name: "foo" }`
/// - `State::get` → `{ module_path: ["State"], name: "get" }`
/// - `a::b::c` → `{ module_path: ["a", "b"], name: "c" }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct UnresolvedName {
    /// The module path prefix (empty for simple names).
    pub module_path: SymbolVec,
    /// The final name segment.
    pub name: Symbol,
    /// Node ID for looking up the span in SpanMap.
    pub id: NodeId,
}

impl UnresolvedName {
    /// Create a new simple (unqualified) name reference.
    pub fn simple(name: Symbol, id: NodeId) -> Self {
        Self {
            module_path: SymbolVec::new(),
            name,
            id,
        }
    }

    /// Create a new qualified name reference.
    pub fn qualified(module_path: SymbolVec, name: Symbol, id: NodeId) -> Self {
        Self {
            module_path,
            name,
            id,
        }
    }

    /// Returns true if this is a simple (unqualified) name.
    pub fn is_simple(&self) -> bool {
        self.module_path.is_empty()
    }
}

impl Display for UnresolvedName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        QualifiedName {
            module_path: &self.module_path,
            name: self.name,
        }
        .fmt(f)
    }
}

// ============================================================================
// Phase 2: Resolved (after name resolution)
// ============================================================================

/// A unique identifier for a local variable binding.
///
/// LocalIds are unique within a function scope and are assigned
/// during name resolution. They provide stable identity for
/// variables even if the same name is shadowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct LocalId(u32);

impl LocalId {
    /// Sentinel value for unresolved names.
    ///
    /// Used when a name cannot be resolved during the name resolution phase.
    /// Later passes should check for this value using [`is_unresolved()`](Self::is_unresolved)
    /// and report appropriate errors.
    pub const UNRESOLVED: LocalId = LocalId(u32::MAX);

    /// Create a new LocalId from a raw value.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw value.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Check if this is an unresolved sentinel value.
    pub const fn is_unresolved(self) -> bool {
        self.0 == u32::MAX
    }
}

/// A unique identifier for a function definition.
///
/// FuncDefId is interned (not tracked) so that the same (module_path, name)
/// always produces the same FuncDefId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct FuncDefId<'db> {
    /// The module path (e.g., ["foo", "bar"]).
    #[returns(ref)]
    pub module_path: SymbolVec,
    /// The function name.
    pub name: Symbol,
}

impl<'db> FuncDefId<'db> {
    /// Build a qualified name for IR generation.
    ///
    /// Returns a displayable qualified name like "foo::bar::func_name".
    pub fn qualified_name(self, db: &'db dyn salsa::Database) -> impl Display + 'db {
        QualifiedName {
            module_path: self.module_path(db),
            name: self.name(db),
        }
    }
}

/// A unique identifier for a type definition (struct or enum).
///
/// TypeDefId identifies the type definition itself, not its constructors.
/// For example, `Option` as a type has one TypeDefId, while its constructors
/// `Some` and `None` each have their own CtorId.
///
/// TypeDefId is interned (not tracked) so that the same (module_path, name)
/// always produces the same TypeDefId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct TypeDefId<'db> {
    /// The module path (e.g., ["std", "option"]).
    #[returns(ref)]
    pub module_path: SymbolVec,
    /// The type name (enum or struct name).
    pub name: Symbol,
}

impl<'db> TypeDefId<'db> {
    /// Build a qualified name for IR generation.
    ///
    /// Returns a displayable qualified name like "std::option::Option".
    pub fn qualified_name(self, db: &'db dyn salsa::Database) -> impl Display + 'db {
        QualifiedName {
            module_path: self.module_path(db),
            name: self.name(db),
        }
    }
}

/// A unique identifier for a constructor (enum variant or struct constructor).
///
/// CtorId identifies a specific constructor, not the type itself.
/// For structs, the struct name is both the type and its constructor.
/// For enums, each variant has its own CtorId.
///
/// CtorId is interned (not tracked) so that the same (module_path, ctor_name)
/// always produces the same CtorId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct CtorId<'db> {
    /// The module path (e.g., ["std", "option"]).
    #[returns(ref)]
    pub module_path: SymbolVec,
    /// The constructor name (struct name or enum variant name).
    pub ctor_name: Symbol,
}

impl<'db> CtorId<'db> {
    /// Build a qualified name for IR generation.
    ///
    /// Returns a displayable qualified name like "std::option::Some".
    pub fn qualified_name(self, db: &'db dyn salsa::Database) -> impl Display + 'db {
        QualifiedName {
            module_path: self.module_path(db),
            name: self.ctor_name(db),
        }
    }
}

/// A unique identifier for an ability definition.
///
/// AbilityId identifies an ability (effect) definition with its module path.
/// For example, `State` ability in module `foo::bar` would have
/// module_path `["foo", "bar"]` and name `"State"`.
///
/// AbilityId is interned (not tracked) so that the same (module_path, name)
/// always produces the same AbilityId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct AbilityId<'db> {
    /// The module path (e.g., ["std", "state"]).
    #[returns(ref)]
    pub module_path: SymbolVec,
    /// The ability name.
    pub name: Symbol,
}

impl<'db> AbilityId<'db> {
    /// Build a qualified name for IR generation.
    ///
    /// Returns a displayable qualified name like "std::state::State".
    pub fn qualified_name(self, db: &'db dyn salsa::Database) -> impl Display + 'db {
        QualifiedName {
            module_path: self.module_path(db),
            name: self.name(db),
        }
    }
}

/// Reference to a builtin operation or value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum BuiltinRef {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // Comparison operations
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Boolean operations
    And,
    Or,
    Not,

    // String operations
    Concat,

    // List operations
    Cons,
    ListConcat,

    // IO operations
    Print,
    ReadLine,
}

/// Module path reference for qualified imports.
#[salsa::tracked(debug)]
pub struct ModulePath<'db> {
    /// The path segments (e.g., ["std", "collections", "List"])
    #[returns(ref)]
    pub segments: Vec<Symbol>,
}

/// A resolved reference to a definition.
///
/// After name resolution, we know exactly what each name refers to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum ResolvedRef<'db> {
    /// Reference to a local variable (function parameter or let binding).
    Local {
        /// Unique identifier for the local variable.
        id: LocalId,
        /// The variable name (for error messages).
        name: Symbol,
    },

    /// Reference to a function definition.
    Function { id: FuncDefId<'db> },

    /// Reference to a constructor (enum variant or struct).
    Constructor {
        /// The constructor ID.
        id: CtorId<'db>,
        /// The variant name (for enum variants) or type name (for structs).
        variant: Symbol,
    },

    /// Reference to a type definition (struct or enum).
    ///
    /// This variant is used when a type name is referenced in expression context
    /// without being resolved to a specific constructor. For structs, this is
    /// typically overwritten by the Constructor binding. For enums, using the
    /// enum name in expression context (e.g., `Option`) will resolve to this
    /// variant, which should result in an error during type checking.
    TypeDef {
        /// The type definition ID.
        id: TypeDefId<'db>,
    },

    /// Reference to a builtin operation.
    Builtin(BuiltinRef),

    /// Reference to a module (for qualified paths).
    Module { path: ModulePath<'db> },

    /// Reference to an ability operation.
    ///
    /// Ability operations like `State::get()` are resolved to this variant,
    /// which is lowered directly to `cont.shift` + runtime calls.
    AbilityOp {
        /// The ability identifier (e.g., AbilityId for "State").
        ability: AbilityId<'db>,
        /// The operation name (e.g., "get").
        op: Symbol,
    },

    /// Reference to an ability definition.
    ///
    /// Used in handler patterns to identify which ability is being handled.
    Ability {
        /// The ability identifier.
        id: AbilityId<'db>,
    },
}

impl<'db> ResolvedRef<'db> {
    /// Create a local variable reference.
    pub fn local(id: LocalId, name: Symbol) -> Self {
        Self::Local { id, name }
    }

    /// Create a function reference.
    pub fn function(id: FuncDefId<'db>) -> Self {
        Self::Function { id }
    }

    /// Create a constructor reference.
    pub fn constructor(id: CtorId<'db>, variant: Symbol) -> Self {
        Self::Constructor { id, variant }
    }

    /// Create a type definition reference.
    pub fn type_def(id: TypeDefId<'db>) -> Self {
        Self::TypeDef { id }
    }

    /// Create a builtin reference.
    pub fn builtin(builtin: BuiltinRef) -> Self {
        Self::Builtin(builtin)
    }

    /// Create an ability operation reference.
    pub fn ability_op(ability: AbilityId<'db>, op: Symbol) -> Self {
        Self::AbilityOp { ability, op }
    }

    /// Create an ability reference.
    pub fn ability(id: AbilityId<'db>) -> Self {
        Self::Ability { id }
    }
}

// ============================================================================
// Phase 3: Typed (after type checking)
// ============================================================================

/// A resolved reference with type information.
///
/// After type checking, every reference has a known type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct TypedRef<'db> {
    /// The resolved reference.
    pub resolved: ResolvedRef<'db>,
    /// The inferred or checked type.
    pub ty: Type<'db>,
}

impl<'db> TypedRef<'db> {
    /// Create a new typed reference.
    pub fn new(resolved: ResolvedRef<'db>, ty: Type<'db>) -> Self {
        Self { resolved, ty }
    }
}

// ============================================================================
// Generator for LocalIds
// ============================================================================

/// Generator for unique LocalIds within a scope.
#[derive(Debug, Default)]
pub struct LocalIdGen(u32);

impl LocalIdGen {
    /// Create a new generator.
    pub fn new() -> Self {
        Self(0)
    }

    /// Generate a fresh unique LocalId.
    ///
    /// # Panics
    /// Panics if the counter would overflow and produce `LocalId::UNRESOLVED`.
    pub fn fresh(&mut self) -> LocalId {
        let id = LocalId(self.0);
        self.0 = self.0.checked_add(1).expect("LocalId overflow");
        debug_assert!(!id.is_unresolved(), "LocalIdGen produced UNRESOLVED");
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_id_unresolved_constant() {
        assert_eq!(LocalId::UNRESOLVED.raw(), u32::MAX);
    }

    #[test]
    fn test_local_id_is_unresolved() {
        assert!(LocalId::UNRESOLVED.is_unresolved());
        assert!(!LocalId::new(0).is_unresolved());
        assert!(!LocalId::new(100).is_unresolved());
    }

    #[test]
    fn test_local_id_gen_does_not_produce_unresolved() {
        let mut id_gen = LocalIdGen::new();
        // Generate many IDs and ensure none are UNRESOLVED
        for _ in 0..1000 {
            let id = id_gen.fresh();
            assert!(
                !id.is_unresolved(),
                "LocalIdGen should not produce UNRESOLVED values"
            );
        }
    }
}
