//! Compilation phase markers for the AST.
//!
//! This module defines types that represent different stages of name resolution:
//! - `UnresolvedName`: A name as parsed from source (phase 1)
//! - `ResolvedRef`: A reference to a resolved definition (phase 2)
//! - `TypedRef`: A resolved reference with type information (phase 3)
//!
//! The AST uses a generic parameter `V` that varies by phase, allowing
//! the same structure to represent the AST at different compilation stages.

use trunk_ir::{Symbol, SymbolVec};

use super::node_id::NodeId;
use super::types::Type;

// ============================================================================
// Phase 1: Unresolved (after parsing)
// ============================================================================

/// A name reference as it appears in source code, before resolution.
///
/// At this stage, we don't know whether `foo` refers to a local variable,
/// a function, a constructor, or something else.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct UnresolvedName {
    /// The name as written in source.
    pub name: Symbol,
    /// Node ID for looking up the span in SpanMap.
    pub id: NodeId,
}

impl UnresolvedName {
    /// Create a new unresolved name reference.
    pub fn new(name: Symbol, id: NodeId) -> Self {
        Self { name, id }
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
    /// Build a qualified name string for IR generation.
    ///
    /// Returns the full qualified name like "foo::bar::func_name".
    pub fn qualified_name(self, db: &'db dyn salsa::Database) -> String {
        let module_path = self.module_path(db);
        if module_path.is_empty() {
            self.name(db).to_string()
        } else {
            let path_str = module_path
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            format!("{}::{}", path_str, self.name(db))
        }
    }
}

/// A unique identifier for a constructor (enum variant or struct).
///
/// CtorId is interned (not tracked) so that the same (module_path, type_name)
/// always produces the same CtorId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct CtorId<'db> {
    /// The module path (e.g., ["std", "option"]).
    #[returns(ref)]
    pub module_path: SymbolVec,
    /// The type name (enum or struct name).
    pub type_name: Symbol,
}

impl<'db> CtorId<'db> {
    /// Build a qualified name string for IR generation.
    ///
    /// Returns the full qualified name like "std::option::Option".
    pub fn qualified_name(self, db: &'db dyn salsa::Database) -> String {
        let module_path = self.module_path(db);
        if module_path.is_empty() {
            self.type_name(db).to_string()
        } else {
            let path_str = module_path
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            format!("{}::{}", path_str, self.type_name(db))
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

    /// Reference to a builtin operation.
    Builtin(BuiltinRef),

    /// Reference to a module (for qualified paths).
    Module { path: ModulePath<'db> },
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

    /// Create a builtin reference.
    pub fn builtin(builtin: BuiltinRef) -> Self {
        Self::Builtin(builtin)
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
