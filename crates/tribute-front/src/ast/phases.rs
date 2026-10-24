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

use trunk_ir::{Symbol, SymbolPath};

use super::decl::OpDeclKind;
use super::expr::Candidates;
use super::node_id::NodeId;
use super::types::Type;

// ============================================================================
// Phase 1: Unresolved (after parsing)
// ============================================================================

/// A name reference as it appears in source code, before resolution.
///
/// At this stage, we don't know whether `foo` refers to a local variable,
/// a function, a constructor, or something else.
///
/// The path holds one component per segment the source wrote:
///
/// - `foo` → `[foo]`
/// - `State::get` → `[State, get]`
/// - `a::b::c` → `[a, b, c]`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct UnresolvedName {
    /// The path as written, one component per segment.
    pub path: SymbolPath,
    /// Node ID for looking up the span in SpanMap.
    pub id: NodeId,
    /// Whether the name is an implicit constructor `.Name`, whose type its
    /// context gives (`new-plans/type-inference.md` § 암묵적 생성자).
    pub implicit: bool,
}

impl UnresolvedName {
    /// Create a new name reference (simple or qualified).
    pub fn new(path: SymbolPath, id: NodeId) -> Self {
        Self {
            path,
            id,
            implicit: false,
        }
    }

    /// Create the implicit constructor `.name`.
    pub fn implicit(name: Symbol, id: NodeId) -> Self {
        Self {
            path: SymbolPath::new([name]),
            id,
            implicit: true,
        }
    }

    /// Returns true if this is a simple (unqualified) name.
    pub fn is_simple(&self) -> bool {
        self.path.is_simple()
    }

    /// Returns the final name segment (e.g. `println` from `std::io::println`).
    pub fn name(&self) -> Symbol {
        self.path
            .leaf()
            .expect("a name reference has a segment")
            .clone()
    }

    /// Returns the namespace prefix, if any (e.g. `std::io` from `std::io::println`).
    pub fn namespace(&self) -> Option<SymbolPath> {
        self.path.parent().filter(|namespace| !namespace.is_root())
    }
}

impl Display for UnresolvedName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.implicit {
            f.write_str(".")?;
        }
        Display::fmt(&self.path, f)
    }
}

// ============================================================================
// Phase 2: Resolved (after name resolution)
// ============================================================================

/// A unique identifier for a local variable binding.
///
/// LocalIds are assigned during name resolution and are unique within one
/// module-level function, including its lambdas and handler arms. Each such
/// function numbers its own, so an edit to one function leaves the others'
/// LocalIds unchanged. They provide stable identity for variables even if
/// the same name is shadowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
/// FuncDefId is interned (not tracked) so that the same qualified name
/// always produces the same FuncDefId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct FuncDefId<'db> {
    /// The enclosing modules' names, then the function's own (e.g.
    /// `[foo, bar, func_name]`).
    pub path: SymbolPath,
}

impl<'db> FuncDefId<'db> {
    /// The path's components joined with `::`, for IR symbols and messages.
    pub fn qualified(self, db: &'db dyn salsa::Database) -> Symbol {
        self.path(db).to_symbol()
    }

    /// Returns the unqualified function name (last segment).
    pub fn name(self, db: &'db dyn salsa::Database) -> Symbol {
        definition_name(self.path(db))
    }
}

/// The own name of the definition at `path`.
fn definition_name(path: &SymbolPath) -> Symbol {
    path.leaf()
        .expect("a definition's path has its name")
        .clone()
}

/// Compiler-owned nominal types whose identity cannot be supplied by source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    List,
    /// The runtime module's fixed-size immutable array `Slots(a)`.
    Slots,
    /// The runtime module's handler dispatch closure `Dispatch`.
    Dispatch,
}

/// The origin of a nominal type definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TypeOrigin {
    /// A source declaration, identified by its path and by how many type
    /// declarations of the same name precede it in its module. The path
    /// includes the declaring package. Reparsing the same declaration, or
    /// editing another, gives the same identity unless the edit changes
    /// the declaration's path or the same-named declarations before it.
    Source { disambiguator: u32 },
    /// A compiler-owned nominal type.
    Builtin(BuiltinType),
    /// A synthetic identity used by isolated tests and pre-resolution helpers.
    Synthetic,
}

/// A unique identifier for a nominal type definition.
///
/// TypeDefId identifies the type definition itself, not its constructors.
///
/// Source and compiler-owned types remain distinct even when their display
/// names are identical.
#[salsa::interned(debug)]
pub struct TypeDefId<'db> {
    #[returns(copy)]
    pub origin: TypeOrigin,
    /// The enclosing modules' names, then the type's own (e.g.
    /// `[std, Option]`).
    pub path: SymbolPath,
}

impl<'db> TypeDefId<'db> {
    /// The source declaration at `path` that `disambiguator` type
    /// declarations of the same name precede.
    pub fn declared(db: &'db dyn salsa::Database, path: SymbolPath, disambiguator: u32) -> Self {
        Self::new(db, TypeOrigin::Source { disambiguator }, path)
    }

    /// The source type a name at `path` refers to: the first declaration
    /// there.
    pub fn source(db: &'db dyn salsa::Database, path: SymbolPath) -> Self {
        Self::declared(db, path, 0)
    }

    pub fn synthetic(db: &'db dyn salsa::Database, path: SymbolPath) -> Self {
        Self::new(db, TypeOrigin::Synthetic, path)
    }

    pub fn builtin_list(db: &'db dyn salsa::Database) -> Self {
        Self::new(
            db,
            TypeOrigin::Builtin(BuiltinType::List),
            SymbolPath::new(["List"]),
        )
    }

    pub fn builtin_slots(db: &'db dyn salsa::Database) -> Self {
        Self::new(
            db,
            TypeOrigin::Builtin(BuiltinType::Slots),
            SymbolPath::new(["std", "runtime", "Slots"]),
        )
    }

    pub fn builtin_dispatch(db: &'db dyn salsa::Database) -> Self {
        Self::new(
            db,
            TypeOrigin::Builtin(BuiltinType::Dispatch),
            SymbolPath::new(["std", "runtime", "Dispatch"]),
        )
    }

    /// The type of the same origin at `path`.
    pub fn with_path(self, db: &'db dyn salsa::Database, path: SymbolPath) -> Self {
        Self::new(db, self.origin(db), path)
    }

    /// The path's components joined with `::`, for IR symbols and messages.
    pub fn qualified(self, db: &'db dyn salsa::Database) -> Symbol {
        self.path(db).to_symbol()
    }

    pub fn is_builtin_list(self, db: &'db dyn salsa::Database) -> bool {
        self.origin(db) == TypeOrigin::Builtin(BuiltinType::List)
    }

    pub fn is_builtin_slots(self, db: &'db dyn salsa::Database) -> bool {
        self.origin(db) == TypeOrigin::Builtin(BuiltinType::Slots)
    }

    pub fn is_builtin_dispatch(self, db: &'db dyn salsa::Database) -> bool {
        self.origin(db) == TypeOrigin::Builtin(BuiltinType::Dispatch)
    }

    /// Returns the unqualified type name (last segment).
    pub fn name(self, db: &'db dyn salsa::Database) -> Symbol {
        definition_name(self.path(db))
    }
}

/// A unique identifier for a constructor (enum variant or struct constructor).
///
/// CtorId identifies a specific constructor, not the type itself.
/// For structs, the struct name is both the type and its constructor.
/// For enums, each variant has its own CtorId.
///
/// CtorId is interned (not tracked) so that the same qualified name
/// always produces the same CtorId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct CtorId<'db> {
    /// The enclosing modules' names, then the constructor's own (e.g.
    /// `[std, Some]`).
    pub path: SymbolPath,
}

impl<'db> CtorId<'db> {
    /// The path's components joined with `::`, for IR symbols and messages.
    pub fn qualified(self, db: &'db dyn salsa::Database) -> Symbol {
        self.path(db).to_symbol()
    }

    /// Returns the unqualified constructor name (last segment).
    pub fn name(self, db: &'db dyn salsa::Database) -> Symbol {
        definition_name(self.path(db))
    }
}

/// Compiler-owned abilities with semantics that source declarations cannot request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinAbility {
    Io,
}

/// The origin of an ability identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbilityOrigin {
    Source,
    Builtin(BuiltinAbility),
}

/// A unique identifier for an ability definition.
///
/// Source and compiler-owned builtin abilities remain distinct even when their
/// display paths are the same.
///
/// AbilityId is interned (not tracked) so that the same origin and qualified
/// name always produce the same AbilityId, regardless of where it's created.
#[salsa::interned(debug)]
pub struct AbilityId<'db> {
    #[returns(copy)]
    pub origin: AbilityOrigin,
    /// The enclosing modules' names, then the ability's own (e.g.
    /// `[std, state, State]`).
    pub path: SymbolPath,
}

/// The path of the compiler-owned `Io` ability.
const BUILTIN_IO: [&str; 3] = ["std", "io", "Io"];

impl<'db> AbilityId<'db> {
    pub fn source(db: &'db dyn salsa::Database, path: SymbolPath) -> Self {
        Self::new(db, AbilityOrigin::Source, path)
    }

    pub fn builtin_io(db: &'db dyn salsa::Database) -> Self {
        Self::new(
            db,
            AbilityOrigin::Builtin(BuiltinAbility::Io),
            SymbolPath::new(BUILTIN_IO),
        )
    }

    /// Look up a compiler-owned ability by its source path segments.
    pub fn builtin_from_path(db: &'db dyn salsa::Database, path: &[Symbol]) -> Option<Self> {
        path.iter()
            .map(Symbol::as_str)
            .eq(BUILTIN_IO)
            .then(|| Self::builtin_io(db))
    }

    /// Resolve a fully-qualified source path through the builtin registry.
    pub fn from_resolved_path(db: &'db dyn salsa::Database, path: SymbolPath) -> Self {
        Self::builtin_from_path(db, &path).unwrap_or_else(|| Self::source(db, path))
    }

    /// The path's components joined with `::`, for IR symbols and messages.
    pub fn qualified(self, db: &'db dyn salsa::Database) -> Symbol {
        self.path(db).to_symbol()
    }

    pub fn is_builtin_io(self, db: &'db dyn salsa::Database) -> bool {
        self.origin(db) == AbilityOrigin::Builtin(BuiltinAbility::Io)
    }

    /// Returns the unqualified ability name (last segment).
    pub fn name(self, db: &'db dyn salsa::Database) -> Symbol {
        definition_name(self.path(db))
    }
}

/// Module path reference for qualified imports.
#[salsa::tracked(debug)]
pub struct ModulePath<'db> {
    /// The path segments (e.g., ["std", "collections", "List"])
    #[returns(deref)]
    pub segments: Vec<Symbol>,
}

/// A resolved reference to a definition.
///
/// After name resolution, we know exactly what each name refers to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
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

    /// An implicit constructor `.Name`, which type checking resolves to a
    /// [`Constructor`](Self::Constructor) of the type its context gives,
    /// falling back to `candidates`: the constructors `Name` of the types in
    /// scope, by rank (`new-plans/modules.md` § 암묵적 생성자의 후보).
    ImplicitConstructor {
        name: Symbol,
        candidates: Candidates<CtorId<'db>>,
        /// The constructors `Name` the current module does not see, with
        /// why, which its context's type may not name either.
        hidden: Vec<(CtorId<'db>, String)>,
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

    /// Reference to a module (for qualified paths).
    Module { path: ModulePath<'db> },

    /// Reference to an ability operation.
    ///
    /// Ability operations like `State::get()` are resolved to this variant,
    /// which is lowered to `ability.perform` with CPS continuations.
    AbilityOp {
        /// The ability identifier (e.g., AbilityId for "State").
        ability: AbilityId<'db>,
        /// The operation name (e.g., "get").
        op: Symbol,
        /// Whether this is a `fn` (tail-resumptive) or `op` (general) operation.
        kind: OpDeclKind,
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

    /// Create an ability operation reference.
    pub fn ability_op(ability: AbilityId<'db>, op: Symbol, kind: OpDeclKind) -> Self {
        Self::AbilityOp { ability, op, kind }
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
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

/// Generator for the LocalIds of one module-level function.
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

    #[test]
    fn builtin_and_source_abilities_with_the_same_path_are_distinct() {
        let db = salsa::DatabaseImpl::default();
        let builtin = AbilityId::builtin_io(&db);
        let source = AbilityId::source(&db, trunk_ir::SymbolPath::new(["std", "io", "Io"]));

        assert_ne!(builtin, source);
        assert!(builtin.is_builtin_io(&db));
        assert!(!source.is_builtin_io(&db));
    }
}
