//! Type system types for the AST.
//!
//! This module defines the core type representations used during compilation:
//! - `Type`: Monomorphic types (concrete types, type variables)
//! - `TypeScheme`: Polymorphic types with universally quantified type parameters
//!
//! The distinction between Type and TypeScheme is important for type inference:
//! - Functions are stored with TypeSchemes (they can be polymorphic)
//! - During type checking, TypeSchemes are instantiated to Types with fresh type variables

mod display;
pub mod fold;

use std::borrow::Cow;
use std::convert::Infallible;
use std::ops::ControlFlow;
use trunk_ir::SymbolPath;

use trunk_ir::Symbol;

use super::NodeId;
use super::phases::{FuncDefId, TypeDefId};
use fold::{TypeVisitable, TypeVisitor};

/// A monomorphic type.
///
/// Types represent concrete types during type checking. Type variables
/// (`UniVar`) are used for unknowns during inference and are resolved
/// by unification.
#[salsa::interned(debug)]
pub struct Type<'db> {
    pub kind: TypeKind<'db>,
}

/// Source of a unification variable.
///
/// This identifies where a type variable came from, enabling unique identification
/// across functions and aiding debugging.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum UniVarSource<'db> {
    /// Type variable created within a specific function.
    /// - `func_id`: The function definition ID (provides unique identification)
    /// - `index`: Local index within that function (0, 1, 2, ...)
    FunctionLocal { func_id: FuncDefId<'db>, index: u64 },

    /// Anonymous type variable (for tests or contexts without a function name).
    /// The u64 is a unique counter value.
    Anonymous(u64),

    /// Type variable created during post-solve deferred method resolution.
    /// Distinct from FunctionLocal/Anonymous to prevent aliasing.
    Solver { index: u64 },
}

/// A unification variable ID.
///
/// Each instantiation of a polymorphic type scheme creates fresh UniVarIds
/// so that different call sites get independent type variables for unification.
#[salsa::interned(debug)]
pub struct UniVarId<'db> {
    /// The source of this type variable.
    #[returns(copy)]
    pub source: UniVarSource<'db>,
    /// Index within the type scheme's parameters (0, 1, 2, ...).
    #[returns(copy)]
    pub index: u32,
}

/// The different kinds of types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum TypeKind<'db> {
    // === Primitive types ===
    /// Signed integer type (arbitrary precision)
    Int,
    /// Natural number type (non-negative integers)
    Nat,
    /// 64-bit floating point
    Float,
    /// Boolean type
    Bool,
    /// Byte sequence type
    Bytes,
    /// Unicode code point (i32)
    Rune,
    /// Unit type (empty tuple)
    Nil,
    /// Bottom type (no values, indicates divergence)
    Never,

    // === Type variables ===
    /// Bound type variable (De Bruijn index within a TypeScheme).
    ///
    /// Index 0 refers to the innermost binder, following De Bruijn convention.
    BoundVar { index: u32 },

    /// Type variable quantified by a local `let` binding.
    ///
    /// Local binders have no enclosing `TypeScheme`, so their owner must remain
    /// part of typed-body metadata instead of being confused with a function
    /// scheme's `BoundVar`.
    LocalBoundVar { scope: NodeId, index: u32 },

    /// Unification variable (unknown during inference).
    ///
    /// These are created during type checking and resolved by unification.
    /// For polymorphic instantiation (functions/constructors), the ID is
    /// deterministic based on the source, enabling Salsa caching.
    UniVar { id: UniVarId<'db> },

    // === Compound types ===
    /// Named type (struct, enum, or type alias) with optional type arguments.
    Named {
        /// Resolved nominal declaration identity.
        id: TypeDefId<'db>,
        /// The type name (may be qualified path like `std::List`)
        name: Symbol,
        /// Type arguments for generic types
        args: Vec<Type<'db>>,
    },

    /// Function type with parameters, result, and ability row.
    Func {
        params: Vec<Type<'db>>,
        result: Type<'db>,
        effect: AbilityRow<'db>,
    },

    /// Tuple type.
    Tuple(Vec<Type<'db>>),

    /// Continuation type for effect handlers.
    ///
    /// Represents a captured continuation that can be resumed with a value.
    /// - `arg`: The type of value passed when resuming
    /// - `result`: The type returned after resuming
    /// - `effect`: The ability row of the continuation body
    Continuation {
        arg: Type<'db>,
        result: Type<'db>,
        effect: AbilityRow<'db>,
    },

    /// Error type (used when type checking fails).
    ///
    /// This propagates through the type system to prevent cascading errors.
    Error,
}

impl<'db> TypeKind<'db> {
    /// Returns the canonical name for primitive types, or `None` for compound/variable types.
    pub fn primitive_name(&self) -> Option<&'static str> {
        match self {
            Self::Int => Some("Int"),
            Self::Nat => Some("Nat"),
            Self::Float => Some("Float"),
            Self::Bool => Some("Bool"),
            Self::Bytes => Some("Bytes"),
            Self::Rune => Some("Rune"),
            Self::Nil => Some("Nil"),
            Self::Never => Some("Never"),
            _ => None,
        }
    }

    /// Create a synthetic String type for isolated contexts.
    pub fn string(db: &'db dyn salsa::Database) -> Self {
        let name = Symbol::new("String");
        Self::Named {
            id: TypeDefId::synthetic(db, SymbolPath::new([name.clone()])),
            name,
            args: vec![],
        }
    }

    /// Returns the `TypeKind` for a primitive type name, or `None` if the name
    /// is not a primitive.
    pub fn from_primitive_name(name: &str) -> Option<Self> {
        match name {
            "Int" => Some(Self::Int),
            "Nat" => Some(Self::Nat),
            "Float" => Some(Self::Float),
            "Bool" => Some(Self::Bool),
            "Bytes" => Some(Self::Bytes),
            "Rune" => Some(Self::Rune),
            "Nil" => Some(Self::Nil),
            "Never" => Some(Self::Never),
            _ => None,
        }
    }
}

/// A polymorphic type scheme with universally quantified type parameters.
///
/// TypeSchemes represent types that can be instantiated with different type arguments.
/// For example, `fn identity(x: a) -> a` has the scheme `forall a. a -> a`.
#[salsa::interned(debug, constructor = intern)]
pub struct TypeScheme<'db> {
    /// Type parameters (universally quantified).
    ///
    /// The order matters: `type_params[0]` corresponds to `BoundVar { index: 0 }`.
    #[returns(deref)]
    pub type_params: Vec<TypeParam>,
    /// Ability-row variables quantified by this scheme.
    #[returns(deref)]
    pub effect_params: Vec<RowVar>,
    /// Retained exact effect unions, quantified together with the body.
    #[returns(deref)]
    pub row_unions: Vec<RowUnion<'db>>,
    #[returns(deref)]
    pub row_removals: Vec<RowRemoval<'db>>,
    /// The body type with BoundVar references to type_params.
    #[returns(copy)]
    pub body: Type<'db>,
}

/// An exact set union retained through inference and generalization.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct RowUnion<'db> {
    pub sources: Vec<AbilityRow<'db>>,
    pub result: AbilityRow<'db>,
}

/// Exact handler subtraction, retained while source tails or type arguments are open.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct RowRemoval<'db> {
    pub source: AbilityRow<'db>,
    pub removed: AbilityRow<'db>,
    pub result: AbilityRow<'db>,
}

impl<'db> RowRemoval<'db> {
    pub fn rows(&self) -> [AbilityRow<'db>; 3] {
        [self.source, self.removed, self.result]
    }

    /// Visit source, removed, then result, updating their row handles in place.
    pub fn for_each_row_mut(&mut self, mut f: impl FnMut(&mut AbilityRow<'db>)) {
        f(&mut self.source);
        f(&mut self.removed);
        f(&mut self.result);
    }
}

impl<'db> RowUnion<'db> {
    /// Visit sources in order, then result, reusing the source vector.
    pub fn for_each_row_mut(&mut self, mut f: impl FnMut(&mut AbilityRow<'db>)) {
        for source in &mut self.sources {
            f(source);
        }
        f(&mut self.result);
    }
}

impl<'db> TypeScheme<'db> {
    pub fn new(
        db: &'db dyn salsa::Database,
        type_params: Vec<TypeParam>,
        effect_params: Vec<RowVar>,
        body: Type<'db>,
    ) -> Self {
        Self::builder(type_params, effect_params, body).build(db)
    }

    pub fn builder(
        type_params: Vec<TypeParam>,
        effect_params: Vec<RowVar>,
        body: Type<'db>,
    ) -> TypeSchemeBuilder<'db> {
        TypeSchemeBuilder {
            type_params: Cow::Owned(type_params),
            effect_params: Cow::Owned(effect_params),
            row_unions: Cow::Borrowed(&[]),
            row_removals: Cow::Borrowed(&[]),
            body,
        }
    }

    /// Start rewriting the complete scheme before publication. The builder
    /// borrows the interned fields and copies only those it changes.
    pub fn to_builder(self, db: &'db dyn salsa::Database) -> TypeSchemeBuilder<'db> {
        TypeSchemeBuilder {
            type_params: Cow::Borrowed(self.type_params(db)),
            effect_params: Cow::Borrowed(self.effect_params(db)),
            row_unions: Cow::Borrowed(self.row_unions(db)),
            row_removals: Cow::Borrowed(self.row_removals(db)),
            body: self.body(db),
        }
    }

    /// Visit the body and each type argument in retained row constraints.
    /// The caller owns recursive traversal within each type.
    pub fn for_each_type(self, db: &'db dyn salsa::Database, visit: impl FnMut(Type<'db>)) {
        let ControlFlow::Continue(()) = self.visit_with(&mut fold::ForEachType { db, visit });
    }

    /// Create a monomorphic scheme (no quantified variables).
    pub fn mono(db: &'db dyn salsa::Database, ty: Type<'db>) -> Self {
        Self::new(db, Vec::new(), Vec::new(), ty)
    }

    /// Check if this scheme has no quantified variables (is monomorphic).
    pub fn is_mono(&self, db: &'db dyn salsa::Database) -> bool {
        self.type_params(db).is_empty() && self.effect_params(db).is_empty()
    }

    /// Get the number of type parameters.
    pub fn arity(&self, db: &'db dyn salsa::Database) -> usize {
        self.type_params(db).len()
    }
}

/// Mutable construction data; only `build` interns a scheme.
#[must_use]
pub struct TypeSchemeBuilder<'db> {
    type_params: Cow<'db, [TypeParam]>,
    effect_params: Cow<'db, [RowVar]>,
    row_unions: Cow<'db, [RowUnion<'db>]>,
    row_removals: Cow<'db, [RowRemoval<'db>]>,
    body: Type<'db>,
}

impl<'db> TypeSchemeBuilder<'db> {
    pub fn type_params(mut self, params: Vec<TypeParam>) -> Self {
        self.type_params = Cow::Owned(params);
        self
    }

    pub fn row_unions(mut self, unions: Vec<RowUnion<'db>>) -> Self {
        self.row_unions = Cow::Owned(unions);
        self
    }

    pub fn row_removals(mut self, removals: Vec<RowRemoval<'db>>) -> Self {
        self.row_removals = Cow::Owned(removals);
        self
    }

    /// Transform the body and every type argument in retained constraints.
    /// The callback owns recursive traversal within each type; binder identities
    /// and constraint order remain unchanged. Borrowed constraint lists are
    /// copied only when a row changes; interned rows remain immutable.
    pub fn map_types(
        self,
        db: &'db dyn salsa::Database,
        map: impl FnMut(Type<'db>) -> Type<'db>,
    ) -> Self {
        let Ok(builder) = self.try_fold_types(&mut fold::MapTypes { db, map });
        builder
    }

    pub fn build(self, db: &'db dyn salsa::Database) -> TypeScheme<'db> {
        TypeScheme::intern(
            db,
            self.type_params,
            self.effect_params,
            self.row_unions,
            self.row_removals,
            self.body,
        )
    }
}

/// A type parameter in a type scheme.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct TypeParam {
    /// Optional name for the type parameter (for error messages).
    pub name: Option<Symbol>,
}

impl TypeParam {
    /// Create an unnamed type parameter.
    pub fn anonymous() -> Self {
        Self { name: None }
    }

    /// Create a named type parameter.
    pub fn named(name: Symbol) -> Self {
        Self { name: Some(name) }
    }
}

/// The ability row of a function type: the ability instances a function may
/// use.
#[salsa::interned(debug)]
pub struct AbilityRow<'db> {
    /// The ability instances the row names.
    #[returns(deref)]
    pub instances: Vec<AbilityInstance<'db>>,
    /// The row variable of an open row.
    ///
    /// If Some, the row may hold further unknown instances.
    /// If None, the row is closed: exactly the listed instances.
    #[returns(copy)]
    pub rest: Option<RowVar>,
}

impl<'db> AbilityRow<'db> {
    /// Create an empty ability row (pure function).
    pub fn pure(db: &'db dyn salsa::Database) -> Self {
        Self::new(db, Vec::new(), None)
    }

    /// Create a row with a single effect.
    pub fn single(db: &'db dyn salsa::Database, effect: AbilityInstance<'db>) -> Self {
        Self::new(db, vec![effect], None)
    }

    /// Create an open row variable (unknown effects).
    pub fn open(db: &'db dyn salsa::Database, var: RowVar) -> Self {
        Self::new(db, Vec::new(), Some(var))
    }

    /// Check if this row is pure (no effects and closed).
    pub fn is_pure(&self, db: &'db dyn salsa::Database) -> bool {
        self.instances(db).is_empty() && self.rest(db).is_none()
    }
}

/// An individual effect (ability).
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct AbilityInstance<'db> {
    /// The ability identifier (module path + name).
    pub ability_id: super::AbilityId<'db>,
    /// Type arguments for parameterized abilities.
    pub args: Vec<Type<'db>>,
}

/// Ability row variable for row-polymorphic effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowVar {
    pub id: u64,
}

/// Collect ability-row variables occurring in a type, preserving first appearance.
pub fn collect_row_vars<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Vec<RowVar> {
    struct RowVars<'db> {
        db: &'db dyn salsa::Database,
        vars: Vec<RowVar>,
    }

    impl<'db> TypeVisitor<'db> for RowVars<'db> {
        type Break = Infallible;

        fn db(&self) -> &'db dyn salsa::Database {
            self.db
        }

        fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<Infallible> {
            row.super_visit_with(self)?;
            if let Some(var) = row.rest(self.db)
                && !self.vars.contains(&var)
            {
                self.vars.push(var);
            }
            ControlFlow::Continue(())
        }
    }

    let mut collector = RowVars {
        db,
        vars: Vec::new(),
    };
    let ControlFlow::Continue(()) = ty.visit_with(&mut collector);
    collector.vars
}

// =========================================================================
// Shared annotation → AbilityRow conversion
// =========================================================================

/// Source origins for the concrete effects produced from an annotation list.
///
/// The IDs have the same order as the concrete effects in the converted row.
/// Row-variable annotations are intentionally omitted. Keeping this metadata
/// outside [`AbilityRow`] preserves the row's semantic identity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AbilityAnnotationOrigins {
    concrete: Vec<NodeId>,
}

impl AbilityAnnotationOrigins {
    /// Find the first duplicate in a resolved ability row and recover both
    /// annotation origins associated with it.
    pub fn find_duplicate<'db>(
        &self,
        db: &'db dyn salsa::Database,
        row: AbilityRow<'db>,
    ) -> Option<DuplicateAbilityAnnotations<'db>> {
        self.find_overlap(db, row, |a, b| a == b)
    }

    /// Find the first annotated effect that `overlaps` an earlier one and
    /// recover both annotation origins. The reported effects are the earlier
    /// effect and every later one that overlaps it.
    pub fn find_overlap<'db>(
        &self,
        db: &'db dyn salsa::Database,
        row: AbilityRow<'db>,
        mut overlaps: impl FnMut(&AbilityInstance<'db>, &AbilityInstance<'db>) -> bool,
    ) -> Option<DuplicateAbilityAnnotations<'db>> {
        let effects = row.instances(db);
        // Solving an open row may append inferred effects after the concrete
        // annotations. Only the prefix produced by the source conversion has
        // annotation origins and can represent duplicate annotations.
        let annotated = effects.iter().zip(&self.concrete);

        for (index, (later, &annotation_id)) in annotated.clone().enumerate() {
            let Some((first, &first_annotation_id)) = annotated
                .clone()
                .take(index)
                .find(|(first, _)| overlaps(first, later))
            else {
                continue;
            };
            let effects = std::iter::once(first.clone())
                .chain(
                    annotated
                        .clone()
                        .skip(index)
                        .filter(|(candidate, _)| overlaps(first, candidate))
                        .map(|(candidate, _)| candidate.clone()),
                )
                .collect();
            return Some(DuplicateAbilityAnnotations {
                effects,
                first_annotation_id,
                duplicate_annotation_id: annotation_id,
            });
        }

        None
    }
}

/// A duplicate effect together with its first and repeated source annotations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuplicateAbilityAnnotations<'db> {
    pub effects: Vec<AbilityInstance<'db>>,
    pub first_annotation_id: NodeId,
    pub duplicate_annotation_id: NodeId,
}

/// Result of source-aware effect-annotation conversion.
#[derive(Clone, Debug)]
pub struct AbilityRowConversion<'db> {
    pub row: AbilityRow<'db>,
    pub origins: AbilityAnnotationOrigins,
}

/// Check whether a symbol name looks like a type variable (starts with lowercase).
pub fn is_type_variable(name: &Symbol) -> bool {
    name.with_str(|s| s.starts_with(|c: char| c.is_ascii_lowercase()))
}

/// The qualified name of the ability an annotation path names; `None` for a
/// row variable such as `e`.
fn ability_path(path: &[Symbol]) -> Option<SymbolPath> {
    path.last()
        .is_some_and(|name| !is_type_variable(name))
        .then(|| SymbolPath::new(path.iter().cloned()))
}

/// Convert a single type annotation to an `AbilityInstance`.
///
/// Returns `None` for annotations that represent row variables (lowercase names, `Infer`)
/// rather than concrete abilities.
///
/// The `prefix` parameter is used to construct the AbilityId. If the ability
/// is referenced without a qualified path, it uses the current module prefix.
pub fn annotation_to_effect<'db>(
    db: &'db dyn salsa::Database,
    annotation: &TypeAnnotation,
    prefix: &SymbolPath,
    convert_type: &mut impl FnMut(&TypeAnnotation) -> Type<'db>,
) -> Option<AbilityInstance<'db>> {
    match &annotation.kind {
        TypeAnnotationKind::Named(name) if !is_type_variable(name) => {
            let ability_id = super::AbilityId::source(db, prefix.child(name.clone()));
            Some(AbilityInstance {
                ability_id,
                args: vec![],
            })
        }
        TypeAnnotationKind::Path(path) => {
            let ability_id = super::AbilityId::from_resolved_path(db, ability_path(path)?);
            Some(AbilityInstance {
                ability_id,
                args: vec![],
            })
        }
        TypeAnnotationKind::App { ctor, args } => {
            let qualified = match &ctor.kind {
                TypeAnnotationKind::Named(n) if !is_type_variable(n) => prefix.child(n.clone()),
                TypeAnnotationKind::Path(path) => ability_path(path)?,
                _ => return None,
            };
            let type_args: Vec<Type<'db>> = args.iter().map(&mut *convert_type).collect();
            let ability_id = match &ctor.kind {
                TypeAnnotationKind::Path(_) => super::AbilityId::from_resolved_path(db, qualified),
                _ => super::AbilityId::source(db, qualified),
            };
            Some(AbilityInstance {
                ability_id,
                args: type_args,
            })
        }
        // Named lowercase, Infer, Error, etc. → not a concrete effect
        _ => None,
    }
}

/// Convert a slice of ability annotations to an `AbilityRow`.
///
/// - Uppercase names / `App` → concrete effects
/// - Lowercase names / `Infer` → open row variable (via `fresh_row_var`)
///
/// The `prefix` parameter is used to construct AbilityIds for unqualified ability names.
pub fn abilities_to_ability_row<'db>(
    db: &'db dyn salsa::Database,
    abilities: &[TypeAnnotation],
    prefix: &SymbolPath,
    convert_type: &mut impl FnMut(&TypeAnnotation) -> Type<'db>,
    fresh_row_var: impl FnOnce() -> RowVar,
) -> AbilityRow<'db> {
    abilities_to_ability_row_with_origins(db, abilities, prefix, convert_type, fresh_row_var).row
}

/// Convert ability annotations while preserving each concrete effect's origin.
///
/// Consumers that only need semantic effect identity should continue to use
/// [`abilities_to_ability_row`].
pub fn abilities_to_ability_row_with_origins<'db>(
    db: &'db dyn salsa::Database,
    abilities: &[TypeAnnotation],
    prefix: &SymbolPath,
    convert_type: &mut impl FnMut(&TypeAnnotation) -> Type<'db>,
    fresh_row_var: impl FnOnce() -> RowVar,
) -> AbilityRowConversion<'db> {
    let mut effects = Vec::new();
    let mut concrete_origins = Vec::new();
    let mut has_row_var = false;

    for ann in abilities {
        match &ann.kind {
            TypeAnnotationKind::Named(name) if is_type_variable(name) => {
                has_row_var = true;
            }
            TypeAnnotationKind::Infer => {
                has_row_var = true;
            }
            _ => {
                if let Some(effect) = annotation_to_effect(db, ann, prefix, convert_type) {
                    effects.push(effect);
                    concrete_origins.push(ann.id);
                }
            }
        }
    }

    let rest = if has_row_var {
        Some(fresh_row_var())
    } else {
        None
    };

    AbilityRowConversion {
        row: AbilityRow::new(db, effects, rest),
        origins: AbilityAnnotationOrigins {
            concrete: concrete_origins,
        },
    }
}

/// Type annotation as written in source code.
///
/// This represents a type before resolution and checking.
/// It may contain unresolved names that need to be looked up.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeAnnotation {
    /// The node ID for span lookup.
    pub id: NodeId,
    /// The kind of type annotation.
    pub kind: TypeAnnotationKind,
}

/// Kinds of type annotations in source code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeAnnotationKind {
    /// A simple type name: `Int`, `Bool`, `MyType`
    Named(Symbol),
    /// A qualified type path: `std::collections::List`
    Path(Vec<Symbol>),
    /// A parameterized type: `List(a)`, `Result(T, E)`
    App {
        ctor: Box<TypeAnnotation>,
        args: Vec<TypeAnnotation>,
    },
    /// A function type: `fn(Int, Int) -> Int`
    ///
    /// The `abilities` field contains the ability row items:
    /// - `vec![]`: explicit pure function (`fn(a) ->{} b`)
    /// - `vec![Infer]`: effect polymorphic (`fn(a) -> b` — no ability row written)
    /// - `vec![Named("State")]`: explicit effects (`fn(a) ->{State} b`)
    /// - `vec![Named("State"), Named("e")]`: mixed (`fn(a) ->{State, e} b`)
    Func {
        params: Vec<TypeAnnotation>,
        result: Box<TypeAnnotation>,
        abilities: Vec<TypeAnnotation>,
    },
    /// A tuple type: `(Int, String)`
    Tuple(Vec<TypeAnnotation>),
    /// Inferred type (omitted annotation): `_`
    Infer,
    /// Error in parsing
    Error,
}
