//! Type system types for the AST.
//!
//! This module defines the core type representations used during compilation:
//! - `Type`: Monomorphic types (concrete types, type variables)
//! - `TypeScheme`: Polymorphic types with universally quantified type parameters
//!
//! The distinction between Type and TypeScheme is important for type inference:
//! - Functions are stored with TypeSchemes (they can be polymorphic)
//! - During type checking, TypeSchemes are instantiated to Types with fresh type variables

use trunk_ir::Symbol;

use super::{CtorId, FuncDefId, NodeId};

/// A monomorphic type.
///
/// Types represent concrete types during type checking. Type variables
/// (`UniVar`) are used for unknowns during inference and are resolved
/// by unification.
#[salsa::interned(debug)]
pub struct Type<'db> {
    #[returns(ref)]
    pub kind: TypeKind<'db>,
}

/// Source of a unification variable.
///
/// This identifies where a type variable came from, enabling deterministic
/// ID generation for Salsa memoization of polymorphic instantiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum UniVarSource<'db> {
    /// Type variable from instantiating a polymorphic function.
    Function(FuncDefId<'db>),
    /// Type variable from instantiating a polymorphic constructor.
    Constructor(CtorId<'db>),
    /// Anonymous type variable (for lambdas, local bindings, etc.).
    ///
    /// These don't need caching since they're only created once during type checking.
    /// The u64 is a unique counter value.
    Anonymous(u64),
}

/// A unification variable ID for polymorphic instantiation.
///
/// These are created when instantiating polymorphic type schemes (functions
/// and constructors). The combination of source and index ensures deterministic
/// IDs, enabling Salsa to cache instantiation results.
#[salsa::interned(debug)]
pub struct UniVarId<'db> {
    /// The source of this type variable (function or constructor being instantiated).
    pub source: UniVarSource<'db>,
    /// Index within the type scheme's parameters (0, 1, 2, ...).
    pub index: u32,
}

/// The different kinds of types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
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
    /// UTF-8 string type
    String,
    /// Byte sequence type
    Bytes,
    /// Unicode code point (i32)
    Rune,
    /// Unit type (empty tuple)
    Nil,

    // === Type variables ===
    /// Bound type variable (De Bruijn index within a TypeScheme).
    ///
    /// Index 0 refers to the innermost binder, following De Bruijn convention.
    BoundVar { index: u32 },

    /// Unification variable (unknown during inference).
    ///
    /// These are created during type checking and resolved by unification.
    /// For polymorphic instantiation (functions/constructors), the ID is
    /// deterministic based on the source, enabling Salsa caching.
    UniVar { id: UniVarId<'db> },

    // === Compound types ===
    /// Named type (struct, enum, or type alias) with optional type arguments.
    Named {
        /// The type name (may be qualified path like `std::List`)
        name: Symbol,
        /// Type arguments for generic types
        args: Vec<Type<'db>>,
    },

    /// Function type with parameters, result, and effect row.
    Func {
        params: Vec<Type<'db>>,
        result: Type<'db>,
        effect: EffectRow<'db>,
    },

    /// Tuple type.
    Tuple(Vec<Type<'db>>),

    /// Type application (for higher-kinded types).
    App {
        /// The type constructor (e.g., `List`)
        ctor: Type<'db>,
        /// The type arguments
        args: Vec<Type<'db>>,
    },

    /// Error type (used when type checking fails).
    ///
    /// This propagates through the type system to prevent cascading errors.
    Error,
}

/// A polymorphic type scheme with universally quantified type parameters.
///
/// TypeSchemes represent types that can be instantiated with different type arguments.
/// For example, `fn identity(x: a) -> a` has the scheme `forall a. a -> a`.
#[salsa::interned(debug)]
pub struct TypeScheme<'db> {
    /// Type parameters (universally quantified).
    ///
    /// The order matters: `type_params[0]` corresponds to `BoundVar { index: 0 }`.
    #[returns(ref)]
    pub type_params: Vec<TypeParam>,
    /// The body type with BoundVar references to type_params.
    pub body: Type<'db>,
}

impl<'db> TypeScheme<'db> {
    /// Create a monomorphic scheme (no type parameters).
    pub fn mono(db: &'db dyn salsa::Database, ty: Type<'db>) -> Self {
        Self::new(db, Vec::new(), ty)
    }

    /// Check if this scheme has no type parameters (is monomorphic).
    pub fn is_mono(&self, db: &'db dyn salsa::Database) -> bool {
        self.type_params(db).is_empty()
    }

    /// Get the number of type parameters.
    pub fn arity(&self, db: &'db dyn salsa::Database) -> usize {
        self.type_params(db).len()
    }
}

/// A type parameter in a type scheme.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct TypeParam {
    /// Optional name for the type parameter (for error messages).
    pub name: Option<Symbol>,
    /// Optional kind constraint (for higher-kinded types).
    pub kind: Option<Kind>,
}

impl TypeParam {
    /// Create an unnamed type parameter with default kind.
    pub fn anonymous() -> Self {
        Self {
            name: None,
            kind: None,
        }
    }

    /// Create a named type parameter with default kind.
    pub fn named(name: Symbol) -> Self {
        Self {
            name: Some(name),
            kind: None,
        }
    }
}

/// Kind (type of types) for higher-kinded type support.
///
/// Currently simple, can be extended for full higher-kinded polymorphism.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum Kind {
    /// The kind of concrete types (e.g., Int, Bool, List(Int)).
    Type,
    /// Function kind: `* -> *` for type constructors like `List`.
    Arrow(Box<Kind>, Box<Kind>),
}

/// Effect row for tracking computational effects.
///
/// Effect rows are used in function types to track what effects
/// a function may perform.
#[salsa::interned(debug)]
pub struct EffectRow<'db> {
    /// Known effects in this row.
    #[returns(ref)]
    pub effects: Vec<Effect<'db>>,
    /// Optional row variable for open effect rows.
    ///
    /// If Some, this row can have additional unknown effects.
    /// If None, this is a closed row with exactly the listed effects.
    pub rest: Option<EffectVar>,
}

impl<'db> EffectRow<'db> {
    /// Create an empty effect row (pure function).
    pub fn pure(db: &'db dyn salsa::Database) -> Self {
        Self::new(db, Vec::new(), None)
    }

    /// Create a row with a single effect.
    pub fn single(db: &'db dyn salsa::Database, effect: Effect<'db>) -> Self {
        Self::new(db, vec![effect], None)
    }

    /// Create an open row variable (unknown effects).
    pub fn open(db: &'db dyn salsa::Database, var: EffectVar) -> Self {
        Self::new(db, Vec::new(), Some(var))
    }

    /// Check if this row is pure (no effects and closed).
    pub fn is_pure(&self, db: &'db dyn salsa::Database) -> bool {
        self.effects(db).is_empty() && self.rest(db).is_none()
    }
}

/// An individual effect (ability).
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Effect<'db> {
    /// The ability name.
    pub name: Symbol,
    /// Type arguments for parameterized abilities.
    pub args: Vec<Type<'db>>,
}

/// Effect row variable for row-polymorphic effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct EffectVar {
    pub id: u64,
}

// =========================================================================
// Shared annotation → EffectRow conversion
// =========================================================================

/// Check whether a symbol name looks like a type variable (starts with lowercase).
fn is_type_variable(name: &Symbol) -> bool {
    name.with_str(|s| s.starts_with(|c: char| c.is_lowercase()))
}

/// Convert a single type annotation to an `Effect`.
///
/// Returns `None` for annotations that represent row variables (lowercase names, `Infer`)
/// rather than concrete abilities.
pub fn annotation_to_effect<'db>(
    db: &'db dyn salsa::Database,
    annotation: &TypeAnnotation,
    convert_type: &mut impl FnMut(&TypeAnnotation) -> Type<'db>,
) -> Option<Effect<'db>> {
    match &annotation.kind {
        TypeAnnotationKind::Named(name) if !is_type_variable(name) => Some(Effect {
            name: *name,
            args: vec![],
        }),
        TypeAnnotationKind::Path(path) if !path.is_empty() => {
            let name = *path.last()?;
            if is_type_variable(&name) {
                return None;
            }
            Some(Effect { name, args: vec![] })
        }
        TypeAnnotationKind::App { ctor, args } => {
            let name = match &ctor.kind {
                TypeAnnotationKind::Named(n) if !is_type_variable(n) => *n,
                TypeAnnotationKind::Path(path) => {
                    let n = *path.last()?;
                    if is_type_variable(&n) {
                        return None;
                    }
                    n
                }
                _ => return None,
            };
            let type_args: Vec<Type<'db>> = args.iter().map(&mut *convert_type).collect();
            let _ = db; // used by callers via convert_type
            Some(Effect {
                name,
                args: type_args,
            })
        }
        // Named lowercase, Infer, Error, etc. → not a concrete effect
        _ => None,
    }
}

/// Convert a slice of ability annotations to an `EffectRow`.
///
/// - Uppercase names / `App` → concrete effects
/// - Lowercase names / `Infer` → open row variable (via `fresh_row_var`)
pub fn abilities_to_effect_row<'db>(
    db: &'db dyn salsa::Database,
    abilities: &[TypeAnnotation],
    convert_type: &mut impl FnMut(&TypeAnnotation) -> Type<'db>,
    fresh_row_var: impl FnOnce() -> EffectVar,
) -> EffectRow<'db> {
    let mut effects = Vec::new();
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
                if let Some(effect) = annotation_to_effect(db, ann, convert_type) {
                    effects.push(effect);
                }
            }
        }
    }

    let rest = if has_row_var {
        Some(fresh_row_var())
    } else {
        None
    };

    EffectRow::new(db, effects, rest)
}

/// Type annotation as written in source code.
///
/// This represents a type before resolution and checking.
/// It may contain unresolved names that need to be looked up.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct TypeAnnotation {
    /// The node ID for span lookup.
    pub id: NodeId,
    /// The kind of type annotation.
    pub kind: TypeAnnotationKind,
}

/// Kinds of type annotations in source code.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
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
    /// - `vec![]`: pure function (`fn(a) ->{} b`)
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
    /// A type with effects: `() ->{IO} ()`
    WithEffects {
        inner: Box<TypeAnnotation>,
        effects: Vec<TypeAnnotation>,
    },
    /// Inferred type (omitted annotation): `_`
    Infer,
    /// Error in parsing
    Error,
}
