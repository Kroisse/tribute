//! Type checking for the AST.
//!
//! This module transforms `Module<ResolvedRef<'db>>` into `Module<TypedRef<'db>>`
//! by performing bidirectional type checking with row-polymorphic effects.
//!
//! ## Pipeline
//!
//! 1. Collect type definitions and function signatures
//! 2. Generate type constraints by walking the AST
//! 3. Solve constraints via unification
//! 4. Apply the solved substitution to produce typed AST
//!
//! ## Type System Features
//!
//! - Hindley-Milner type inference with let-polymorphism
//! - Bidirectional type checking (infer/check modes)
//! - Row-polymorphic effect tracking
//! - Algebraic data types (structs, enums)
//! - Pattern matching exhaustiveness (future)

mod checker;
mod constraint;
mod context;
pub mod effect_row;
mod func_context;
mod solver;
pub mod subst;

pub use checker::{Mode, TypeChecker};
pub use constraint::{Constraint, ConstraintSet};
pub use context::ModuleTypeEnv;

use crate::ast::SpanMap;
pub use func_context::FunctionInferenceContext;
pub use solver::{RowSubst, SolveError, TypeSolver, TypeSubst};

use trunk_ir::Symbol;

use crate::ast::{CtorId, FuncDefId, Module, ResolvedRef, Type, TypeParam, TypeScheme, TypedRef};

/// Salsa-tracked struct holding the complete type checking output.
///
/// Bundles the typed AST module with function type schemes so that
/// both can be derived from a single type checking invocation.
#[salsa::tracked]
pub struct TypeCheckOutput<'db> {
    /// The type-checked AST module.
    pub module: Module<TypedRef<'db>>,
    /// Function type schemes collected during type checking.
    /// Stored as Vec<(Symbol, TypeScheme)> because FuncDefId doesn't implement Ord.
    #[returns(ref)]
    pub function_types: Vec<(Symbol, TypeScheme<'db>)>,
}

/// Prelude's exported type information.
///
/// This struct holds type information extracted from the prelude after it has been
/// fully type-checked. All types are resolved TypeSchemes with no UniVars - only
/// BoundVars for polymorphic parameters.
///
/// This allows user code to use prelude types without sharing a TypeContext,
/// avoiding UniVar counter conflicts that caused issues in the AST merge approach.
#[salsa::tracked]
pub struct PreludeExports<'db> {
    /// Function type schemes keyed by FuncDefId.
    #[returns(ref)]
    pub function_types: Vec<(FuncDefId<'db>, TypeScheme<'db>)>,

    /// Constructor type schemes keyed by CtorId.
    #[returns(ref)]
    pub constructor_types: Vec<(CtorId<'db>, TypeScheme<'db>)>,

    /// Type definitions keyed by name.
    #[returns(ref)]
    pub type_defs: Vec<(Symbol, TypeScheme<'db>)>,

    /// Struct field definitions: struct_name → (type_params, [(field_name, field_type)]).
    #[returns(ref)]
    pub struct_fields: Vec<(Symbol, (Vec<TypeParam>, Vec<(Symbol, Type<'db>)>))>,

    /// Enum variant information: enum_name → [variant_names].
    #[returns(ref)]
    pub enum_variants: Vec<(Symbol, Vec<Symbol>)>,
}

/// Type check a module.
///
/// This is the main entry point for type checking.
/// Returns both the typed AST and function type schemes.
pub fn typecheck_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<ResolvedRef<'db>>,
    span_map: SpanMap,
) -> TypeCheckOutput<'db> {
    let checker = TypeChecker::new(db, span_map);
    let (typed_module, function_types) = checker.check_module(module);
    TypeCheckOutput::new(db, typed_module, function_types)
}
