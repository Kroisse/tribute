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
pub use context::{
    AbilityInfo, AbilityOpInfo, MethodEntry, ModuleTypeEnv, extract_type_name_from_type,
    receiver_type_matches,
};

use crate::ast::SpanMap;
pub use func_context::FunctionInferenceContext;
pub use solver::{RowSubst, SolveError, TypeSolver, TypeSubst};

use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, CallingConvention, CtorId, FuncDefId, Module, NodeId, ResolvedRef, Type, TypeDefId,
    TypeParam, TypeScheme, TypedRef,
};

/// Exact, monomorphic semantic signature selected for a handler arm.
///
/// This is deliberately separate from expression node types: a handler arm is
/// not an expression, and lowering must not recover its operation instance from
/// parameter/body shapes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct InstantiatedHandlerOperation<'db> {
    pub ability: AbilityId<'db>,
    pub ability_args: Vec<Type<'db>>,
    pub kind: crate::ast::OpDeclKind,
    pub params: Vec<Type<'db>>,
    pub result: Type<'db>,
}

/// Exact, monomorphic semantic signature selected for an ability-operation
/// call. This is distinct from the concrete node-type table used by legacy
/// physical lowering, so preserving a logical effect instance cannot change
/// legacy value representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct InstantiatedPerformOperation<'db> {
    pub ability: AbilityId<'db>,
    pub ability_args: Vec<Type<'db>>,
    pub kind: crate::ast::OpDeclKind,
    pub params: Vec<Type<'db>>,
    pub result: Type<'db>,
}

/// Fully solved callable signature for a lambda expression.
///
/// Lambdas are expressions, but their source-logical callable signature must
/// not be recovered from their body or from the concrete node-type table used
/// by legacy lowering.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct LambdaSignature<'db> {
    /// The solved full source function type.  Retaining its effect row avoids
    /// confusing the ABI lower bound with the selected convention.
    pub function_type: Type<'db>,
    pub convention: CallingConvention,
}

/// Deterministic schema form of an ability declaration for the public typed
/// frontend boundary. Operations are sorted by name.
pub type AbilitySchema<'db> = (AbilityId<'db>, Vec<TypeParam>, Vec<AbilityOpInfo<'db>>);

/// Convert module-local ability definitions into deterministic public schemas.
pub fn ability_schemas<'db>(
    definitions: &[(AbilityId<'db>, AbilityInfo<'db>)],
) -> Vec<AbilitySchema<'db>> {
    let schemas: Vec<_> = definitions
        .iter()
        .map(|(id, info)| {
            let mut operations: Vec<_> = info.operations.values().cloned().collect();
            operations.sort_by(|left, right| {
                left.name
                    .with_str(|left| right.name.with_str(|right| left.cmp(right)))
            });
            (*id, info.type_params.clone(), operations)
        })
        .collect();
    schemas
}

/// Rebuild lowering's lookup representation from public deterministic schemas.
pub fn ability_definitions_from_schemas<'db>(
    schemas: &[AbilitySchema<'db>],
) -> std::collections::HashMap<AbilityId<'db>, AbilityInfo<'db>> {
    schemas
        .iter()
        .map(|(id, type_params, operations)| {
            let operations = operations
                .iter()
                .cloned()
                .map(|operation| (operation.name, operation))
                .collect();
            (
                *id,
                AbilityInfo {
                    id: *id,
                    type_params: type_params.clone(),
                    operations,
                },
            )
        })
        .collect()
}

/// Semantic identities supplied by the prelude and required downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct DefinitionIdentity {
    pub source: u64,
    pub start: usize,
    pub end: usize,
}

impl DefinitionIdentity {
    pub fn new(declaration: NodeId, span: trunk_ir::Span) -> Self {
        Self {
            source: declaration.source(),
            start: span.start,
            end: span.end,
        }
    }
}

/// Semantic identities supplied by the prelude and required downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct WellKnownType<'db> {
    pub ty: Type<'db>,
    pub definition: DefinitionIdentity,
}

/// Semantic identities supplied by the prelude and required downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct WellKnownTypes<'db> {
    pub string: Option<WellKnownType<'db>>,
}

/// A typed key for extracting a semantic type from the prelude.
pub trait WellKnownTypeKey: Copy {
    fn name(self) -> Symbol;
}

/// Key for the prelude-defined `String` type.
#[derive(Clone, Copy, Debug, Default)]
pub struct StringType;

impl WellKnownTypeKey for StringType {
    fn name(self) -> Symbol {
        Symbol::new("String")
    }
}

impl WellKnownTypes<'_> {
    pub const fn empty() -> Self {
        Self { string: None }
    }
}

/// Salsa-tracked struct holding the complete type checking output.
///
/// Bundles the typed AST module with function type schemes so that
/// both can be derived from a single type checking invocation.
/// Also stores the SpanMap so that downstream stages (e.g., ast_to_ir)
/// can look up source spans without a separate plumbing path.
#[salsa::tracked]
pub struct TypeCheckOutput<'db> {
    /// The type-checked AST module.
    #[returns(ref)]
    pub module: Module<TypedRef<'db>>,
    /// Function type schemes collected during type checking.
    /// Stored as Vec<(Symbol, TypeScheme)> because FuncDefId doesn't implement Ord.
    #[returns(ref)]
    pub function_types: Vec<(Symbol, TypeScheme<'db>)>,
    /// Constructor schemes used to build logical nominal layouts without
    /// reinterpreting source annotations.
    #[returns(ref)]
    pub constructor_types: Vec<(CtorId<'db>, TypeScheme<'db>)>,
    /// Node types collected during type checking.
    /// Maps NodeId to its inferred type (after substitution but before generalization).
    /// Used by IR lowering to get lambda effect types.
    /// Stored as Vec for Salsa compatibility (HashMap doesn't implement Hash).
    #[returns(ref)]
    pub node_types: Vec<(NodeId, Type<'db>)>,
    /// Ability-level calling-convention requirements.
    #[returns(ref)]
    pub ability_conventions: Vec<(AbilityId<'db>, CallingConvention)>,
    /// Deterministic ability schemas required by the public logical-lowering
    /// boundary. This preserves semantic declarations without re-inspection.
    #[returns(ref)]
    pub ability_definitions: Vec<AbilitySchema<'db>>,
    /// Exact semantic operation instances for handler arms.
    #[returns(ref)]
    pub handler_operations: Vec<(NodeId, InstantiatedHandlerOperation<'db>)>,
    /// Exact semantic operation instances for ability-operation calls.
    #[returns(ref)]
    pub perform_operations: Vec<(NodeId, InstantiatedPerformOperation<'db>)>,
    /// Fully solved callable signatures for lambda expressions.
    #[returns(ref)]
    pub lambda_signatures: Vec<(NodeId, LambdaSignature<'db>)>,
    /// Case expressions which type checking proved exhaustive.
    #[returns(ref)]
    pub exhaustive_cases: Vec<NodeId>,
    /// Prelude-defined semantic type identities.
    pub well_known_types: WellKnownTypes<'db>,
    /// Source span information for AST nodes.
    pub span_map: SpanMap,
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

    /// Struct field definitions keyed by nominal declaration identity.
    #[returns(ref)]
    pub struct_fields: Vec<(TypeDefId<'db>, (Vec<TypeParam>, Vec<(Symbol, Type<'db>)>))>,

    /// Enum variant information: enum_name → [variant_names].
    #[returns(ref)]
    pub enum_variants: Vec<(Symbol, Vec<Symbol>)>,

    /// Method index for UFCS resolution: method_name → candidates.
    #[returns(ref)]
    pub method_index: Vec<(Symbol, Vec<MethodEntry<'db>>)>,
    /// Ability-level calling-convention requirements exported by the prelude.
    #[returns(ref)]
    pub ability_conventions: Vec<(AbilityId<'db>, CallingConvention)>,
    /// Ability operation schemas exported by the prelude as deterministic
    /// vectors (the module environment rebuilds its lookup map on injection).
    #[returns(ref)]
    pub ability_definitions: Vec<(
        AbilityId<'db>,
        Vec<TypeParam>,
        Vec<crate::typeck::context::AbilityOpInfo<'db>>,
    )>,
    /// Prelude-defined semantic type identities.
    pub well_known_types: WellKnownTypes<'db>,
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
    let checker = TypeChecker::new(db, span_map.clone());
    let result = checker.check_module(module);
    TypeCheckOutput::new(
        db,
        result.module,
        result.function_types,
        result.constructor_types,
        result.node_types,
        result.ability_conventions,
        ability_schemas(&result.ability_definitions),
        result.handler_operations,
        result.perform_operations,
        result.lambda_signatures,
        result.exhaustive_cases,
        result.well_known_types,
        span_map,
    )
}
