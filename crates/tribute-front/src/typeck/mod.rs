//! Type checking for the AST.
//!
//! This module transforms `DeclTable<ResolvedRef<'db>>` into `DeclTable<TypedRef<'db>>`
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

pub mod ability_row;
mod checker;
mod constraint;
mod context;
mod evidence_plan;
mod func_context;
#[cfg(test)]
pub(crate) mod prop;
mod solver;
pub mod subst;

pub use checker::{Mode, ModuleCheckResult, TypeChecker};
pub use constraint::{Constraint, ConstraintSet};
pub use context::{AbilityInfo, AbilityOpInfo, MethodEntry, ModuleTypeEnv, parameter_type_matches};

use crate::SortedMap;
use crate::ast::SpanMap;
pub use func_context::FunctionInferenceContext;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
pub use solver::{RowSubst, SolveError, TypeSolver, TypeSubst};
use trunk_ir::SymbolPath;

use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, CtorId, DeclTable, FuncDefId, NodeId, ResolvedRef, Type, TypeDefId, TypeParam,
    TypeScheme, TypedRef,
};

/// Exact, monomorphic semantic signature selected for a handler arm.
///
/// This is deliberately separate from expression node types: a handler arm is
/// not an expression, and lowering must not recover its operation instance from
/// parameter/body shapes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct InstantiatedHandlerOperation<'db> {
    pub ability: AbilityId<'db>,
    pub ability_args: Vec<Type<'db>>,
    pub kind: crate::ast::OpDeclKind,
    pub params: Vec<Type<'db>>,
    pub result: Type<'db>,
}

/// Exact, monomorphic semantic signature selected for an ability-operation
/// call. This records the operation instance independently of expression node
/// types so lowering never reconstructs a declaration from an erased value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
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
/// not be recovered from their body or from a concrete expression node type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct LambdaSignature<'db> {
    /// The solved full source function type.
    pub function_type: Type<'db>,
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
) -> HashMap<AbilityId<'db>, AbilityInfo<'db>> {
    schemas
        .iter()
        .map(|(id, type_params, operations)| {
            let operations = operations
                .iter()
                .cloned()
                .map(|operation| (operation.name.clone(), operation))
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct WellKnownType<'db> {
    pub ty: Type<'db>,
    pub definition: TypeDefId<'db>,
}

/// Semantic identities supplied by the prelude and required downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct WellKnownTypes<'db> {
    pub string: Option<WellKnownType<'db>>,
    /// The `==` that `String` literal patterns compare with.
    pub string_equality: Option<FuncDefId<'db>>,
    /// The `==` that `Bytes` literal patterns compare with.
    pub bytes_equality: Option<FuncDefId<'db>>,
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
        Self {
            string: None,
            string_equality: None,
            bytes_equality: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum FunctionInstanceOrigin<'db> {
    Declaration,
    /// A function a struct has for its named field, with no declaration of
    /// its own.
    FieldAccessor {
        owner: TypeDefId<'db>,
        field: Symbol,
        kind: FieldFunctionKind,
    },
}

/// The functions a struct `T` has for each named field `f`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum FieldFunctionKind {
    /// `T::f : fn(T) -> F`
    Get,
    /// `T::f::set : fn(T, F) -> T`
    Set,
    /// `T::f::modify : fn(T, fn(F) ->{e} F) ->{e} T`
    Modify,
}

impl FieldFunctionKind {
    /// The function's name in the namespace `T::f`; the getter is `T::f`
    /// itself.
    pub fn name(self) -> Option<&'static str> {
        match self {
            Self::Get => None,
            Self::Set => Some(crate::ast::FIELD_LENS_FUNCTIONS[0]),
            Self::Modify => Some(crate::ast::FIELD_LENS_FUNCTIONS[1]),
        }
    }

    /// The path of this function for the field `field` of the struct at
    /// `owner`.
    pub fn path(self, owner: &SymbolPath, field: &Symbol) -> SymbolPath {
        let getter = owner.child(field.clone());
        match self.name() {
            None => getter,
            Some(name) => getter.child(name),
        }
    }
}

/// The instantiation chosen for one source function reference. Argument order
/// is the source scheme's binder order, never recovered from a callable type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct FunctionInstance<'db> {
    pub origin: FunctionInstanceOrigin<'db>,
    pub function: FuncDefId<'db>,
    pub scheme: TypeScheme<'db>,
    pub type_arguments: Vec<Type<'db>>,
    pub row_arguments: Vec<crate::ast::AbilityRow<'db>>,
    pub callable: Type<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct LocalCallableInstance<'db> {
    pub binding: NodeId,
    pub local: crate::ast::LocalId,
    /// Only schemes without local data-type quantifiers are exported here.
    pub scheme: TypeScheme<'db>,
    pub row_arguments: Vec<crate::ast::AbilityRow<'db>>,
    pub callable: Type<'db>,
}

/// One change a call makes to its caller's evidence before the callee runs.
///
/// A call keeps the caller's evidence except for the ability instances its
/// plan names (`new-plans/type-inference.md`, 호출의 evidence 선택).
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum EvidenceStep<'db> {
    /// The callee reaches this caller-explicit instance only through its row
    /// tail, so the caller's handler is hidden from it.
    Mask(crate::ast::AbilityInstance<'db>),
    /// One more callee position takes the caller's handler for this instance.
    Dup(crate::ast::AbilityInstance<'db>),
    /// The callee's row tail is the caller's row tail of this index, so the
    /// callee takes that tail's evidence.
    Select(u32),
    /// The callee names this caller-explicit instance, which the selected
    /// row tail's evidence does not hold.
    Push(crate::ast::AbilityInstance<'db>),
    /// The callee's row is a union of row tails: each takes the caller's
    /// evidence after its own plan.
    Tails(Vec<Vec<EvidenceStep<'db>>>),
}

impl<'db> EvidenceStep<'db> {
    /// Call `visit` with each ability instance the plan step names.
    pub fn for_each_instance(&self, mut visit: impl FnMut(&crate::ast::AbilityInstance<'db>)) {
        self.visit_instances(&mut visit);
    }

    // The recursion passes one `&mut dyn` down: a generic parameter would
    // grow by a reference at each level of `Tails`.
    fn visit_instances(&self, visit: &mut dyn FnMut(&crate::ast::AbilityInstance<'db>)) {
        match self {
            Self::Mask(instance) | Self::Dup(instance) | Self::Push(instance) => visit(instance),
            Self::Select(_) => {}
            Self::Tails(plans) => {
                for step in plans.iter().flatten() {
                    step.visit_instances(visit);
                }
            }
        }
    }

    /// Apply `map` to the type arguments of each instance.
    pub fn map_types(&self, mut map: impl FnMut(Type<'db>) -> Type<'db>) -> Self {
        self.map_instance_types(&mut map)
    }

    fn map_instance_types(&self, map: &mut dyn FnMut(Type<'db>) -> Type<'db>) -> Self {
        let mut instance =
            |instance: &crate::ast::AbilityInstance<'db>| crate::ast::AbilityInstance {
                ability_id: instance.ability_id,
                args: instance.args.iter().map(|ty| map(*ty)).collect(),
            };
        match self {
            Self::Mask(effect) => Self::Mask(instance(effect)),
            Self::Dup(effect) => Self::Dup(instance(effect)),
            Self::Push(effect) => Self::Push(instance(effect)),
            Self::Select(index) => Self::Select(*index),
            Self::Tails(plans) => Self::Tails(
                plans
                    .iter()
                    .map(|plan| {
                        plan.iter()
                            .map(|step| step.map_instance_types(map))
                            .collect()
                    })
                    .collect(),
            ),
        }
    }
}

/// What type checking fixed for each expression and reference, keyed by
/// node.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct ExpressionTypeMetadata<'db> {
    pub node_types: SortedMap<NodeId, Type<'db>>,
    pub function_instances: SortedMap<NodeId, FunctionInstance<'db>>,
    pub local_instances: SortedMap<NodeId, LocalCallableInstance<'db>>,
    /// Non-identity evidence selections of calls, resumes, and handle
    /// installations, keyed by their expressions.
    pub evidence_plans: SortedMap<NodeId, Vec<EvidenceStep<'db>>>,
    /// Exact semantic operation instances for handler arms.
    pub handler_operations: SortedMap<NodeId, InstantiatedHandlerOperation<'db>>,
    /// Exact semantic operation instances for ability-operation calls.
    pub perform_operations: SortedMap<NodeId, InstantiatedPerformOperation<'db>>,
    /// Fully solved callable signatures for lambda expressions.
    pub lambda_signatures: SortedMap<NodeId, LambdaSignature<'db>>,
    /// Case expressions which type checking proved exhaustive.
    pub exhaustive_cases: Vec<NodeId>,
}

impl<'db> ExpressionTypeMetadata<'db> {
    /// This metadata after that of the package it depends on, as one table:
    /// an entry here replaces the dependency's for the same node.
    pub fn after_dependency(&self, dependency: &Self) -> Self {
        fn layered<V: Clone>(
            dependency: &SortedMap<NodeId, V>,
            own: &SortedMap<NodeId, V>,
        ) -> SortedMap<NodeId, V> {
            let mut layered = dependency.clone();
            layered.extend(own.iter().cloned());
            layered
        }
        let mut exhaustive_cases: Vec<_> = dependency
            .exhaustive_cases
            .iter()
            .chain(&self.exhaustive_cases)
            .copied()
            .collect();
        exhaustive_cases.sort();
        exhaustive_cases.dedup();
        Self {
            node_types: layered(&dependency.node_types, &self.node_types),
            function_instances: layered(&dependency.function_instances, &self.function_instances),
            local_instances: layered(&dependency.local_instances, &self.local_instances),
            evidence_plans: layered(&dependency.evidence_plans, &self.evidence_plans),
            handler_operations: layered(&dependency.handler_operations, &self.handler_operations),
            perform_operations: layered(&dependency.perform_operations, &self.perform_operations),
            lambda_signatures: layered(&dependency.lambda_signatures, &self.lambda_signatures),
            exhaustive_cases,
        }
    }
}

/// Constructor declarations and exact schemes for cloned enum variants.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct ConstructorTypeMetadata<'db> {
    pub schemes: Vec<(CtorId<'db>, TypeScheme<'db>)>,
    pub specialized_enum_variants: SortedMap<NodeId, TypeScheme<'db>>,
}

impl<'db> From<Vec<(CtorId<'db>, TypeScheme<'db>)>> for ConstructorTypeMetadata<'db> {
    fn from(schemes: Vec<(CtorId<'db>, TypeScheme<'db>)>) -> Self {
        Self {
            schemes,
            specialized_enum_variants: SortedMap::default(),
        }
    }
}

/// What type checking fixed for the module's declarations. A module's
/// output holds only its own package's declarations; the dependency's join
/// them in specialization (`DeclarationTypeMetadata::after_dependency`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct DeclarationTypeMetadata<'db> {
    /// Function type schemes, keyed by path because `FuncDefId` does not
    /// implement `Ord`.
    pub function_types: Vec<(SymbolPath, TypeScheme<'db>)>,
    /// Constructor schemes used to build logical nominal layouts without
    /// reinterpreting source annotations.
    pub constructor_types: ConstructorTypeMetadata<'db>,
    /// Deterministic ability schemas required by the public logical-lowering
    /// boundary. This preserves semantic declarations without re-inspection.
    pub ability_definitions: Vec<AbilitySchema<'db>>,
    /// The well-known identities the package declares; empty for any package
    /// but `std`.
    pub well_known_types: WellKnownTypes<'db>,
}

impl<'db> DeclarationTypeMetadata<'db> {
    /// These declarations after those of the package they depend on, as
    /// one table: a declaration here replaces the dependency's with the same
    /// identity, and the well-known identities come from whichever package
    /// declares them.
    pub fn after_dependency(&self, dependency: &Self) -> Self {
        fn layered<K: Eq + std::hash::Hash, V: Clone>(
            dependency: &[V],
            own: &[V],
            key: impl Fn(&V) -> K,
        ) -> Vec<V> {
            let own_keys: HashSet<K> = own.iter().map(&key).collect();
            dependency
                .iter()
                .filter(|entry| !own_keys.contains(&key(entry)))
                .chain(own)
                .cloned()
                .collect()
        }
        let constructor_types = ConstructorTypeMetadata {
            schemes: layered(
                &dependency.constructor_types.schemes,
                &self.constructor_types.schemes,
                |(id, _)| *id,
            ),
            specialized_enum_variants: dependency
                .constructor_types
                .specialized_enum_variants
                .iter()
                .chain(&self.constructor_types.specialized_enum_variants)
                .map(|(node, scheme)| (*node, *scheme))
                .collect(),
        };
        Self {
            function_types: layered(
                &dependency.function_types,
                &self.function_types,
                |(path, _)| path.clone(),
            ),
            constructor_types,
            ability_definitions: layered(
                &dependency.ability_definitions,
                &self.ability_definitions,
                |(id, ..)| *id,
            ),
            well_known_types: if self.well_known_types == WellKnownTypes::empty() {
                dependency.well_known_types
            } else {
                self.well_known_types
            },
        }
    }
}

/// The complete output of type checking one module: the typed AST with
/// the metadata of its declarations and of its expressions. Source
/// locations stay in the span map of the parse it was checked from.
#[salsa::tracked]
pub struct TypeCheckOutput<'db> {
    /// The type-checked AST module.
    pub module: DeclTable<TypedRef<'db>>,
    pub declarations: DeclarationTypeMetadata<'db>,
    pub expression_types: ExpressionTypeMetadata<'db>,
}

impl<'db> TypeCheckOutput<'db> {
    /// The output of a finished module check.
    pub fn from_check(db: &'db dyn salsa::Database, result: ModuleCheckResult<'db>) -> Self {
        Self::new(
            db,
            result.module,
            DeclarationTypeMetadata {
                function_types: result.function_types,
                constructor_types: result.constructor_types.into(),
                ability_definitions: ability_schemas(&result.ability_definitions),
                well_known_types: result.well_known_types,
            },
            ExpressionTypeMetadata {
                node_types: result.node_types,
                function_instances: result.function_instances,
                local_instances: result.local_instances,
                evidence_plans: result.evidence_plans,
                handler_operations: result.handler_operations,
                perform_operations: result.perform_operations,
                lambda_signatures: result.lambda_signatures,
                exhaustive_cases: result.exhaustive_cases,
            },
        )
    }
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
    #[returns(deref)]
    pub function_types: Vec<(FuncDefId<'db>, TypeScheme<'db>)>,

    /// Functions declared `extern`, sorted by fully qualified name.
    #[returns(deref)]
    pub extern_functions: Vec<FuncDefId<'db>>,

    /// Constructor type schemes keyed by CtorId.
    #[returns(deref)]
    pub constructor_types: Vec<(CtorId<'db>, TypeScheme<'db>)>,

    /// Type definitions keyed by name.
    #[returns(deref)]
    pub type_defs: Vec<(SymbolPath, TypeScheme<'db>)>,

    /// Struct field definitions keyed by nominal declaration identity.
    #[returns(deref)]
    pub struct_fields: Vec<(TypeDefId<'db>, (Vec<TypeParam>, Vec<(Symbol, Type<'db>)>))>,

    /// Enum variant information: enum_name → [variant_names].
    #[returns(deref)]
    pub enum_variants: Vec<(SymbolPath, Vec<Symbol>)>,

    /// Field names of constructors whose fields are all named.
    #[returns(deref)]
    pub constructor_field_names: Vec<(CtorId<'db>, Vec<Symbol>)>,

    /// Ability operation schemas exported by the prelude as deterministic
    /// vectors (a dependent module's environment indexes them for lookup).
    #[returns(deref)]
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
    module: &DeclTable<ResolvedRef<'db>>,
    span_map: &'db SpanMap,
) -> TypeCheckOutput<'db> {
    let checker = TypeChecker::new(db, span_map);
    TypeCheckOutput::from_check(db, checker.check_module(module))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeKind;

    fn expressions<'db>(
        node_types: impl IntoIterator<Item = (NodeId, Type<'db>)>,
        exhaustive_cases: Vec<NodeId>,
    ) -> ExpressionTypeMetadata<'db> {
        ExpressionTypeMetadata {
            node_types: node_types.into_iter().collect(),
            function_instances: SortedMap::default(),
            local_instances: SortedMap::default(),
            evidence_plans: SortedMap::default(),
            handler_operations: SortedMap::default(),
            perform_operations: SortedMap::default(),
            lambda_signatures: SortedMap::default(),
            exhaustive_cases,
        }
    }

    #[test]
    fn expression_metadata_layers_over_its_dependency() {
        let db = salsa::DatabaseImpl::default();
        let (int, nat) = (Type::new(&db, TypeKind::Int), Type::new(&db, TypeKind::Nat));
        let [shared, library, own] = [1, 2, 3].map(NodeId::from_raw);
        let dependency = expressions([(shared, int), (library, int)], vec![library, shared]);
        let module = expressions([(shared, nat), (own, nat)], vec![own]);

        let layered = module.after_dependency(&dependency);
        assert_eq!(
            &*layered.node_types,
            [(shared, nat), (library, int), (own, nat)]
        );
        assert_eq!(layered.exhaustive_cases, [shared, library, own]);
    }

    #[test]
    fn declaration_metadata_layers_over_its_dependency() {
        let db = salsa::DatabaseImpl::default();
        let scheme = |kind| TypeScheme::mono(&db, Type::new(&db, kind));
        let declarations = |function_types, well_known_types| DeclarationTypeMetadata {
            function_types,
            constructor_types: ConstructorTypeMetadata::default(),
            ability_definitions: Vec::new(),
            well_known_types,
        };
        let path = |name: &str| SymbolPath::new(["std", name]);
        let well_known = WellKnownTypes {
            string_equality: Some(FuncDefId::new(&db, path("eq"))),
            ..WellKnownTypes::empty()
        };
        let dependency = declarations(
            vec![
                (path("a"), scheme(TypeKind::Int)),
                (path("b"), scheme(TypeKind::Int)),
            ],
            well_known,
        );
        let module = declarations(
            vec![
                (path("b"), scheme(TypeKind::Nat)),
                (SymbolPath::new(["c"]), scheme(TypeKind::Nat)),
            ],
            WellKnownTypes::empty(),
        );

        let layered = module.after_dependency(&dependency);
        assert_eq!(
            layered.function_types,
            [
                (path("a"), scheme(TypeKind::Int)),
                (path("b"), scheme(TypeKind::Nat)),
                (SymbolPath::new(["c"]), scheme(TypeKind::Nat)),
            ]
        );
        assert_eq!(layered.well_known_types, well_known);
    }
}
