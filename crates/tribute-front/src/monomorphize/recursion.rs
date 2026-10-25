//! Reject polymorphic recursion before specialization
//! (`new-plans/generics.md`, "Polymorphic Recursion").
use super::validate::{InstanceError, InstanceErrorKind};
use crate::ast::{
    AbilityRow, Decl, DeclTable, ExprKind, FuncDefId, NodeId, ResolvedRef, Type, TypeKind,
    TypeScheme, TypedRef,
};
use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use trunk_ir::SymbolPath;

/// One function reference in a generic function's body.
struct Reference<'db> {
    node: NodeId,
    callee: FuncDefId<'db>,
    type_arguments: &'db [Type<'db>],
}

/// Report every reference from a generic function to a function of its own
/// recursive group whose type arguments strictly contain the caller's type
/// parameters. Every function of `module` is checked, reached or not.
pub(super) fn check<'db>(
    db: &'db dyn salsa::Database,
    module: &DeclTable<TypedRef<'db>>,
    schemes: &HashMap<SymbolPath, TypeScheme<'db>>,
    function_instances: &'db crate::SortedMap<NodeId, crate::typeck::FunctionInstance<'db>>,
) -> Vec<InstanceError> {
    let mut calls: HashMap<FuncDefId<'db>, Vec<Reference<'db>>> = HashMap::default();
    for entry in &module.entries {
        let Decl::Function(func) = &entry.decl else {
            continue;
        };
        let caller = FuncDefId::new(db, entry.module.child(func.name.clone()));
        let references = calls.entry(caller).or_default();
        func.body.for_each(|expr| {
            if let ExprKind::Var(TypedRef {
                resolved: ResolvedRef::Function { id },
                ..
            }) = expr.kind.as_ref()
                && let Some(instance) = function_instances.get(&expr.id)
                && instance.function == *id
            {
                references.push(Reference {
                    node: expr.id,
                    callee: *id,
                    type_arguments: &instance.type_arguments,
                });
            }
        });
    }
    let mut reachable: HashMap<FuncDefId<'db>, HashSet<FuncDefId<'db>>> = HashMap::default();
    let mut errors = Vec::new();
    for (&caller, references) in &calls {
        let Some(scheme) = schemes.get(caller.path(db)) else {
            continue;
        };
        let params = scheme.type_params(db);
        if params.is_empty() {
            continue;
        }
        for reference in references {
            if !reference
                .type_arguments
                .iter()
                .any(|ty| strictly_contains_bound_var(db, *ty))
            {
                continue;
            }
            let recursive = reference.callee == caller
                || reachable
                    .entry(reference.callee)
                    .or_insert_with(|| reachable_from(&calls, reference.callee))
                    .contains(&caller);
            if !recursive {
                continue;
            }
            errors.push(InstanceError {
                node: reference.node,
                kind: InstanceErrorKind::PolymorphicRecursion {
                    caller: caller.path(db).to_string(),
                    callee: (reference.callee != caller)
                        .then(|| reference.callee.path(db).to_string()),
                    instantiation: reference
                        .type_arguments
                        .iter()
                        .map(|ty| ty.display_with_params(db, params))
                        .join(", "),
                },
            });
        }
    }
    errors
}

/// The functions `start` reaches through references, `start` excluded
/// unless it reaches itself.
fn reachable_from<'db>(
    calls: &HashMap<FuncDefId<'db>, Vec<Reference<'db>>>,
    start: FuncDefId<'db>,
) -> HashSet<FuncDefId<'db>> {
    let mut seen = HashSet::default();
    let mut stack = vec![start];
    while let Some(function) = stack.pop() {
        for reference in calls.get(&function).into_iter().flatten() {
            if seen.insert(reference.callee) {
                stack.push(reference.callee);
            }
        }
    }
    seen
}

/// Whether `ty` mentions a type parameter of the enclosing scheme without
/// being one.
fn strictly_contains_bound_var<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    !matches!(ty.kind(db), TypeKind::BoundVar { .. }) && mentions_bound_var(db, ty)
}

fn mentions_bound_var<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    let row = |row: &AbilityRow<'db>| {
        row.instances(db)
            .iter()
            .any(|effect| effect.args.iter().any(|arg| mentions_bound_var(db, *arg)))
    };
    match ty.kind(db) {
        TypeKind::BoundVar { .. } => true,
        TypeKind::Named { args, .. } | TypeKind::Tuple(args) => {
            args.iter().any(|arg| mentions_bound_var(db, *arg))
        }
        TypeKind::App { ctor, args } => {
            mentions_bound_var(db, *ctor) || args.iter().any(|arg| mentions_bound_var(db, *arg))
        }
        TypeKind::Func {
            params,
            result,
            effect,
        } => {
            params.iter().any(|param| mentions_bound_var(db, *param))
                || mentions_bound_var(db, *result)
                || row(effect)
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => mentions_bound_var(db, *arg) || mentions_bound_var(db, *result) || row(effect),
        TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::UniVar { .. }
        | TypeKind::Error => false,
    }
}
