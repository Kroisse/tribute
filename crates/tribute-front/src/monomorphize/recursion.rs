//! Reject polymorphic recursion before specialization
//! (`new-plans/generics.md`, "Polymorphic Recursion").
use std::ops::ControlFlow;

use super::validate::{InstanceError, InstanceErrorKind};
use crate::ast::fold::{TypeVisitable, TypeVisitor};
use crate::ast::{
    Decl, DeclTable, ExprKind, FuncDefId, NodeId, ResolvedRef, Type, TypeKind, TypeScheme, TypedRef,
};
use crate::resolve::NameDisplay;
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
    names: &NameDisplay,
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
                    caller: names.path(caller.path(db)),
                    callee: (reference.callee != caller)
                        .then(|| names.path(reference.callee.path(db))),
                    instantiation: reference
                        .type_arguments
                        .iter()
                        .map(|ty| ty.display_with_params(db, params, names))
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
    struct MentionsBoundVar<'db>(&'db dyn salsa::Database);

    impl<'db> TypeVisitor<'db> for MentionsBoundVar<'db> {
        type Break = ();

        fn db(&self) -> &'db dyn salsa::Database {
            self.0
        }

        fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<()> {
            match ty.kind(self.0) {
                TypeKind::BoundVar { .. } => ControlFlow::Break(()),
                _ => ty.super_visit_with(self),
            }
        }
    }

    ty.visit_with(&mut MentionsBoundVar(db)).is_break()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, AbilityInstance, AbilityRow, RowVar, TypeDefId};
    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    #[salsa_test]
    fn strict_containment_looks_through_every_type_form(db: &salsa::DatabaseImpl) {
        let var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(db, TypeKind::Int);
        let local = Type::new(
            db,
            TypeKind::LocalBoundVar {
                scope: NodeId::from_raw(0),
                index: 0,
            },
        );
        let pure = AbilityRow::pure(db);
        let row_of = |arg| {
            AbilityRow::new(
                db,
                vec![AbilityInstance {
                    ability_id: AbilityId::source(db, SymbolPath::new(["State"])),
                    args: vec![arg],
                }],
                Some(RowVar { id: 1 }),
            )
        };
        let func = |params, result, effect| {
            Type::new(
                db,
                TypeKind::Func {
                    params,
                    result,
                    effect,
                },
            )
        };
        let continuation = |arg, result, effect| {
            Type::new(
                db,
                TypeKind::Continuation {
                    arg,
                    result,
                    effect,
                },
            )
        };
        let named = |args| {
            Type::new(
                db,
                TypeKind::Named {
                    id: TypeDefId::synthetic(db, SymbolPath::new(["Pair"])),
                    name: Symbol::new("Pair"),
                    args,
                },
            )
        };
        let cases = [
            (var, false),
            (int, false),
            (local, false),
            (Type::new(db, TypeKind::Error), false),
            (named(vec![int]), false),
            (named(vec![var]), true),
            (Type::new(db, TypeKind::Tuple(vec![int, var])), true),
            (func(vec![var], int, pure), true),
            (func(vec![int], var, pure), true),
            (func(vec![int], int, row_of(var)), true),
            (func(vec![local], int, row_of(int)), false),
            (continuation(var, int, pure), true),
            (continuation(int, var, pure), true),
            (continuation(int, int, row_of(var)), true),
            (continuation(int, int, row_of(int)), false),
        ];
        for (ty, expected) in cases {
            assert_eq!(
                strictly_contains_bound_var(db, ty),
                expected,
                "{:?}",
                ty.kind(db)
            );
        }
    }
}
