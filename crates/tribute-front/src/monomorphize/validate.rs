//! Validate checker-selected instances on executable reference paths before erasure.
use super::{MonomorphizeMetadata, collect::is_concrete_type};
use crate::ast::{
    Decl, DeclTable, ExprKind, FuncDecl, FuncDefId, NodeId, ResolvedRef, Type, TypeScheme, TypedRef,
};
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use trunk_ir::SymbolPath;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InstanceErrorKind {
    MissingInstance,
    WrongDeclaration,
    TypeArgumentArity { expected: usize, found: usize },
    RowArgumentArity { expected: usize, found: usize },
    IncompleteTypeArgument,
    IncompleteAbilityArgument,
    InconsistentCallable,
    ExpansionLimit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceError {
    pub node: NodeId,
    pub kind: InstanceErrorKind,
}

fn substitute<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    arguments: &[Type<'db>],
) -> Option<Type<'db>> {
    match crate::typeck::subst::substitute_bound_vars(db, ty, arguments) {
        crate::typeck::subst::SubstResult::Ok(ty) => Some(ty),
        _ => None,
    }
}

pub(super) fn validate<'db>(
    db: &'db dyn salsa::Database,
    module: &DeclTable<TypedRef<'db>>,
    schemes: &HashMap<SymbolPath, TypeScheme<'db>>,
    metadata: &MonomorphizeMetadata<'db>,
) -> Vec<InstanceError> {
    let mut functions: HashMap<FuncDefId<'db>, &FuncDecl<TypedRef<'db>>> = HashMap::default();
    let mut fields = HashSet::default();
    for entry in &module.entries {
        match &entry.decl {
            Decl::Function(func) => {
                let name = entry.module.child(func.name.clone());
                functions.insert(FuncDefId::new(db, name), func);
            }
            Decl::Struct(decl) => {
                let name = entry.module.child(decl.name.clone());
                let owner = crate::ast::TypeDefId::declared(db, name, decl.disambiguator);
                for field in &decl.fields {
                    if let Some(name) = field.name.clone() {
                        fields.insert((owner, name));
                    }
                }
            }
            _ => {}
        }
    }
    let mut queue: Vec<_> = functions
        .iter()
        .filter(|(id, func)| {
            id.path(db) == ["main"]
                || (func.visibility == crate::ast::Visibility::Public
                    && schemes
                        .get(id.path(db))
                        .is_some_and(|scheme| scheme.type_params(db).is_empty()))
        })
        .map(|(id, _)| (*id, Vec::new()))
        .collect();
    let mut seen = HashSet::default();
    let mut errors = Vec::new();
    while let Some((id, arguments)) = queue.pop() {
        if !seen.insert((id, arguments.clone())) {
            continue;
        }
        let Some(func) = functions.get(&id) else {
            continue;
        };
        if seen.len() > 4096 {
            errors.push(InstanceError {
                node: func.id,
                kind: InstanceErrorKind::ExpansionLimit,
            });
            break;
        }
        let mut nodes = Vec::new();
        func.body.for_each(|node| nodes.push(node));
        for expr in nodes {
            if let Some(op) = metadata.perform_operations.get(&expr.id)
                && op.ability_args.iter().any(|ty| {
                    substitute(db, *ty, &arguments).is_none_or(|ty| !is_concrete_type(db, ty))
                })
            {
                errors.push(InstanceError {
                    node: expr.id,
                    kind: InstanceErrorKind::IncompleteAbilityArgument,
                });
            }
            if let ExprKind::Handle { handlers, .. } = expr.kind.as_ref() {
                for arm in handlers {
                    if let Some(op) = metadata.handler_operations.get(&arm.id)
                        && op.ability_args.iter().any(|ty| {
                            substitute(db, *ty, &arguments)
                                .is_none_or(|ty| !is_concrete_type(db, ty))
                        })
                    {
                        errors.push(InstanceError {
                            node: arm.id,
                            kind: InstanceErrorKind::IncompleteAbilityArgument,
                        });
                    }
                }
            }
            let ExprKind::Var(TypedRef {
                resolved: ResolvedRef::Function { id: target },
                ..
            }) = expr.kind.as_ref()
            else {
                continue;
            };
            let fail = |kind| InstanceError {
                node: expr.id,
                kind,
            };
            let Some(instance) = metadata.function_instances.get(&expr.id) else {
                errors.push(fail(InstanceErrorKind::MissingInstance));
                continue;
            };
            let known_declaration = match instance.origin.clone() {
                crate::typeck::FunctionInstanceOrigin::Declaration => {
                    schemes.get(target.path(db)) == Some(&instance.scheme)
                }
                crate::typeck::FunctionInstanceOrigin::FieldAccessor { owner, field, kind } => {
                    fields.contains(&(owner, field.clone()))
                        && *target.path(db) == kind.path(owner.path(db), &field)
                }
            };
            if instance.function != *target || !known_declaration {
                errors.push(fail(InstanceErrorKind::WrongDeclaration));
                continue;
            }
            let expected = instance.scheme.type_params(db).len();
            if instance.type_arguments.len() != expected {
                errors.push(fail(InstanceErrorKind::TypeArgumentArity {
                    expected,
                    found: instance.type_arguments.len(),
                }));
                continue;
            }
            let expected = instance.scheme.effect_params(db).len();
            if instance.row_arguments.len() != expected {
                errors.push(fail(InstanceErrorKind::RowArgumentArity {
                    expected,
                    found: instance.row_arguments.len(),
                }));
                continue;
            }
            let concrete: Option<Vec<_>> = instance
                .type_arguments
                .iter()
                .map(|ty| substitute(db, *ty, &arguments).filter(|ty| is_concrete_type(db, *ty)))
                .collect();
            let Some(concrete) = concrete else {
                errors.push(fail(InstanceErrorKind::IncompleteTypeArgument));
                continue;
            };
            let Some(callable) = substitute(db, instance.callable, &arguments) else {
                errors.push(fail(InstanceErrorKind::InconsistentCallable));
                continue;
            };
            let mut solver = crate::typeck::TypeSolver::new(db);
            solver.reserve_row_vars_in_type(callable);
            for row in &instance.row_arguments {
                solver.reserve_row_vars_in_row(*row);
            }
            let selected = crate::typeck::subst::instantiate_scheme_details_for_solver(
                db,
                instance.scheme,
                &mut solver,
            );
            let mut constraints = crate::typeck::ConstraintSet::new();
            for (variable, argument) in selected.type_args.iter().zip(&concrete) {
                constraints.add_type_eq(*variable, *argument);
            }
            let mut invalid_row = false;
            for (variable, row) in selected.row_args.iter().zip(&instance.row_arguments) {
                let Ok(row) = crate::typeck::subst::substitute_ability_row(db, *row, &arguments)
                else {
                    invalid_row = true;
                    break;
                };
                constraints.add_row_eq(*variable, row);
            }
            constraints.add_type_eq(selected.ty, callable);
            if invalid_row
                || solver.solve(constraints).is_err()
                || solver.finalize_relations().is_err()
            {
                errors.push(fail(InstanceErrorKind::InconsistentCallable));
                continue;
            }
            queue.push((*target, concrete));
        }
    }
    errors.sort_by_key(|error| error.node);
    errors.dedup();
    errors
}
