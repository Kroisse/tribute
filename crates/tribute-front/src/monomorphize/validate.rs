//! Validate checker-selected instances on executable reference paths before erasure.
use super::instance::InstanceKey;
use super::{Dependency, Expressions, collect::is_concrete_type, mangle::mangle_instance_path};
use crate::ast::fold::TypeFoldable;
use crate::ast::{
    Decl, DeclTable, ExprKind, FuncDecl, FuncDefId, NodeId, ResolvedRef, SpanMaps, Type,
    TypeScheme, TypedRef,
};
use crate::typeck::subst::BoundVarSubst;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::SymbolPath;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InstanceErrorKind {
    MissingInstance,
    WrongDeclaration,
    TypeArgumentArity {
        expected: usize,
        found: usize,
    },
    RowArgumentArity {
        expected: usize,
        found: usize,
    },
    IncompleteTypeArgument,
    IncompleteAbilityArgument,
    InconsistentCallable,
    /// A backstop for runaway expansion that polymorphic recursion
    /// rejection does not catch.
    ExpansionLimit,
    /// A reference to a function of the caller's own recursive group at
    /// type arguments that strictly contain the caller's type parameters,
    /// spelled with the caller's parameter names. `callee` is `None` when
    /// the caller references itself.
    PolymorphicRecursion {
        caller: String,
        callee: Option<String>,
        instantiation: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceError {
    pub node: NodeId,
    pub kind: InstanceErrorKind,
}

impl InstanceError {
    /// The source diagnostic of this error, at the reference whose instance
    /// is invalid.
    pub fn diagnostic(&self, spans: &SpanMaps<'_>) -> Diagnostic {
        let message = match &self.kind {
            InstanceErrorKind::ExpansionLimit => "specialization did not terminate: generic \
                 definitions keep requiring instances at new type arguments"
                .to_owned(),
            InstanceErrorKind::PolymorphicRecursion {
                caller,
                callee: None,
                instantiation,
            } => format!(
                "`{caller}` calls itself at `{instantiation}`; polymorphic recursion is not \
                 supported"
            ),
            InstanceErrorKind::PolymorphicRecursion {
                caller,
                callee: Some(callee),
                instantiation,
            } => format!(
                "`{caller}` calls `{callee}` of its own recursive group at `{instantiation}`; \
                 polymorphic recursion is not supported"
            ),
            kind => format!("invalid specialization instance at {}: {kind:?}", self.node),
        };
        Diagnostic::new(
            message,
            spans.get_or_default(self.node),
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
    }
}

fn substitute<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    arguments: &[Type<'db>],
) -> Option<Type<'db>> {
    ty.try_fold_with(&mut BoundVarSubst::new(db, arguments))
        .ok()
}

/// Validate the instances reached from `module`'s entry points. A reference
/// into `dependency` is followed only to an instance its result does not
/// have, since its specialization validated those.
pub(super) fn validate<'db>(
    db: &'db dyn salsa::Database,
    module: &DeclTable<TypedRef<'db>>,
    dependency: Option<&Dependency<'db>>,
    schemes: &HashMap<SymbolPath, TypeScheme<'db>>,
    expressions: Expressions<'_, 'db>,
) -> Vec<InstanceError> {
    // Each function, and whether `module` declares it.
    let mut functions: HashMap<FuncDefId<'db>, (&FuncDecl<TypedRef<'db>>, bool)> =
        HashMap::default();
    let mut fields = HashSet::default();
    let tables = dependency
        .map(|dependency| (dependency.module, false))
        .into_iter()
        .chain([(module, true)]);
    for (entry, own) in
        tables.flat_map(|(table, own)| table.entries.iter().map(move |entry| (entry, own)))
    {
        match &entry.decl {
            Decl::Function(func) => {
                let name = entry.module.child(func.name.clone());
                functions.insert(FuncDefId::new(db, name), (func, own));
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
        .filter(|(id, (func, own))| {
            *own && (id.path(db) == ["main"]
                || (func.visibility == crate::ast::Visibility::Public
                    && schemes
                        .get(id.path(db))
                        .is_some_and(|scheme| scheme.type_params(db).is_empty())))
        })
        .map(|(id, _)| (*id, Vec::new()))
        .collect();
    let mut seen = HashSet::default();
    let mut errors = Vec::new();
    while let Some((id, arguments)) = queue.pop() {
        if !seen.insert((id, arguments.clone())) {
            continue;
        }
        let Some(&(func, _)) = functions.get(&id) else {
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
            if let Some(op) = expressions.get(|table| &table.perform_operations, &expr.id)
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
                    if let Some(op) = expressions.get(|table| &table.handler_operations, &arm.id)
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
            let Some(instance) = expressions.get(|table| &table.function_instances, &expr.id)
            else {
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
                let Ok(row) = row.try_fold_with(&mut BoundVarSubst::new(db, &arguments)) else {
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
            if let Some(dependency) = dependency
                && functions.get(target).is_some_and(|(_, own)| !own)
                && (concrete.is_empty()
                    || dependency
                        .function_instances
                        .contains(&mangle_instance_path(
                            db,
                            target.path(db),
                            &InstanceKey {
                                type_args: concrete.iter().copied().collect(),
                            },
                        )))
            {
                continue;
            }
            queue.push((*target, concrete));
        }
    }
    errors.sort_by_key(|error| error.node);
    errors.dedup();
    errors
}
