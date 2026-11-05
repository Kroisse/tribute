//! Validate checker-selected instances on executable reference paths before erasure.
use super::{Dependency, Expressions, collect::is_concrete_type};
use crate::ast::fold::{TypeFoldable, TypeVisitable, TypeVisitor};
use crate::ast::{
    AbilityRow, Decl, DeclTable, ExprKind, FuncDecl, FuncDefId, NodeId, ResolvedRef, SpanMaps,
    Type, TypeDefId, TypeKind, TypeScheme, TypedRef,
};
use crate::typeck::subst::BoundVarSubst;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;
use tribute_core::diagnostic::{CompilationPhase, DiagnosticSeverity, SpanDiagnostic};
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
    /// A generic struct or enum named at an argument that holds an open
    /// row, which specialization cannot make concrete.
    IncompleteNominalArgument,
    InconsistentCallable,
    /// A backstop for runaway expansion that polymorphic recursion
    /// rejection does not catch.
    ExpansionLimit,
    /// The specialization output breaks its contract: a compiler
    /// inconsistency, not a source error.
    SpecializationInvariant(super::SpecializationInvariant),
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
    pub fn diagnostic(&self, spans: &SpanMaps<'_>) -> SpanDiagnostic {
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
            InstanceErrorKind::IncompleteNominalArgument => "a generic struct or enum is used at \
                 a type argument that holds an open ability row; it does not specialize, so give \
                 the argument a closed row"
                .to_owned(),
            InstanceErrorKind::SpecializationInvariant(violation) => format!(
                "internal compiler error: the specialization output breaks its contract at {}: \
                 {violation:?}",
                self.node
            ),
            kind => format!("invalid specialization instance at {}: {kind:?}", self.node),
        };
        SpanDiagnostic::new(
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

/// Whether a row that `ty` spells, at the type arguments `arguments`, names
/// an ability instance whose argument is not concrete. A row variable is
/// never substituted, so an open row inside an ability argument stays
/// unresolved.
fn names_incomplete_ability_argument<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    arguments: &[Type<'db>],
) -> bool {
    struct Incomplete<'db> {
        db: &'db dyn salsa::Database,
    }
    impl<'db> TypeVisitor<'db> for Incomplete<'db> {
        type Break = ();

        fn db(&self) -> &'db dyn salsa::Database {
            self.db
        }

        fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<()> {
            if row
                .instances(self.db)
                .iter()
                .flat_map(|ability| &ability.args)
                .any(|argument| !is_concrete_type(self.db, *argument))
            {
                return ControlFlow::Break(());
            }
            row.super_visit_with(self)
        }
    }
    substitute(db, ty, arguments).is_some_and(|ty| ty.visit_with(&mut Incomplete { db }).is_break())
}

/// Whether `ty`, at the type arguments `arguments`, names a generic struct
/// or enum at an argument holding an open row. A row variable is not
/// substituted, so the instance would get no specialized declaration.
fn names_incomplete_nominal_argument<'db>(
    db: &'db dyn salsa::Database,
    generic: &HashSet<TypeDefId<'db>>,
    ty: Type<'db>,
    arguments: &[Type<'db>],
) -> bool {
    struct Incomplete<'a, 'db> {
        db: &'db dyn salsa::Database,
        generic: &'a HashSet<TypeDefId<'db>>,
        in_argument: bool,
    }
    impl<'db> TypeVisitor<'db> for Incomplete<'_, 'db> {
        type Break = ();

        fn db(&self) -> &'db dyn salsa::Database {
            self.db
        }

        fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<()> {
            match ty.kind(self.db) {
                TypeKind::Named { id, .. } if !self.in_argument && self.generic.contains(id) => {
                    self.in_argument = true;
                    let flow = ty.super_visit_with(self);
                    self.in_argument = false;
                    flow
                }
                _ => ty.super_visit_with(self),
            }
        }

        fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<()> {
            if self.in_argument && row.rest(self.db).is_some() {
                return ControlFlow::Break(());
            }
            row.super_visit_with(self)
        }
    }
    substitute(db, ty, arguments).is_some_and(|ty| {
        ty.visit_with(&mut Incomplete {
            db,
            generic,
            in_argument: false,
        })
        .is_break()
    })
}

/// Validate the instances reached from every monomorphic function `module`
/// declares. A reference into `dependency` is followed only to an instance
/// of its generics; the dependency's specialization validates its own
/// monomorphic functions.
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
    let mut generic = HashSet::default();
    let tables = dependency
        .into_iter()
        .flat_map(|dependency| dependency.modules.iter().map(|&table| (table, false)))
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
                let owner = TypeDefId::declared(db, name, decl.disambiguator);
                if !decl.type_params.is_empty() {
                    generic.insert(owner);
                }
                for field in &decl.fields {
                    if let Some(name) = field.name.clone() {
                        fields.insert((owner, name));
                    }
                }
            }
            Decl::Enum(decl) if !decl.type_params.is_empty() => {
                let name = entry.module.child(decl.name.clone());
                generic.insert(TypeDefId::declared(db, name, decl.disambiguator));
            }
            _ => {}
        }
    }
    // Every monomorphic function is lowered whether or not it is reached, so
    // each one is a root.
    let mut queue: Vec<_> = functions
        .iter()
        .filter(|(id, (_, own))| {
            *own && schemes
                .get(id.path(db))
                .is_some_and(|scheme| scheme.type_params(db).is_empty())
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
        let mut nominal_reported = false;
        if let Some(scheme) = schemes.get(id.path(db)) {
            if names_incomplete_ability_argument(db, *scheme.body(db), &arguments) {
                errors.push(InstanceError {
                    node: func.id,
                    kind: InstanceErrorKind::IncompleteAbilityArgument,
                });
            }
            nominal_reported =
                names_incomplete_nominal_argument(db, &generic, *scheme.body(db), &arguments);
            if nominal_reported {
                errors.push(InstanceError {
                    node: func.id,
                    kind: InstanceErrorKind::IncompleteNominalArgument,
                });
            }
        }
        for expr in nodes {
            if let Some(ty) = expressions.get(|table| &table.node_types, &expr.id) {
                if names_incomplete_ability_argument(db, *ty, &arguments) {
                    errors.push(InstanceError {
                        node: expr.id,
                        kind: InstanceErrorKind::IncompleteAbilityArgument,
                    });
                }
                // One argument shows up in every type that names it, so
                // only its first occurrence in a function is reported.
                if !nominal_reported
                    && names_incomplete_nominal_argument(db, &generic, *ty, &arguments)
                {
                    nominal_reported = true;
                    errors.push(InstanceError {
                        node: expr.id,
                        kind: InstanceErrorKind::IncompleteNominalArgument,
                    });
                }
            }
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
            if concrete.is_empty() && functions.get(target).is_some_and(|(_, own)| !own) {
                continue;
            }
            queue.push((*target, concrete));
        }
    }
    errors.sort_by_key(|error| error.node);
    errors.dedup();
    errors
}
