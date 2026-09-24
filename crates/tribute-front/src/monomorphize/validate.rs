//! Validate checker-selected instances on executable reference paths before erasure.
use super::{MonomorphizeMetadata, collect::is_concrete_type};
use crate::ast::{
    Decl, Expr, ExprKind, FuncDecl, FuncDefId, Module, NodeId, ResolvedRef, Stmt, Type, TypeScheme,
    TypedRef,
};
use std::collections::{HashMap, HashSet};
use trunk_ir::Symbol;

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
    module: &Module<TypedRef<'db>>,
    schemes: &HashMap<Symbol, TypeScheme<'db>>,
    metadata: &MonomorphizeMetadata<'db>,
) -> Vec<InstanceError> {
    fn declarations<'a, 'db>(
        db: &'db dyn salsa::Database,
        decls: &'a [Decl<TypedRef<'db>>],
        prefix: &mut String,
        functions: &mut HashMap<FuncDefId<'db>, &'a FuncDecl<TypedRef<'db>>>,
        fields: &mut HashSet<(crate::ast::TypeDefId<'db>, Symbol)>,
    ) {
        for decl in decls {
            match decl {
                Decl::Function(func) => {
                    let name = crate::qualified_symbol(prefix, func.name);
                    functions.insert(FuncDefId::new(db, name), func);
                }
                Decl::Struct(decl) => {
                    let name = crate::qualified_symbol(prefix, decl.name);
                    let owner = crate::ast::TypeDefId::source(db, name, decl.id);
                    for field in &decl.fields {
                        if let Some(name) = field.name {
                            fields.insert((owner, name));
                        }
                    }
                }
                Decl::Module(module) => {
                    if let Some(body) = &module.body {
                        let saved = crate::push_prefix(prefix, module.name);
                        declarations(db, body, prefix, functions, fields);
                        prefix.truncate(saved);
                    }
                }
                _ => {}
            }
        }
    }
    let mut functions = HashMap::new();
    let mut fields = HashSet::new();
    declarations(
        db,
        &module.decls,
        &mut String::new(),
        &mut functions,
        &mut fields,
    );
    let mut queue: Vec<_> = functions
        .iter()
        .filter(|(id, func)| {
            id.qualified(db) == Symbol::new("main")
                || (func.is_pub
                    && schemes
                        .get(&id.qualified(db))
                        .is_some_and(|scheme| scheme.type_params(db).is_empty()))
        })
        .map(|(id, _)| (*id, Vec::new()))
        .collect();
    let mut seen = HashSet::new();
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
        walk(&func.body, &mut |node| nodes.push(node));
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
            let known_declaration = match instance.origin {
                crate::typeck::FunctionInstanceOrigin::Declaration => {
                    schemes.get(&target.qualified(db)) == Some(&instance.scheme)
                }
                crate::typeck::FunctionInstanceOrigin::FieldAccessor { owner, field } => {
                    let mut prefix = owner.qualified(db).to_string();
                    fields.contains(&(owner, field))
                        && target.qualified(db) == crate::qualified_symbol(&mut prefix, field)
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
            solver.reserve_effect_vars_in_type(callable);
            for row in &instance.row_arguments {
                solver.reserve_effect_vars_in_row(*row);
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
                if row
                    .effects(db)
                    .iter()
                    .flat_map(|effect| &effect.args)
                    .any(|ty| substitute(db, *ty, &arguments).is_none())
                {
                    invalid_row = true;
                    break;
                }
                constraints.add_row_eq(
                    *variable,
                    crate::typeck::subst::substitute_effect_row(db, *row, &arguments),
                );
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

pub(crate) fn walk<'a, 'db>(
    expr: &'a Expr<TypedRef<'db>>,
    visit: &mut impl FnMut(&'a Expr<TypedRef<'db>>),
) {
    visit(expr);
    match expr.kind.as_ref() {
        ExprKind::Call { callee, args } => {
            walk(callee, visit);
            for arg in args {
                walk(arg, visit);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Let { value, .. } => walk(value, visit),
                    Stmt::Expr { expr, .. } => walk(expr, visit),
                }
            }
            walk(value, visit);
        }
        ExprKind::Case { scrutinee, arms } => {
            walk(scrutinee, visit);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    walk(guard, visit);
                }
                walk(&arm.body, visit);
            }
        }
        ExprKind::Lambda { body, .. } => walk(body, visit),
        ExprKind::Handle { body, handlers } => {
            walk(body, visit);
            for arm in handlers {
                walk(&arm.body, visit);
            }
        }
        ExprKind::Resume { arg, .. } => walk(arg, visit),
        ExprKind::Cons { args, .. } | ExprKind::Tuple(args) | ExprKind::List(args) => {
            for arg in args {
                walk(arg, visit);
            }
        }
        ExprKind::Record { fields, spread, .. } => {
            for (_, expr) in fields {
                walk(expr, visit);
            }
            if let Some(spread) = spread {
                walk(spread, visit);
            }
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            walk(lhs, visit);
            walk(rhs, visit);
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            walk(receiver, visit);
            for arg in args {
                walk(arg, visit);
            }
        }
        _ => {}
    }
}
