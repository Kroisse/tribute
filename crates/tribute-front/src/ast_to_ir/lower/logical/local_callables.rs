//! Bounded local callable materialization, before any source lambda is emitted.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::dialect::tribute_control;
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{TypeRef, ValueRef};

use crate::ast::{
    Expr, ExprKind, LocalId, NodeId, Pattern, PatternKind, ResolvedRef, Stmt, Type, TypeKind,
    TypedRef,
};
use crate::ast_to_ir::context::IrLoweringCtx;
use crate::ast_to_ir::lower::{FuncSignature, IrBuilder};
use crate::monomorphize::is_concrete_type;
use crate::typeck::LambdaSignature;

use super::callable::{RowArguments, lower_lambda};
use super::declarations::Declarations;

type Key<'db> = (NodeId, Type<'db>, TypeRef);

#[derive(Default)]
pub(super) struct Plan<'db> {
    requests: HashMap<NodeId, Vec<Key<'db>>>,
    uses: HashMap<NodeId, Key<'db>>,
}

impl<'db> Plan<'db> {
    pub(super) fn collect(
        ctx: &mut IrLoweringCtx<'db>,
        ir: &mut IrContext,
        body: &Expr<TypedRef<'db>>,
        declarations: &Declarations<'db>,
        parent_type_parameters: usize,
    ) -> Self {
        let mut nodes = Vec::new();
        body.for_each(|expr| nodes.push(expr));
        let mut origins: HashMap<LocalId, (NodeId, &Expr<TypedRef<'db>>)> = HashMap::default();
        let mut bindings = HashMap::default();
        let mut named = HashMap::default();
        let mut aliases = HashSet::default();
        for expr in &nodes {
            if let ExprKind::Block { stmts, .. } = &*expr.kind {
                for stmt in stmts {
                    if let Stmt::Let { pattern, value, .. } = stmt
                        && let PatternKind::Bind {
                            local_id: Some(local),
                            ..
                        } = &*pattern.kind
                    {
                        bindings.insert(*local, pattern.id);
                        match &*value.kind {
                            ExprKind::Lambda { .. } => {
                                origins.insert(*local, (pattern.id, value));
                            }
                            ExprKind::Var(reference) => {
                                match reference.resolved {
                                    ResolvedRef::Function { id } => {
                                        named.insert(*local, (id, value.id));
                                    }
                                    ResolvedRef::Local { id, .. } => {
                                        if let Some(function) = named.get(&id).copied() {
                                            named.insert(*local, function);
                                        }
                                    }
                                    _ => {}
                                }
                                if let ResolvedRef::Local { id, .. } = reference.resolved
                                    && let Some(origin) = origins.get(&id).copied()
                                {
                                    origins.insert(*local, origin);
                                    aliases.insert(value.id);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        // Named consumers may retain an open worker ABI even when this source
        // call's checked effect instance is pure.
        let mut expected = HashMap::default();
        for expr in &nodes {
            if let ExprKind::Call { callee, args, .. } = &*expr.kind {
                let parameters = match &*callee.kind {
                    ExprKind::Var(reference) => match reference.resolved {
                        ResolvedRef::Function { id } => {
                            instantiated_parameters(ctx, ir, declarations, id, callee.id)
                        }
                        ResolvedRef::Local { id, .. } if named.contains_key(&id) => {
                            let (function, reference) = named[&id];
                            instantiated_parameters(ctx, ir, declarations, function, reference)
                        }
                        _ => match reference.ty.kind(ctx.db) {
                            TypeKind::Func { params, .. } => Some(
                                params
                                    .iter()
                                    .map(|ty| ctx.convert_logical_type(ir, *ty))
                                    .collect(),
                            ),
                            _ => None,
                        },
                    },
                    _ => None,
                };
                if let Some(parameters) = parameters {
                    for (arg, ty) in args.iter().zip(parameters) {
                        expected.insert(arg.id, ty);
                    }
                }
            }
        }
        let mut plan = Self::default();
        let mut unsupported = HashSet::default();
        for expr in nodes {
            let ExprKind::Var(reference) = &*expr.kind else {
                continue;
            };
            let ResolvedRef::Local { id, .. } = reference.resolved else {
                continue;
            };
            if aliases.contains(&expr.id) {
                continue;
            }
            let Some((binding, lambda)) = origins.get(&id).copied() else {
                continue;
            };
            let Some(instance) = declarations.local_instances.get(&expr.id) else {
                unsupported.insert(binding);
                continue;
            };
            let Some(seed) = declarations.lambda_signatures.get(&lambda.id) else {
                continue;
            };
            let ExprKind::Lambda { body, .. } = &*lambda.kind else {
                unreachable!()
            };
            let root = declarations.local_instances.values().find(|instance| {
                instance.binding == binding && bindings.get(&instance.local) == Some(&binding)
            });
            if instance.local != id
                || bindings.get(&id) != Some(&instance.binding)
                || instance.callable != reference.ty
                || root.is_none_or(|root| {
                    root.scheme.body(ctx.db) != seed.function_type
                        || !fixed_instance(
                            ctx,
                            body,
                            seed,
                            root,
                            declarations,
                            parent_type_parameters,
                        )
                })
                || !fixed_instance(
                    ctx,
                    body,
                    seed,
                    instance,
                    declarations,
                    parent_type_parameters,
                )
            {
                unsupported.insert(binding);
                continue;
            }
            let required = expected
                .get(&expr.id)
                .copied()
                .unwrap_or_else(|| ctx.convert_logical_type(ir, instance.callable));
            let selected = ctx.convert_logical_type(ir, instance.callable);
            let (Some(wanted), Some(actual)) = (
                tribute_control::FuncSig::from_type_ref(ir, required),
                tribute_control::FuncSig::from_type_ref(ir, selected),
            ) else {
                unsupported.insert(binding);
                continue;
            };
            if wanted.inputs(ir) != actual.inputs(ir)
                || wanted.result(ir) != actual.result(ir)
                || !wanted.row(ir).includes(&actual.row(ir))
            {
                unsupported.insert(binding);
                continue;
            }
            let key = (binding, instance.callable, required);
            let requests = plan.requests.entry(binding).or_default();
            if !requests.contains(&key) {
                requests.push(key);
            }
            plan.uses.insert(expr.id, key);
        }
        // Never replace a binding with one chosen instance while leaving an
        // unplanned use to consume that instance as the original value.
        plan.requests
            .retain(|binding, _| !unsupported.contains(binding));
        plan.uses.retain(|_, key| !unsupported.contains(&key.0));
        plan
    }
}

/// Parent-owned variables are fixed within this body, including the retained
/// generic template. Local quantifiers and unsolved variables are not fixed.
fn fixed_in_parent<'db>(db: &'db dyn salsa::Database, ty: Type<'db>, parameters: usize) -> bool {
    let fixed = |ty| fixed_in_parent(db, ty, parameters);
    let fixed_row = |row: crate::ast::AbilityRow<'db>| {
        row.rest(db).is_none()
            && row
                .instances(db)
                .iter()
                .all(|effect| effect.args.iter().copied().all(fixed))
    };
    match ty.kind(db) {
        TypeKind::BoundVar { index } => (*index as usize) < parameters,
        TypeKind::Named { args, .. } | TypeKind::Tuple(args) => args.iter().copied().all(fixed),
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => params.iter().copied().all(fixed) && fixed(*result) && fixed_row(*effect),
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => fixed(*arg) && fixed(*result) && fixed_row(*effect),
        _ => is_concrete_type(db, ty),
    }
}

/// This boundary instantiates only a lambda's latent row. If that row also
/// occurs in body metadata, general row-specialization owns the transformation.
fn fixed_instance<'db>(
    ctx: &IrLoweringCtx<'db>,
    body: &Expr<TypedRef<'db>>,
    seed: &LambdaSignature<'db>,
    instance: &crate::typeck::LocalCallableInstance<'db>,
    declarations: &Declarations<'db>,
    parent_type_parameters: usize,
) -> bool {
    let db = ctx.db;
    if !instance.scheme.type_params(db).is_empty()
        || instance.scheme.effect_params(db).len() != instance.row_arguments.len()
    {
        return false;
    }
    let (
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        },
        TypeKind::Func {
            params: actual_params,
            result: actual_result,
            ..
        },
    ) = (seed.function_type.kind(db), instance.callable.kind(db))
    else {
        return false;
    };
    if params != actual_params
        || result != actual_result
        || !params
            .iter()
            .chain(std::iter::once(result))
            .all(|ty| fixed_in_parent(db, *ty, parent_type_parameters))
    {
        return false;
    }
    let mut next_row = crate::ast::collect_row_vars(db, instance.scheme.body(db))
        .into_iter()
        .chain(instance.scheme.effect_params(db).iter().copied())
        .chain(instance.row_arguments.iter().flat_map(|row| {
            let wrapper = Type::new(
                db,
                TypeKind::Func {
                    params: vec![],
                    result: Type::new(db, TypeKind::Nil),
                    effect: *row,
                },
            );
            crate::ast::collect_row_vars(db, wrapper)
        }))
        .map(|var| var.id)
        .max()
        .unwrap_or(0)
        + 1;
    let fresh: Vec<_> = instance
        .row_arguments
        .iter()
        .map(|_| {
            let var = crate::ast::RowVar { id: next_row };
            next_row += 1;
            var
        })
        .collect();
    // Only this local scheme's rows are owned here. BoundVars in its body
    // belong to the enclosing function and must survive row instantiation.
    let mut freshening = crate::typeck::RowSubst::new();
    for (owned, fresh) in instance.scheme.effect_params(db).iter().zip(&fresh) {
        freshening.insert(owned.id, crate::ast::AbilityRow::open(db, *fresh));
    }
    let selected =
        crate::typeck::TypeSubst::new().apply_with_rows(db, instance.scheme.body(db), &freshening);
    let mut rows = crate::typeck::RowSubst::new();
    for (var, row) in fresh.into_iter().zip(&instance.row_arguments) {
        rows.insert(var.id, *row);
    }
    if crate::typeck::TypeSubst::new().apply_with_rows(db, selected, &rows) != instance.callable {
        return false;
    }
    let TypeKind::Func {
        effect: scheme_effect,
        ..
    } = instance.scheme.body(db).kind(db)
    else {
        return false;
    };
    // Aliases can rename an owned latent row, but cannot introduce effects or
    // turn a free environment row into a newly quantified row.
    if scheme_effect.instances(db) != effect.instances(db)
        || match scheme_effect.rest(db) {
            Some(row) => instance.scheme.effect_params(db) != [row],
            None => !instance.scheme.effect_params(db).is_empty(),
        }
    {
        return false;
    }
    let changes_row = |ty| {
        effect
            .rest(db)
            .is_some_and(|row| crate::ast::collect_row_vars(db, ty).contains(&row))
    };
    let mut independent = true;
    body.for_each(|expr| {
        independent &= !ctx
            .get_node_type(expr.id)
            .is_some_and(|ty| changes_row(*ty));
        if let ExprKind::Var(reference) = &*expr.kind {
            independent &= !changes_row(reference.ty);
        }
        if let Some(signature) = declarations.lambda_signatures.get(&expr.id) {
            independent &= !changes_row(signature.function_type);
        }
        if let Some(operation) = declarations.perform_operations.get(&expr.id) {
            independent &= !operation
                .params
                .iter()
                .chain(&operation.ability_args)
                .chain(std::iter::once(&operation.result))
                .any(|ty| changes_row(*ty));
        }
        if let ExprKind::Handle { handlers, .. } = &*expr.kind {
            for handler in handlers {
                if let Some(operation) = declarations.handler_operations.get(&handler.id) {
                    independent &= !operation
                        .params
                        .iter()
                        .chain(&operation.ability_args)
                        .chain(std::iter::once(&operation.result))
                        .any(|ty| changes_row(*ty));
                }
            }
        }
        // A resume carrier must not be cloned into independent capture paths.
        if matches!(&*expr.kind, ExprKind::Resume { .. }) {
            independent = false;
        }
    });
    independent
}

pub(super) fn materialize<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    pattern: &Pattern<TypedRef<'db>>,
    lambda: &Expr<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let requests = declarations
        .local_callables
        .requests
        .get(&pattern.id)?
        .clone();
    let ExprKind::Lambda { params, body } = &*lambda.kind else {
        return None;
    };
    let mut first = None;
    for key in requests {
        let signature = LambdaSignature {
            function_type: key.1,
        };
        let value = lower_lambda(
            builder,
            builder.location(lambda.id),
            signature,
            params.clone(),
            body.clone(),
            declarations,
            Some(key.2),
        )?;
        builder.ctx.bind_local_callable(key, value);
        first.get_or_insert(value);
    }
    first
}

pub(super) fn lookup<'db>(
    ctx: &IrLoweringCtx<'db>,
    node: NodeId,
    declarations: &Declarations<'db>,
) -> Option<ValueRef> {
    declarations
        .local_callables
        .uses
        .get(&node)
        .and_then(|key| ctx.lookup_local_callable(*key))
}

/// Source order of actual external SSA uses, with duplicates removed.
pub(super) fn captures(
    ctx: &IrLoweringCtx<'_>,
    ir: &IrContext,
    region: trunk_ir::refs::RegionRef,
) -> Vec<ValueRef> {
    fn collect(
        ir: &IrContext,
        region: trunk_ir::refs::RegionRef,
        defined: &mut HashSet<ValueRef>,
        uses: &mut Vec<ValueRef>,
    ) {
        for block in &ir.region(region).blocks {
            for index in 0..ir.block(*block).args.len() {
                defined.insert(ir.block_arg(*block, index as u32));
            }
            for op in &ir.block(*block).ops {
                defined.extend(ir.op_results(*op));
                uses.extend(ir.op_operands(*op));
                for nested in ir.op_regions(*op) {
                    collect(ir, nested, defined, uses);
                }
            }
        }
    }
    let mut defined = HashSet::default();
    let mut uses = Vec::new();
    collect(ir, region, &mut defined, &mut uses);
    let mut seen = HashSet::default();
    let external: HashSet<_> = uses
        .iter()
        .copied()
        .filter(|value| !defined.contains(value))
        .collect();
    ctx.all_bindings()
        .map(|(_, _, value)| value)
        .chain(uses)
        .filter(|value| external.contains(value) && seen.insert(*value))
        .collect()
}

/// The parameter types of `function` as the reference `reference` sees them.
fn instantiated_parameters<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    declarations: &Declarations<'db>,
    function: crate::ast::FuncDefId<'db>,
    reference: NodeId,
) -> Option<Vec<trunk_ir::refs::TypeRef>> {
    let name = function.path(ctx.db).clone();
    let declared = FuncSignature::lookup_logical(ctx, ir, &name)?;
    let args = RowArguments::Instance(reference).rows(ctx, declarations, &name, &declared);
    Some(declared.instantiate(ir, &args).param_types)
}
