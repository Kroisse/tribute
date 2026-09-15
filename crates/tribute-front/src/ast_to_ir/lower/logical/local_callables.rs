//! Bounded local callable materialization, before any source lambda is emitted.
use super::*;
use crate::ast::{LocalId, NodeId, Type};
use crate::monomorphize::{is_concrete_type, walk_typed_expr};
use crate::typeck::LambdaSignature;

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
        walk_typed_expr(body, &mut |expr| nodes.push(expr));
        let mut origins: HashMap<LocalId, (NodeId, &Expr<TypedRef<'db>>)> = HashMap::new();
        let mut bindings = HashMap::new();
        let mut named = HashMap::new();
        let mut aliases = HashSet::new();
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
                                        named.insert(*local, id);
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
        let mut expected = HashMap::new();
        for expr in &nodes {
            if let ExprKind::Call { callee, args } = &*expr.kind {
                let parameters = match &*callee.kind {
                    ExprKind::Var(reference) => match reference.resolved {
                        ResolvedRef::Function { id } => {
                            FuncSignature::lookup_logical(ctx, ir, id.qualified(ctx.db))
                                .map(|signature| signature.param_types)
                        }
                        ResolvedRef::Local { id, .. } if named.contains_key(&id) => {
                            FuncSignature::lookup_logical(ctx, ir, named[&id].qualified(ctx.db))
                                .map(|signature| signature.param_types)
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
        let mut unsupported = HashSet::new();
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
                || tribute_control::func_sig_convention(ir, required)
                    < tribute_control::func_sig_convention(ir, selected)
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
    let fixed_row = |row: crate::ast::EffectRow<'db>| {
        row.rest(db).is_none()
            && row
                .effects(db)
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
            minimum_convention,
            effect,
            ..
        },
        TypeKind::Func {
            params: actual_params,
            result: actual_result,
            minimum_convention: actual_minimum,
            ..
        },
    ) = (seed.function_type.kind(db), instance.callable.kind(db))
    else {
        return false;
    };
    if params != actual_params
        || result != actual_result
        || minimum_convention > actual_minimum
        || !params
            .iter()
            .chain(std::iter::once(result))
            .all(|ty| fixed_in_parent(db, *ty, parent_type_parameters))
    {
        return false;
    }
    let mut next_row = crate::ast::collect_effect_vars(db, instance.scheme.body(db))
        .into_iter()
        .chain(instance.scheme.effect_params(db).iter().copied())
        .chain(instance.row_arguments.iter().flat_map(|row| {
            let wrapper = Type::new(
                db,
                TypeKind::Func {
                    params: vec![],
                    result: Type::new(db, TypeKind::Nil),
                    effect: *row,
                    minimum_convention: CallingConvention::Direct,
                },
            );
            crate::ast::collect_effect_vars(db, wrapper)
        }))
        .map(|var| var.id)
        .max()
        .unwrap_or(0)
        + 1;
    let fresh: Vec<_> = instance
        .row_arguments
        .iter()
        .map(|_| {
            let var = crate::ast::EffectVar { id: next_row };
            next_row += 1;
            var
        })
        .collect();
    // Only this local scheme's rows are owned here. BoundVars in its body
    // belong to the enclosing function and must survive row instantiation.
    let mut freshening = crate::typeck::RowSubst::new();
    for (owned, fresh) in instance.scheme.effect_params(db).iter().zip(&fresh) {
        freshening.insert(owned.id, crate::ast::EffectRow::open(db, *fresh));
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
    if scheme_effect.effects(db) != effect.effects(db)
        || match scheme_effect.rest(db) {
            Some(row) => instance.scheme.effect_params(db) != &[row],
            None => !instance.scheme.effect_params(db).is_empty(),
        }
    {
        return false;
    }
    let changes_row = |ty| {
        effect
            .rest(db)
            .is_some_and(|row| crate::ast::collect_effect_vars(db, ty).contains(&row))
    };
    let mut independent = true;
    walk_typed_expr(body, &mut |expr| {
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
            convention: builder.ctx.calling_convention_for_type(key.1)?,
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
                for nested in &ir.op(*op).regions {
                    collect(ir, *nested, defined, uses);
                }
            }
        }
    }
    let mut defined = HashSet::new();
    let mut uses = Vec::new();
    collect(ir, region, &mut defined, &mut uses);
    let mut seen = HashSet::new();
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
