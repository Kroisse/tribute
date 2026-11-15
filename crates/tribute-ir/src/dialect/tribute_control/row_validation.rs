//! Whole-IR validation of row scopes, local quantification, and evidence
//! tails.

use super::*;

/// The ability instances the module tree declares.
pub(super) fn declared_instances(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
    let mut instances = HashSet::default();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if crate::dialect::ability::Decl::matches(ctx, op)
            && let Some(path) = qualified_name(ctx, op)
        {
            instances.insert(path);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    instances
}

/// Check one row against the binders in scope, `None` where no binder is.
fn check_row(
    ctx: &IrContext,
    owner: OpRef,
    row: &Row,
    binders: Option<usize>,
    instances: &HashSet<SymbolPath>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(tail) = row.tail
        && binders.is_none_or(|binders| usize::try_from(tail).is_ok_and(|tail| tail >= binders))
    {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("row tail {tail} names no row binder in scope"),
        );
    }
    for instance in &row.instances {
        if !instances.contains(instance) {
            push_op_error(
                ctx,
                owner,
                errors,
                format!("row instance @{instance} is not an ability.decl"),
            );
        }
    }
}

fn check_type_rows(
    ctx: &IrContext,
    owner: OpRef,
    ty: TypeRef,
    binders: Option<usize>,
    instances: &HashSet<SymbolPath>,
    errors: &mut Vec<ValidationError>,
) {
    visit_rows(ctx, ty, &mut |row| match row {
        Ok(row) => check_row(ctx, owner, &row, binders, instances, errors),
        Err(message) => push_op_error(ctx, owner, errors, message),
    });
}

/// Every row tail inside a definition names one of its binders, and every
/// row instance is declared. Module-level types have no binders.
/// Every `local` binder is quantified by exactly one lambda, and every use
/// of such a lambda is an `instantiate` whose result is the lambda's type
/// with its row arguments substituted.
pub(super) fn validate_local_quantification(
    ctx: &IrContext,
    module: Module,
    errors: &mut Vec<ValidationError>,
) {
    for &func in module.ops(ctx).iter() {
        if !SourceFunc::is_set(ctx, func) {
            continue;
        }
        let (Ok(binders), Some(body)) = (row_binders(ctx, func), ctx.op_region(func, 0)) else {
            continue;
        };
        let mut owners = HashSet::default();
        walk_region_ops(ctx, body, &mut |op| {
            if SourceLambda::is_set(ctx, op) {
                let quantified = match quantified_rows(ctx, op) {
                    Ok(quantified) => quantified,
                    Err(message) => return push_op_error(ctx, op, errors, message),
                };
                for binder in &quantified {
                    let kind = usize::try_from(*binder)
                        .ok()
                        .and_then(|index| binders.get(index));
                    if kind != Some(&RowBinder::Local) {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            format!("lambda quantifies binder {binder}, which is not local"),
                        );
                    } else if !owners.insert(*binder) {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            format!("local binder {binder} is quantified by two lambdas"),
                        );
                    }
                }
                let result = ctx.op_results(op).first().copied();
                let direct_use = !quantified.is_empty()
                    && result.is_some_and(|result| {
                        ctx.uses(result)
                            .iter()
                            .any(|used| !Instantiate::matches(ctx, used.user))
                    });
                if direct_use {
                    push_op_error(
                        ctx,
                        op,
                        errors,
                        "a lambda that quantifies rows is used only through instantiate",
                    );
                }
            } else if Instantiate::matches(ctx, op) {
                validate_instantiate(ctx, op, errors);
            }
        });
        for (binder, kind) in (0u32..).zip(&binders) {
            if *kind == RowBinder::Local && !owners.contains(&binder) {
                push_op_error(
                    ctx,
                    func,
                    errors,
                    format!("local binder {binder} is quantified by no lambda"),
                );
            }
        }
    }
}

fn validate_instantiate(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let lambda = ctx
        .op_operands(op)
        .first()
        .and_then(|&operand| match ctx.value_def(operand) {
            ValueDef::OpResult(producer, _) if SourceLambda::is_set(ctx, producer) => {
                Some(producer)
            }
            _ => None,
        });
    let quantified = lambda
        .and_then(|lambda| quantified_rows(ctx, lambda).ok())
        .filter(|quantified| !quantified.is_empty());
    let (Some(lambda), Some(quantified)) = (lambda, quantified) else {
        return push_op_error(
            ctx,
            op,
            errors,
            "instantiate takes a lambda that quantifies rows",
        );
    };
    let Ok(args) = row_args(ctx, op) else {
        return;
    };
    if args.len() != quantified.len() {
        return push_op_error(
            ctx,
            op,
            errors,
            format!(
                "instantiate passes {} row arguments for {} quantified binders",
                args.len(),
                quantified.len()
            ),
        );
    }
    let expected = ctx
        .op_result_types(lambda)
        .first()
        .and_then(|&ty| find_substituted_rows(ctx, ty, RowArgs::Binders(&quantified, &args)));
    if expected.is_none() || ctx.op_result_types(op).first().copied() != expected {
        push_op_error(
            ctx,
            op,
            errors,
            "instantiate result is not the lambda's type with its row arguments substituted",
        );
    }
}

pub(super) fn validate_row_scopes(
    ctx: &IrContext,
    module: Module,
    instances: &HashSet<SymbolPath>,
    errors: &mut Vec<ValidationError>,
) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if crate::dialect::ability::Operation::matches(ctx, op) {
            let mut types = Vec::new();
            for attr in ctx.op(op).attributes.values() {
                attr.visit_types(&mut |ty| types.push(ty));
            }
            for ty in types {
                check_type_rows(ctx, op, ty, None, instances, errors);
            }
            return ControlFlow::Continue(WalkAction::Advance);
        }
        if !SourceFunc::is_set(ctx, op) {
            return ControlFlow::Continue(WalkAction::Advance);
        }
        let Ok(binders) = row_binders(ctx, op) else {
            return ControlFlow::Continue(WalkAction::Skip);
        };
        let scope = Some(binders.len());
        let mut types = HashSet::default();
        if let Some(ty) = ctx.op_attrs(op).get_type("type") {
            types.insert(ty);
        }
        let mut rows = Vec::new();
        if let Some(body) = ctx.op_region(op, 0) {
            walk_region_ops(ctx, body, &mut |inner| {
                types.extend(ctx.op_result_types(inner).iter().copied());
                for attr in ctx.op(inner).attributes.values() {
                    attr.visit_types(&mut |ty| {
                        types.insert(ty);
                    });
                }
                for region in ctx.op_regions(inner) {
                    for &block in &ctx.region(region).blocks {
                        types.extend(block_arg_types(ctx, block));
                    }
                }
                if let Ok(args) = row_args(ctx, inner) {
                    rows.extend(args.into_iter().map(|row| (inner, row)));
                }
            });
            for &block in &ctx.region(body).blocks {
                types.extend(block_arg_types(ctx, block));
            }
        }
        for ty in types {
            check_type_rows(ctx, op, ty, scope, instances, errors);
        }
        for (inner, row) in rows {
            check_row(ctx, inner, &row, scope, instances, errors);
        }
        ControlFlow::Continue(WalkAction::Skip)
    });
}

/// The row tail of the callable whose body holds `op`, and the function
/// that declares the binders it names.
fn enclosing_row_tail(ctx: &IrContext, op: OpRef) -> Option<(Option<u32>, OpRef)> {
    let mut current = op;
    let mut tail = None;
    loop {
        let owner = parent_op(ctx, current)?;
        let signature_tail = || {
            let ty = DynSourceCallable::get(ctx, owner)?.callable_signature(ctx)?;
            Some(FuncSig::from_type_ref(ctx, ty)?.row(ctx).tail)
        };
        if SourceLambda::is_set(ctx, owner) && tail.is_none() {
            tail = Some(signature_tail()?);
        }
        if SourceFunc::is_set(ctx, owner) {
            let tail = match tail {
                Some(tail) => tail,
                None => signature_tail()?,
            };
            return Some((tail, owner));
        }
        current = owner;
    }
}

/// The sources of the union `func` declares for `tail`.
fn union_sources(ctx: &IrContext, func: OpRef, tail: Option<u32>) -> Option<Vec<u32>> {
    let tail = tail?;
    row_unions(ctx, func)
        .ok()?
        .into_iter()
        .find(|union| union.result == tail)
        .map(|union| union.sources)
}

fn check_selects(steps: &[EvidenceStep], sources: Option<&[u32]>) -> Result<(), String> {
    for step in steps {
        match step {
            EvidenceStep::Select(index) => {
                let in_range = sources.is_some_and(|sources| {
                    usize::try_from(*index).is_ok_and(|index| index < sources.len())
                });
                if !in_range {
                    return Err(format!(
                        "evidence_plan select {index} names no source of the union for the \
                         callable's row tail"
                    ));
                }
            }
            EvidenceStep::Tails(plans) => {
                for plan in plans {
                    check_selects(plan, sources)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// `select` and `tails` count the sources of row tail unions.
pub(super) fn validate_evidence_tails(
    ctx: &IrContext,
    body: RegionRef,
    symbols: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        let Some(Attribute::List(items)) = ctx.op_attrs(op).get(EVIDENCE_PLAN_ATTR) else {
            return;
        };
        let Ok(steps) = items
            .iter()
            .map(|step| EvidenceStep::from_attribute(ctx, step))
            .collect::<Result<Vec<_>, _>>()
        else {
            return;
        };
        let Some((tail, func)) = enclosing_row_tail(ctx, op) else {
            return;
        };
        let sources = union_sources(ctx, func, tail);
        if let Err(message) = check_selects(&steps, sources.as_deref()) {
            push_op_error(ctx, op, errors, message);
        }
        let Some(EvidenceStep::Tails(plans)) = steps.last() else {
            return;
        };
        let callee = ctx
            .op_attrs(op)
            .get_symbol_ref("callee")
            .and_then(|symbol| symbols.definitions(ctx, op, symbol).first().copied())
            .filter(|&callee| SourceFunc::is_set(ctx, callee));
        let Some(callee) = callee else {
            return;
        };
        let callee_tail = ctx
            .op_attrs(callee)
            .get_type("type")
            .and_then(|ty| FuncSig::from_type_ref(ctx, ty))
            .and_then(|signature| signature.row(ctx).tail);
        let expected = union_sources(ctx, callee, callee_tail).map(|sources| sources.len());
        if expected != Some(plans.len()) {
            push_op_error(
                ctx,
                op,
                errors,
                "evidence_plan tails must hold one plan for each source of the callee's row \
                 tail union",
            );
        }
    });
}
