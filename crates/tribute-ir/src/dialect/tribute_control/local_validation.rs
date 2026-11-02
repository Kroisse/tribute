//! Local validation: each operation's placement and regions.

use super::*;

/// That `op`'s row unions name its binders and each declares a distinct
/// result.
pub(crate) fn verify_row_declarations(ctx: &IrContext, op: OpRef) -> Result<(), String> {
    let binders = row_binders(ctx, op)?;
    let mut results = HashSet::default();
    for union in row_unions(ctx, op)? {
        let in_scope = |index: u32| usize::try_from(index).is_ok_and(|index| index < binders.len());
        if !in_scope(union.result) || !union.sources.iter().copied().all(in_scope) {
            return Err("row union names a tail that is not a binder".into());
        }
        if union.sources.contains(&union.result) {
            return Err("row union lists its result among its sources".into());
        }
        if !results.insert(union.result) {
            return Err(format!(
                "row tail {} is declared as a union twice",
                union.result
            ));
        }
    }
    Ok(())
}

/// That no operand or result type of `op` is unresolved.
pub(crate) fn verify_resolved_types(ctx: &IrContext, op: OpRef) -> Result<(), &'static str> {
    let unresolved = value_types(ctx, ctx.op_operands(op))
        .into_iter()
        .chain(ctx.op_result_types(op).iter().copied())
        .any(|ty| contains_unresolved_type(ctx, ty, &mut HashSet::default()));
    if unresolved {
        Err("operand and result types must be resolved")
    } else {
        Ok(())
    }
}

pub(crate) fn validate_func(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let ty = ctx
        .op_attrs(op)
        .get_type("type")
        .expect("schema-verified source function");
    if let Some(body) = ctx.op_region(op, 0) {
        validate_callable_body(ctx, op, ty, body, errors);
        validate_func_isolation(ctx, op, body, errors);
    }
}

pub(crate) fn validate_lambda(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let ty = ctx.op_result_types(op)[0];
    for body in ctx.op_regions(op).collect::<Vec<_>>() {
        validate_callable_body(ctx, op, ty, body, errors);
    }
}

pub(super) fn validate_return(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    let Some(owner) = parent_op(ctx, op) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a tribute_control.func or lambda body",
        );
        return;
    };
    let callable_ty =
        DynSourceCallable::get(ctx, owner).and_then(|callable| callable.callable_signature(ctx));
    let Some((result, _)) = callable_ty.and_then(|ty| func_sig_parts(ctx, ty)) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a tribute_control.func or lambda body",
        );
        return;
    };
    if let Some(&value) = ctx.op_operands(op).first()
        && ctx.value_ty(value) != result
    {
        push_op_error(
            ctx,
            op,
            errors,
            "operand type does not match enclosing callable result",
        );
    }
}

pub(super) fn is_tail_call(ctx: &IrContext, op: OpRef) -> bool {
    TailCall::matches(ctx, op) || SourceTailCallIndirect::is_set(ctx, op)
}

/// The callable whose body contains `op` and its source signature, or an
/// error when `op` is not in its tail position or a handle or handler lies
/// between them.
pub(super) fn tail_call_owner(
    ctx: &IrContext,
    op: OpRef,
) -> Result<(OpRef, TypeRef), &'static str> {
    let mut exit = op;
    loop {
        let Some(owner) = parent_op(ctx, exit) else {
            return Err("must be inside a tribute_control.func or lambda body");
        };
        if let Some(callable) = DynSourceCallable::get(ctx, owner) {
            return callable
                .callable_signature(ctx)
                .map(|ty| (owner, ty))
                .ok_or("must be inside a tribute_control.func or lambda body");
        }
        if Handle::matches(ctx, owner) || Handler::matches(ctx, owner) {
            return Err("must not be inside a tribute_control.handle body or handler");
        }
        // A structured arm may end with a tail call only when the structured
        // operation is a one-result `scf.if` whose value flows straight into
        // its own block's terminator.
        if !trunk_ir::dialect::scf::If::matches(ctx, owner) || ctx.op_results(owner).len() != 1 {
            return Err("must be in tail position of its callable");
        }
        let Some(block) = ctx.op(owner).parent_block else {
            return Err("must be inside a tribute_control.func or lambda body");
        };
        let ops = &ctx.block(block).ops;
        let position = ops
            .iter()
            .position(|&candidate| candidate == owner)
            .expect("an operation is listed in its parent block");
        let mut rest = ops[position + 1..].iter().copied();
        let terminator =
            rest.find(|&candidate| !core::UnrealizedConversionCast::matches(ctx, candidate));
        let Some(terminator) = terminator else {
            return Err("must be in tail position of its callable");
        };
        if ops.last().copied() != Some(terminator) {
            return Err("must be in tail position of its callable");
        }
        if Return::matches(ctx, terminator) {
            exit = terminator;
            continue;
        }
        if trunk_ir::dialect::scf::Yield::matches(ctx, terminator) {
            exit = owner;
            continue;
        }
        return Err("must be in tail position of its callable");
    }
}

pub(crate) fn validate_tail_call(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    let (owner, owner_ty) = match tail_call_owner(ctx, op) {
        Ok(owner) => owner,
        Err(message) => {
            push_op_error(ctx, op, errors, message);
            return;
        }
    };
    if SourceTailCallIndirect::is_set(ctx, op) {
        let callee_ty = ctx.value_ty(ctx.op_operands(op)[0]);
        let (Some((callee_result, _)), Some((owner_result, _))) = (
            func_sig_parts(ctx, callee_ty),
            func_sig_parts(ctx, owner_ty),
        ) else {
            return;
        };
        if callee_result != owner_result {
            push_op_error(
                ctx,
                op,
                errors,
                "callee result type does not match enclosing callable result",
            );
        }
        if let (Some(callee_cc), Some(owner_cc)) = (
            tribute_cps::declared_convention(ctx, op),
            tribute_cps::convention_at(ctx, owner, op),
        ) && callee_cc > owner_cc
        {
            push_op_error(
                ctx,
                op,
                errors,
                "callee convention must not be stronger than the enclosing callable",
            );
        }
    }
}

pub(super) fn validate_handle(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let Some((body_region, completion_region, handlers_region)) =
        ctx.op_regions(op).collect_tuple()
    else {
        return;
    };
    let Some(body_block) = single_block(ctx, op, body_region, "body", errors) else {
        return;
    };
    let Some(completion_block) = single_block(ctx, op, completion_region, "completion", errors)
    else {
        return;
    };
    let Some(handlers_block) = single_block(ctx, op, handlers_region, "handlers", errors) else {
        return;
    };
    if !ctx.block_args(body_block).is_empty() {
        push_op_error(ctx, op, errors, "body block must not have arguments");
    }
    if ctx.block_args(completion_block).len() != 1 {
        push_op_error(
            ctx,
            op,
            errors,
            "completion block must have exactly one argument",
        );
    }
    if !ctx.block_args(handlers_block).is_empty() {
        push_op_error(
            ctx,
            op,
            errors,
            "handlers table block must not have arguments",
        );
    }
    let body_yield = terminator(ctx, op, body_block, Yield::DEF, errors);
    let completion_yield = terminator(ctx, op, completion_block, Yield::DEF, errors);
    if let (Some(body_yield), Some(&completion_arg)) =
        (body_yield, ctx.block_args(completion_block).first())
        && let Some(&body_value) = ctx.op_operands(body_yield).first()
        && ctx.value_ty(body_value) != ctx.value_ty(completion_arg)
    {
        push_op_error(
            ctx,
            op,
            errors,
            "completion argument type must match the body yield type",
        );
    }
    let handle_result = ctx.op_result_types(op).first().copied();
    if let (Some(result), Some(completion_yield)) = (handle_result, completion_yield)
        && let Some(&value) = ctx.op_operands(completion_yield).first()
        && ctx.value_ty(value) != result
    {
        push_op_error(
            ctx,
            op,
            errors,
            "completion yield type must match handle result type",
        );
    }
    let mut clauses = HashSet::default();
    for child in ctx.block(handlers_block).ops.iter().copied() {
        if !Handler::matches(ctx, child) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("handlers block may contain only tribute_control.handler, found {child}"),
            );
            continue;
        }
        let attrs = ctx.op_attrs(child);
        if let (Some(ability), Some(name)) = (
            attrs.get_symbol_ref("ability"),
            attrs.get_string_ref("op_name"),
        ) && !clauses.insert((ability, name))
        {
            push_op_error(
                ctx,
                child,
                errors,
                format!("duplicate handler clause for {ability}::{}", ctx.str(name)),
            );
        }
        if let Some(result) = handle_result
            && let Some(region) = ctx.op_region(child, 0)
            && let Some(&block) = ctx.region(region).blocks.first()
            && let Some(&yield_op) = ctx.block(block).ops.last()
            && Yield::matches(ctx, yield_op)
            && ctx.op_attrs(child).get_str("kind") == Some("op")
            && let Some(&value) = ctx.op_operands(yield_op).first()
            && ctx.value_ty(value) != result
        {
            push_op_error(
                ctx,
                child,
                errors,
                "general handler yield type must match enclosing handle result type",
            );
        }
    }
}

fn region_contains_resume(ctx: &IrContext, region: RegionRef) -> bool {
    let mut found = false;
    walk_region_ops(ctx, region, &mut |op| {
        found |= Resume::matches(ctx, op);
    });
    found
}

pub(super) fn validate_handler(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let kind = ctx.op_attrs(op).get_str("kind");
    let Some(body) = ctx.op_region(op, 0) else {
        return;
    };
    let Some(block) = single_block(ctx, op, body, "body", errors) else {
        return;
    };
    let yield_op = terminator(ctx, op, block, Yield::DEF, errors);
    let result_type = ctx.op_attrs(op).get_type("operation_result_type");
    let handle_result = parent_op(ctx, op)
        .filter(|parent| Handle::matches(ctx, *parent))
        .and_then(|parent| ctx.op_result_types(parent).first().copied());
    let args = ctx.block_args(block);
    let last_token = args
        .last()
        .copied()
        .and_then(|value| resume_token_parts(ctx, ctx.value_ty(value)));
    match (kind, result_type) {
        (Some(kind), Some(operation_result))
            if kind == "op" && !is_never(ctx, operation_result) =>
        {
            let Some((token_input, token_answer)) = last_token else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resumptive op handler requires a final resume_token block argument",
                );
                return;
            };
            if token_input != operation_result
                || handle_result.is_some_and(|answer| token_answer != answer)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resume token input/answer types do not match operation and handle results",
                );
            }
            if let (Some(yield_op), Some(answer)) = (yield_op, handle_result)
                && let Some(&value) = ctx.op_operands(yield_op).first()
                && ctx.value_ty(value) != answer
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "general handler yield type must match handle result type",
                );
            }
        }
        (Some(kind), Some(operation_result)) => {
            if last_token.is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "fn and op -> core.never handlers must not receive a resume token",
                );
            }
            if kind == "op" && is_never(ctx, operation_result) && region_contains_resume(ctx, body)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "op -> core.never handler must not contain tribute_control.resume",
                );
            }
            let expected = if kind == "fn" {
                Some(operation_result)
            } else {
                handle_result
            };
            if let (Some(yield_op), Some(expected)) = (yield_op, expected)
                && let Some(&value) = ctx.op_operands(yield_op).first()
                && ctx.value_ty(value) != expected
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "handler yield type does not match its kind-specific result",
                );
            }
        }
        _ => {}
    }

    let placed = parent_op(ctx, op).is_some_and(|parent| {
        Handle::matches(ctx, parent)
            && ctx
                .op_region(parent, 2)
                .is_some_and(|handlers| parent_region(ctx, op) == Some(handlers))
    });
    if !placed {
        push_op_error(
            ctx,
            op,
            errors,
            "must be a direct child of a tribute_control.handle handlers region",
        );
    }
}

pub(super) fn validate_yield(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    if let Some(&value) = ctx.op_operands(op).first()
        && ResumeToken::matches(ctx, ctx.value_ty(value))
    {
        push_op_error(ctx, op, errors, "must not yield a resume token");
    }
    let valid_owner = parent_op(ctx, op).is_some_and(|owner| {
        if Handler::matches(ctx, owner) {
            true
        } else if Handle::matches(ctx, owner) {
            let region = parent_region(ctx, op);
            ctx.op_region(owner, 0) == region || ctx.op_region(owner, 1) == region
        } else {
            false
        }
    });
    if !valid_owner {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a handle body/completion or handler body",
        );
    }
}

pub(super) fn validate_local_operation(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
) {
    let data = ctx.op(op);
    let dialect = data.name.dialect(ctx);
    if dialect != DIALECT_NAME && dialect != tribute_cps::DIALECT_NAME {
        return;
    }
    let Some(def) = OpDef::of(ctx, op) else {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "unsupported {} operation '{}'",
                data.name.dialect(ctx),
                data.name.strip_dialect(ctx)
            ),
        );
        return;
    };
    // Counts, attribute kinds, and declared type relations come from the
    // schema; the checks below assume them.
    let violations = def.verify(ctx, op);
    if !violations.is_empty() {
        for violation in violations {
            push_op_error(ctx, op, errors, violation.to_string());
        }
        return;
    }
    validate_attr_keys(
        ctx,
        op,
        &def.schema,
        DynSourceCallable::is_implemented(ctx, op),
        errors,
    );
    if let Some(validation) = DynLocalValidation::get(ctx, op) {
        validation.validate(ctx, errors);
    }
}

/// Validate only operation-local and type-local `tribute_control` invariants.
pub fn validate_local(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();
    validate_control_types(ctx, &mut errors);
    if let Some(body) = module.body(ctx) {
        walk_region_ops(ctx, body, &mut |op| {
            validate_local_operation(ctx, op, &mut errors);
        });
    }
    ValidationResult { errors }
}
