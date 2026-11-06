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
/// error when `op` is not in its tail position or a handle lies between
/// them.
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
        if Handle::matches(ctx, owner) {
            return Err("must not be inside a tribute_control.handle body or arm");
        }
        // A structured arm may end with a tail call only when the structured
        // operation is a one-result `scf.if` or `scf.index_switch` whose
        // value flows straight into its own block's terminator.
        let structured = trunk_ir::dialect::scf::If::matches(ctx, owner)
            || trunk_ir::dialect::scf::IndexSwitch::matches(ctx, owner);
        if !structured || ctx.op_results(owner).len() != 1 {
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
    let mut regions = ctx.op_regions(op);
    let (Some(body_region), Some(completion_region)) = (regions.next(), regions.next()) else {
        return;
    };
    let Some(body_block) = single_block(ctx, op, body_region, "body", errors) else {
        return;
    };
    let Some(completion_block) = single_block(ctx, op, completion_region, "completion", errors)
    else {
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
    let handle = Handle::from_op(ctx, op).expect("the local validator runs on a handle");
    for (binding, arm) in handle.arm_bindings(ctx) {
        if let Some(binding) = binding {
            validate_arm(ctx, op, &binding, arm, handle_result, errors);
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

/// The checks of one arm of the handle `op`, whose result is
/// `handle_result`.
fn validate_arm(
    ctx: &IrContext,
    op: OpRef,
    binding: &HandlerBinding,
    arm: RegionRef,
    handle_result: Option<TypeRef>,
    errors: &mut Vec<ValidationError>,
) {
    let name = format!("{}::{}", binding.ability, ctx.str(binding.op_name));
    let Some(block) = single_block(ctx, op, arm, "arm", errors) else {
        return;
    };
    let yield_op = terminator(ctx, op, block, Yield::DEF, errors);
    let yielded = yield_op
        .and_then(|yield_op| ctx.op_operands(yield_op).first().copied())
        .map(|value| ctx.value_ty(value));
    let operation_result = binding.operation_result_type;
    let last_token = ctx
        .block_args(block)
        .last()
        .copied()
        .and_then(|value| resume_token_parts(ctx, ctx.value_ty(value)));
    if binding.is_resumptive(ctx) {
        let Some((token_input, token_answer)) = last_token else {
            push_op_error(
                ctx,
                op,
                errors,
                format!("resumptive op arm {name} requires a final resume_token block argument"),
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
                format!(
                    "resume token input/answer types of arm {name} do not match operation and handle results"
                ),
            );
        }
    } else {
        if last_token.is_some() {
            push_op_error(
                ctx,
                op,
                errors,
                format!("fn and op -> core.never arm {name} must not receive a resume token"),
            );
        }
        if binding.kind == OperationKind::Op && region_contains_resume(ctx, arm) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("op -> core.never arm {name} must not contain tribute_control.resume"),
            );
        }
    }
    let expected = match binding.kind {
        OperationKind::Fn => Some(operation_result),
        OperationKind::Op => handle_result,
    };
    if let (Some(yielded), Some(expected)) = (yielded, expected)
        && yielded != expected
    {
        push_op_error(
            ctx,
            op,
            errors,
            format!("arm {name} yield type does not match its kind-specific result"),
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
    let valid_owner = parent_op(ctx, op).is_some_and(|owner| Handle::matches(ctx, owner));
    if !valid_owner {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a handle body, completion, or arm",
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
