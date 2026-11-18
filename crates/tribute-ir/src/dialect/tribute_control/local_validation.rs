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
    }
}

pub(crate) fn validate_lambda(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let ty = ctx.op_result_types(op)[0];
    for body in ctx.op_regions(op).collect::<Vec<_>>() {
        validate_callable_body(ctx, op, ty, body, errors);
    }
}

pub(super) fn validate_return(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
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

/// The callable whose body contains `op` and its source signature, or an
/// error when `op` is not in its tail position. A `handle` in tail
/// position makes its completion and `op` arms tail positions; its body and
/// `fn` arms are not.
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
        let terminator = if let Ok(handle) = Handle::from_op(ctx, owner) {
            if !is_tail_handle_region(ctx, handle, exit) {
                return Err("must not be inside a tribute_control.handle body or fn arm");
            }
            // The handle's frame is the one its result flows into, so its
            // result is that terminator's operand itself.
            result_terminator(ctx, owner, false)
        } else {
            // A structured arm may end with a tail call only when the
            // structured operation selects one region and has one result,
            // whose value flows straight into its own block's terminator.
            if !DynRegionBranch::selects_one_region(ctx, owner) || ctx.op_results(owner).len() != 1
            {
                return Err("must be in tail position of its callable");
            }
            result_terminator(ctx, owner, true)
        };
        let Some(terminator) = terminator else {
            return Err("must be in tail position of its callable");
        };
        if Return::matches(ctx, terminator)
            || Yield::matches(ctx, terminator)
            || trunk_ir::dialect::scf::Yield::matches(ctx, terminator)
        {
            exit = terminator;
            continue;
        }
        return Err("must be in tail position of its callable");
    }
}

/// Whether `op`'s one result flows straight into a terminator in tail
/// position of its callable, the position a source tail call may take.
pub fn in_tail_position(ctx: &IrContext, op: OpRef) -> bool {
    result_terminator(ctx, op, false).is_some_and(|terminator| {
        (Return::matches(ctx, terminator)
            || Yield::matches(ctx, terminator)
            || trunk_ir::dialect::scf::Yield::matches(ctx, terminator))
            && tail_call_owner(ctx, terminator).is_ok()
    })
}

/// The handle whose body result `op`'s one result is: the operand of the
/// body's `yield` right after `op`, or, through the `scf.yield` of an arm,
/// the result of a one-result operation that selects one region and is
/// such a result itself (`design/cps-effects.md`, "Frame을 그대로 넘기는
/// 자리").
pub fn body_result_handle(ctx: &IrContext, op: OpRef) -> Option<Handle> {
    let mut source = op;
    loop {
        let terminator = result_terminator(ctx, source, false)?;
        let owner = parent_op(ctx, terminator)?;
        if Yield::matches(ctx, terminator) {
            let handle = Handle::from_op(ctx, owner).ok()?;
            return (parent_region(ctx, terminator) == Some(handle.body(ctx))).then_some(handle);
        }
        if !trunk_ir::dialect::scf::Yield::matches(ctx, terminator)
            || !DynRegionBranch::selects_one_region(ctx, owner)
            || ctx.op_results(owner).len() != 1
        {
            return None;
        }
        source = owner;
    }
}

/// The terminator of the block of `op` when it takes `op`'s one result as
/// its only operand and nothing comes between them, or, if `casts` allows
/// them, only conversion casts that each cast the value before it.
fn result_terminator(ctx: &IrContext, op: OpRef, casts: bool) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    let ops = &ctx.block(block).ops;
    let position = ops.iter().position(|&candidate| candidate == op)?;
    let following = &ops[position + 1..];
    let (&terminator, between) = following.split_last()?;
    if !casts && !between.is_empty() {
        return None;
    }
    let &[mut value] = ctx.op_results(op) else {
        return None;
    };
    for &cast in between {
        if !core::UnrealizedConversionCast::matches(ctx, cast) || ctx.op_operands(cast) != [value] {
            return None;
        }
        value = ctx.op_results(cast)[0];
    }
    (ctx.op_operands(terminator) == [value]).then_some(terminator)
}

/// Whether `inner`, an operation in a region of `handle`, is in its
/// completion or an `op` arm, whose results are the handle's.
fn is_tail_handle_region(ctx: &IrContext, handle: Handle, inner: OpRef) -> bool {
    let Some(region) = parent_region(ctx, inner) else {
        return false;
    };
    if handle.completion(ctx) == region {
        return true;
    }
    handle.arm_bindings(ctx).any(|(binding, arm)| {
        arm == region && binding.is_some_and(|binding| binding.kind == OperationKind::Op)
    })
}

/// Whether `block`, the completion or an arm of a handle, ends with a
/// source tail call in place of its `yield`.
pub(super) fn ends_with_tail_call(ctx: &IrContext, block: BlockRef) -> bool {
    ctx.block(block).ops.last().is_some_and(|&last| {
        TailCall::matches(ctx, last) || SourceTailCallIndirect::is_set(ctx, last)
    })
}

/// That the tail transfer `op`, whose callee has the convention `callee`,
/// suits the flow that holds it: in a `Cps` flow a transfer to a callee
/// with a body enters it as `Cps`, and elsewhere the callee is not stronger
/// than the flow (`design/tribute-dialects.md`, "Convention과 class 특수화").
pub(super) fn check_transfer_convention(
    ctx: &IrContext,
    op: OpRef,
    callee: CallingConvention,
    has_body: bool,
    errors: &mut Vec<ValidationError>,
) {
    let Some(flow) = tribute_cps::flow_convention(ctx, op) else {
        return;
    };
    if flow == CallingConvention::Cps && has_body && callee != CallingConvention::Cps {
        push_op_error(
            ctx,
            op,
            errors,
            "a tail transfer in a Cps flow must enter its callee as Cps",
        );
    } else if callee > flow {
        push_op_error(
            ctx,
            op,
            errors,
            "callee convention must not be stronger than the flow that holds the transfer",
        );
    }
}

pub(crate) fn validate_tail_call(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let (_, owner_ty) = match tail_call_owner(ctx, op) {
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
        // In a `Cps` flow it enters the value's `Cps` entry, whatever class
        // the callee type reads.
        if let Some(convention) = tribute_cps::declared_convention(ctx, op) {
            check_transfer_convention(ctx, op, convention, true, errors);
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
    // A source tail call that ends the completion leaves the callable; its
    // own validation checks the placement.
    let completion_yield = if ends_with_tail_call(ctx, completion_block) {
        None
    } else {
        terminator(ctx, op, completion_block, Yield::DEF, errors)
    };
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
    if binding.kind == OperationKind::Op && ends_with_tail_call(ctx, block) {
        return;
    }
    let yield_op = terminator(ctx, op, block, Yield::DEF, errors);
    let yielded = yield_op
        .and_then(|yield_op| ctx.op_operands(yield_op).first().copied())
        .map(|value| ctx.value_ty(value));
    let operation_result = binding.operation_result_type;
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

pub(super) fn validate_resume(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let Some(resumption) = DynResumptionRegions::enclosing(ctx, op) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must be inside a resumptive op arm of a tribute_control.handle, not behind a \
             function, a fn arm, or an op -> core.never arm",
        );
        return;
    };
    let (Some(&value), Some(&result)) =
        (ctx.op_operands(op).first(), ctx.op_result_types(op).first())
    else {
        return;
    };
    if ctx.value_ty(value) != resumption.input {
        push_op_error(
            ctx,
            op,
            errors,
            "operand type does not match the operation result type of the enclosing arm",
        );
    }
    if result != resumption.result {
        push_op_error(
            ctx,
            op,
            errors,
            "result type does not match the body result type of the enclosing handle",
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
