//! Whole-IR validation of returns, declaration uses, and resume tokens.

use super::*;

pub(super) fn compiler_intrinsic_map<'a>(
    declarations: &'a [CompilerIntrinsicDeclaration],
    errors: &mut Vec<ValidationError>,
) -> HashMap<SymbolPath, &'a CompilerIntrinsicDeclaration> {
    let mut map = HashMap::default();
    let mut previous = None;
    for declaration in declarations {
        let key = (declaration.symbol.clone(), declaration.identity.clone());
        if previous.is_some_and(|previous| previous > key) {
            push_type_error(
                errors,
                "compiler intrinsic declarations are not deterministically ordered",
            );
        }
        previous = Some(key);
        if map
            .insert(declaration.symbol.clone(), declaration)
            .is_some()
        {
            push_type_error(
                errors,
                format!(
                    "duplicate compiler intrinsic declaration for @{}",
                    declaration.symbol
                ),
            );
        }
    }
    map
}

pub(super) fn validate_return_contracts(
    ctx: &IrContext,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        if !Return::matches(ctx, op) {
            return;
        }
        let mut owner = parent_op(ctx, op);
        let callable = loop {
            let Some(current) = owner else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "return has no containing semantic callable",
                );
                return;
            };
            if let Some(callable) = DynSourceCallable::get(ctx, current) {
                break callable.callable_signature(ctx);
            }
            owner = parent_op(ctx, current);
        };
        let Some((expected, _)) = callable.and_then(|ty| func_sig_parts(ctx, ty)) else {
            return;
        };
        if let [value] = ctx.op_operands(op)
            && ctx.value_ty(*value) != expected
        {
            push_op_error(
                ctx,
                op,
                errors,
                "return operand type does not match containing semantic callable",
            );
        }
    });
}

pub(super) fn validate_declaration_uses(
    ctx: &IrContext,
    body: RegionRef,
    declarations: &HashMap<(&SymbolPath, StringRef), &OperationDeclaration>,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        let Some(declared) = DynOperationDeclarationUse::get(ctx, op) else {
            return;
        };
        let (Some(claim), Some(args)) = (
            declared.declaration_use(ctx),
            declared.declared_arguments(ctx),
        ) else {
            return;
        };
        let Some(declaration) = declarations.get(&(claim.ability, claim.op_name)) else {
            push_op_error(
                ctx,
                op,
                errors,
                format!(
                    "no resolved operation declaration for {}::{}",
                    claim.ability,
                    ctx.str(claim.op_name)
                ),
            );
            return;
        };
        if claim.kind != Some(declaration.kind) {
            push_op_error(
                ctx,
                op,
                errors,
                format!(
                    "{} does not match the resolved declaration",
                    claim.kind_attr
                ),
            );
        }
        check_types_equal(
            ctx,
            op,
            &value_types(ctx, args),
            &declaration.parameter_types,
            "operation argument",
            errors,
        );
        if declared.declared_result_type(ctx) != Some(declaration.result_type) {
            push_op_error(
                ctx,
                op,
                errors,
                "operation result type does not match the resolved declaration",
            );
        }
    });
}

fn op_is_within_region(ctx: &IrContext, op: OpRef, target: RegionRef) -> bool {
    let mut region = parent_region(ctx, op);
    while let Some(current) = region {
        if current == target {
            return true;
        }
        region = ctx
            .region(current)
            .parent_op
            .and_then(|owner| parent_region(ctx, owner));
    }
    false
}

fn nearest_enclosing_handler(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let mut owner = parent_op(ctx, op);
    while let Some(current) = owner {
        if Handler::matches(ctx, current) {
            return Some(current);
        }
        owner = parent_op(ctx, current);
    }
    None
}

pub(super) fn direct_call_reenters_enclosing_func(ctx: &IrContext, op: OpRef) -> bool {
    if !Call::matches(ctx, op) {
        return false;
    }
    let Some(callee) = ctx.op_attrs(op).get_symbol_ref("callee") else {
        return false;
    };
    let mut owner = parent_op(ctx, op);
    while let Some(current) = owner {
        if SourceFunc::is_set(ctx, current) {
            // The call is looked up in the module enclosing the function,
            // where the function is a symbol under its own name.
            return parent_op(ctx, current)
                .is_some_and(|module| trunk_ir::dialect::core::Module::matches(ctx, module))
                && ctx
                    .op_attrs(current)
                    .get_str("sym_name")
                    .is_some_and(|name| *callee == [name]);
        }
        owner = parent_op(ctx, current);
    }
    false
}

fn validate_affine_lambda_carrier(
    ctx: &IrContext,
    handler: OpRef,
    carrier: ValueRef,
    visited: &mut HashSet<ValueRef>,
    errors: &mut Vec<ValidationError>,
) {
    if !visited.insert(carrier) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier path contains a cycle",
        );
        return;
    }

    let mut captures = Vec::new();
    let mut terminals = Vec::new();
    for use_ in ctx.uses(carrier) {
        if nearest_enclosing_handler(ctx, use_.user) != Some(handler) {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume-token carrier crosses into a different handler at {}",
                    use_.user
                ),
            );
            continue;
        }
        if SourceLambda::is_set(ctx, use_.user) {
            captures.push(use_.user);
        } else if (SourceCallIndirect::is_set(ctx, use_.user) && use_.operand_index == 0)
            // A captured resumption may be passed to the enclosing source
            // function's recursive logical call (for example
            // `run_state(fn() { resume value }, value)`). Other direct calls
            // are rejected: their parameter ownership is not yet proven.
            || direct_call_reenters_enclosing_func(ctx, use_.user)
        {
            terminals.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume-token carrier escapes through {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }

    if has_nonexclusive_pair(ctx, &captures) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier branches into multiple lambda captures",
        );
    }
    if has_nonexclusive_pair(ctx, &terminals) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier has multiple static terminal uses",
        );
    }

    if !captures.is_empty() {
        for terminal in &terminals {
            if SourceCallIndirect::is_set(ctx, *terminal)
                && !captures.iter().any(|capture| {
                    ctx.op_region(*capture, 0)
                        .is_some_and(|region| op_is_within_region(ctx, *terminal, region))
                })
            {
                push_op_error(
                    ctx,
                    handler,
                    errors,
                    "resume-token carrier call is outside its capture path",
                );
            }
        }
    }
    for capture in captures {
        let Some(&next_carrier) = ctx.op_results(capture).first() else {
            push_op_error(
                ctx,
                handler,
                errors,
                "lambda capture carrying a resume token must produce one callable value",
            );
            return;
        };
        validate_affine_lambda_carrier(ctx, handler, next_carrier, visited, errors);
    }
}

fn validate_token_path(
    ctx: &IrContext,
    handler: OpRef,
    token_value: ValueRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut resumes = Vec::new();
    let mut captures = Vec::new();
    for use_ in ctx.uses(token_value) {
        if nearest_enclosing_handler(ctx, use_.user) != Some(handler) {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume token crosses into a different handler at {}",
                    use_.user
                ),
            );
            continue;
        }
        if Resume::matches(ctx, use_.user) && use_.operand_index == 0 {
            resumes.push(use_.user);
        } else if SourceLambda::is_set(ctx, use_.user) {
            captures.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume token has forbidden use by {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }
    if has_nonexclusive_pair(ctx, &resumes) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume token reaches more than one tribute_control.resume",
        );
    }
    for (left, right) in captures.iter().copied().tuple_combinations() {
        let left_region = ctx.op_region(left, 0);
        let right_region = ctx.op_region(right, 0);
        let comparable = left_region.is_some_and(|region| op_is_within_region(ctx, right, region))
            || right_region.is_some_and(|region| op_is_within_region(ctx, left, region));
        if !comparable && !ops_are_mutually_exclusive(ctx, left, right) {
            push_op_error(
                ctx,
                handler,
                errors,
                "resume token is copied into multiple capture paths",
            );
        }
    }
    for capture in &captures {
        let Some(region) = ctx.op_region(*capture, 0) else {
            continue;
        };
        if !resumes
            .iter()
            .any(|resume| op_is_within_region(ctx, *resume, region))
        {
            push_op_error(
                ctx,
                handler,
                errors,
                "resume token capture does not form a single path to resume",
            );
        }
    }
    let mut visited = HashSet::default();
    for capture in captures {
        if let Some(&closure_value) = ctx.op_results(capture).first() {
            validate_affine_lambda_carrier(ctx, handler, closure_value, &mut visited, errors);
        }
    }
}

/// A handler token may have one affine resume on each branch of an ordinary
/// structured conditional: exactly one branch executes at runtime.  Static
/// use counting alone would reject a source `case` whose exhaustive arms each
/// resume the same handler-owned token.
fn has_nonexclusive_pair(ctx: &IrContext, ops: &[OpRef]) -> bool {
    ops.iter()
        .copied()
        .tuple_combinations()
        .any(|(left, right)| !ops_are_mutually_exclusive(ctx, left, right))
}

pub(super) fn ops_are_mutually_exclusive(ctx: &IrContext, left: OpRef, right: OpRef) -> bool {
    let mut region = parent_region(ctx, left);
    while let Some(current) = region {
        let owner = ctx.region(current).parent_op;
        if let Some(owner) = owner
            && let Some(interface) = DynRegionBranch::get(ctx, owner)
            && let Ok(successors) = interface.successors(ctx, RegionBranchPoint::Parent)
        {
            let containing_region = |candidate: OpRef| {
                successors.as_slice().iter().find_map(|successor| {
                    let RegionSuccessor::Region(region) = successor else {
                        return None;
                    };
                    op_is_within_region(ctx, candidate, *region).then_some(*region)
                })
            };
            if let (Some(left_region), Some(right_region)) =
                (containing_region(left), containing_region(right))
                && left_region != right_region
            {
                return true;
            }
        }
        region = owner.and_then(|owner| parent_region(ctx, owner));
    }
    false
}

pub(super) fn validate_resume_ownership(
    ctx: &IrContext,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        if !Handler::matches(ctx, op) {
            return;
        }
        let Some(region) = ctx.op_region(op, 0) else {
            return;
        };
        let Some(&block) = ctx.region(region).blocks.first() else {
            return;
        };
        let Some(&token) = ctx.block_args(block).last() else {
            return;
        };
        if ResumeToken::matches(ctx, ctx.value_ty(token)) {
            validate_token_path(ctx, op, token, errors);
        }
    });
}

pub(super) fn validate_token_placements(
    ctx: &IrContext,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        for ty in ctx.op_result_types(op) {
            if ResumeToken::matches(ctx, *ty) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resume_token must not be produced as an operation result",
                );
            }
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for (index, arg) in ctx.block_args(block).iter().copied().enumerate() {
                    if !ResumeToken::matches(ctx, ctx.value_ty(arg)) {
                        continue;
                    }
                    let allowed =
                        Handler::matches(ctx, op) && index + 1 == ctx.block_args(block).len();
                    if !allowed {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "resume_token block arguments are allowed only as the final handler argument",
                        );
                    }
                }
            }
        }
    });
}
