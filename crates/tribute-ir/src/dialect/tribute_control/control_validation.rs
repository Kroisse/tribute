//! Whole-IR validation of returns, declaration uses, and resumptions.

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
        for claim in declared.declaration_uses(ctx) {
            validate_declaration_use(ctx, op, &claim, declarations, errors);
        }
    });
}

fn validate_declaration_use(
    ctx: &IrContext,
    op: OpRef,
    claim: &DeclarationUse<'_>,
    declarations: &HashMap<(&SymbolPath, StringRef), &OperationDeclaration>,
    errors: &mut Vec<ValidationError>,
) {
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
        &value_types(ctx, claim.arguments),
        &declaration.parameter_types,
        "operation argument",
        errors,
    );
    if claim.result_type != Some(declaration.result_type) {
        push_op_error(
            ctx,
            op,
            errors,
            "operation result type does not match the resolved declaration",
        );
    }
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
    handle: OpRef,
    arm: RegionRef,
    carrier: ValueRef,
    visited: &mut HashSet<ValueRef>,
    errors: &mut Vec<ValidationError>,
) {
    // A carrier reached by two paths was validated by the first.
    if !visited.insert(carrier) {
        return;
    }

    let mut captures = Vec::new();
    let mut terminals = Vec::new();
    for use_ in ctx.uses(carrier) {
        if DynResumptionRegions::enclosing(ctx, use_.user).map(|r| r.arm) != Some(arm) {
            push_op_error(
                ctx,
                handle,
                errors,
                format!(
                    "resumption carrier crosses into a different handler arm at {}",
                    use_.user
                ),
            );
            continue;
        }
        if SourceLambda::is_set(ctx, use_.user) {
            captures.push(use_.user);
        } else if (SourceCallIndirect::is_set(ctx, use_.user) && use_.operand_index == 0)
            // A carrier may be passed to the enclosing source
            // function's recursive logical call (for example
            // `run_state(fn() { resume value }, value)`). Other direct calls
            // are rejected: their parameter ownership is not yet proven.
            || direct_call_reenters_enclosing_func(ctx, use_.user)
        {
            terminals.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handle,
                errors,
                format!(
                    "resumption carrier escapes through {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }

    if has_nonexclusive_pair(ctx, &captures) {
        push_op_error(
            ctx,
            handle,
            errors,
            "resumption carrier branches into multiple lambda captures",
        );
    }
    if has_nonexclusive_pair(ctx, &terminals) {
        push_op_error(
            ctx,
            handle,
            errors,
            "resumption carrier has multiple static terminal uses",
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
                    handle,
                    errors,
                    "resumption carrier call is outside its capture path",
                );
            }
        }
    }
    for capture in captures {
        let Some(&next_carrier) = ctx.op_results(capture).first() else {
            push_op_error(
                ctx,
                handle,
                errors,
                "lambda carrying a resumption must produce one callable value",
            );
            return;
        };
        validate_affine_lambda_carrier(ctx, handle, arm, next_carrier, visited, errors);
    }
}

/// The affine use of the resumption `arm` of `handle` owns: every `resume`
/// that consumes it is on a different branch, and each lambda that carries it
/// stays inside the arm.
fn validate_arm_resumption(
    ctx: &IrContext,
    handle: OpRef,
    arm: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut resumes = Vec::new();
    let mut carriers = Vec::new();
    walk_region_ops(ctx, arm, &mut |op| {
        let owns = |resumption: Option<Resumption>| resumption.is_some_and(|r| r.arm == arm);
        if Resume::matches(ctx, op) {
            if owns(DynResumptionRegions::enclosing(ctx, op)) {
                resumes.push(op);
            }
        } else if owns(DynResumptionRegions::carried_by(ctx, op)) {
            carriers.push(op);
        }
    });
    if has_nonexclusive_pair(ctx, &resumes) {
        push_op_error(
            ctx,
            handle,
            errors,
            "the resumption of an arm reaches more than one tribute_control.resume",
        );
    }
    let mut visited = HashSet::default();
    for carrier in carriers {
        if let Some(&closure_value) = ctx.op_results(carrier).first() {
            validate_affine_lambda_carrier(ctx, handle, arm, closure_value, &mut visited, errors);
        }
    }
}

/// An arm's resumption may have one affine resume on each branch of an
/// ordinary structured conditional: exactly one branch executes at runtime.
/// Static use counting alone would reject a source `case` whose exhaustive
/// arms each resume.
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
        let Ok(handle) = Handle::from_op(ctx, op) else {
            return;
        };
        for arm in handle.arms(ctx) {
            if matches!(
                handle.resumption_scope(ctx, arm),
                ResumptionScope::Provides { .. }
            ) {
                validate_arm_resumption(ctx, op, arm, errors);
            }
        }
    });
}
