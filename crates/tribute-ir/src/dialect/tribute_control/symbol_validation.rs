//! Whole-IR validation of symbol references and lambda captures.

use super::*;

/// The row arguments a reference passes to `target`, once their count
/// matches the binders `target` quantifies; `None` after reporting why not.
fn reference_row_args(
    ctx: &IrContext,
    op: OpRef,
    target: OpRef,
    errors: &mut Vec<ValidationError>,
) -> Option<Vec<Row>> {
    let args = row_args(ctx, op).ok()?;
    let quantified = quantified_count(&row_binders(ctx, target).ok()?);
    if args.len() != quantified {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "passes {} row arguments, but its target quantifies {quantified} row binders",
                args.len()
            ),
        );
        return None;
    }
    Some(args)
}

/// `target`'s signature as a reference with row arguments sees it. A type
/// that cannot be found is `None` and matches no value.
struct SubstitutedSignature {
    result: Option<TypeRef>,
    inputs: Vec<Option<TypeRef>>,
    row: Option<Row>,
}

fn substituted_parts(
    ctx: &IrContext,
    target_ty: TypeRef,
    args: &[Row],
) -> Option<SubstitutedSignature> {
    let signature = FuncSig::from_type_ref(ctx, target_ty)?;
    Some(SubstitutedSignature {
        result: find_substituted_rows(ctx, signature.result(ctx), args),
        inputs: signature
            .inputs(ctx)
            .iter()
            .map(|&input| find_substituted_rows(ctx, input, args))
            .collect(),
        row: signature.row(ctx).substitute(args),
    })
}

fn types_match(actual: &[TypeRef], expected: &[Option<TypeRef>]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| Some(*actual) == *expected)
}

/// Check that the result of `op`, a func_ref, is `target`'s signature with
/// the reference's row arguments substituted.
fn validate_func_ref_signature(
    ctx: &IrContext,
    op: OpRef,
    target: OpRef,
    errors: &mut Vec<ValidationError>,
) {
    let Some(target_ty) = ctx.op_attrs(target).get_type("type") else {
        return;
    };
    let Some(result_ty) = ctx.op_result_types(op).first().copied() else {
        return;
    };
    let Some(args) = reference_row_args(ctx, op, target, errors) else {
        return;
    };
    let same_signature = substituted_parts(ctx, target_ty, &args)
        .zip(FuncSig::from_type_ref(ctx, result_ty))
        .is_some_and(|(expected, signature)| {
            Some(signature.result(ctx)) == expected.result
                && types_match(signature.inputs(ctx), &expected.inputs)
                && expected
                    .row
                    .is_some_and(|row| signature.row(ctx).includes(&row))
        });
    if !same_signature {
        push_op_error(
            ctx,
            op,
            errors,
            "func_ref result source signature does not match its target",
        );
    }
}

/// Check that the targets of `table`, a `tribute_cps.func_table`, share one
/// source signature, each entered with its own convention or, for a `Cps`
/// target, a weaker one.
fn validate_func_table(
    ctx: &IrContext,
    table: OpRef,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    let mut signature = None;
    for (convention, target) in tribute_cps::table_entries(ctx, table).unwrap_or_default() {
        if target.as_simple().is_none() {
            push_op_error(
                ctx,
                table,
                errors,
                format!("function table target @{target} must be in the table's own module"),
            );
            continue;
        }
        let Some(instance) = funcs
            .definitions(ctx, table, &target)
            .first()
            .copied()
            .filter(|&instance| tribute_cps::Func::matches(ctx, instance))
        else {
            push_op_error(
                ctx,
                table,
                errors,
                format!("function table target @{target} is not a tribute_cps.func"),
            );
            continue;
        };
        let ty = ctx.op_attrs(instance).get_type("type");
        if *signature.get_or_insert(ty) != ty {
            push_op_error(
                ctx,
                table,
                errors,
                "function table targets must share one source signature",
            );
        }
        if let Some(own) = tribute_cps::declared_convention(ctx, instance)
            && convention != own
            && !(convention < own && own == CallingConvention::Cps)
        {
            push_op_error(
                ctx,
                table,
                errors,
                format!(
                    "function table entry `{}` is not the convention of its target @{target}",
                    convention.keyword()
                ),
            );
        }
    }
}

fn validate_symbol_uses(
    ctx: &IrContext,
    module: Module,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        validate_symbol_use(ctx, op, funcs, errors);
        ControlFlow::Continue(WalkAction::Advance)
    });
}

fn validate_symbol_use(
    ctx: &IrContext,
    op: OpRef,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    {
        if SourceFuncRef::is_set(ctx, op) {
            let Some(symbol) = ctx.op_attrs(op).get_symbol_ref("func_ref") else {
                return;
            };
            if tribute_cps::FuncRef::matches(ctx, op) {
                if symbol.as_simple().is_none() {
                    push_op_error(
                        ctx,
                        op,
                        errors,
                        format!("func_ref @{symbol} must name a table of its own module"),
                    );
                    return;
                }
                let Some(table) = funcs
                    .definitions(ctx, op, symbol)
                    .first()
                    .copied()
                    .filter(|&table| tribute_cps::FuncTable::matches(ctx, table))
                else {
                    push_op_error(
                        ctx,
                        op,
                        errors,
                        format!("unresolved function table @{symbol}"),
                    );
                    return;
                };
                for (_, target) in tribute_cps::table_entries(ctx, table).unwrap_or_default() {
                    if let Some(target) = funcs
                        .definitions(ctx, table, &target)
                        .first()
                        .copied()
                        .filter(|&target| SourceFunc::is_set(ctx, target))
                    {
                        validate_func_ref_signature(ctx, op, target, errors);
                    }
                }
                return;
            }
            let Some(target) = funcs
                .definitions(ctx, op, symbol)
                .first()
                .copied()
                .filter(|&target| SourceFunc::is_set(ctx, target))
            else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    format!("unresolved function symbol @{symbol}"),
                );
                return;
            };
            validate_func_ref_signature(ctx, op, target, errors);
        } else if tribute_cps::FuncTable::matches(ctx, op) {
            validate_func_table(ctx, op, funcs, errors);
        } else if Call::matches(ctx, op) {
            let Some(symbol) = ctx.op_attrs(op).get_symbol_ref("callee") else {
                return;
            };
            let Some(target) = funcs
                .definitions(ctx, op, symbol)
                .first()
                .copied()
                .filter(|&target| SourceFunc::is_set(ctx, target))
            else {
                push_op_error(ctx, op, errors, format!("unresolved callee @{symbol}"));
                return;
            };
            let Some(target_ty) = ctx.op_attrs(target).get_type("type") else {
                return;
            };
            let Some(args) = reference_row_args(ctx, op, target, errors) else {
                return;
            };
            let Some(SubstitutedSignature {
                result,
                inputs: params,
                ..
            }) = substituted_parts(ctx, target_ty, &args)
            else {
                return;
            };
            if !types_match(&value_types(ctx, ctx.op_operands(op)), &params) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "call argument types do not match the logical signature",
                );
            }
            if !types_match(ctx.op_result_types(op), &[result]) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "call result types do not match the logical signature",
                );
            }
        } else if TailCall::matches(ctx, op) {
            let Some(symbol) = ctx.op_attrs(op).get_symbol_ref("callee") else {
                return;
            };
            let Some(target) = funcs
                .definitions(ctx, op, symbol)
                .first()
                .copied()
                .filter(|&target| SourceFunc::is_set(ctx, target))
            else {
                push_op_error(ctx, op, errors, format!("unresolved callee @{symbol}"));
                return;
            };
            if ctx.op_attrs(target).get("abi").is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    format!("cannot tail call foreign function @{symbol}"),
                );
            }
            let Some(target_ty) = ctx.op_attrs(target).get_type("type") else {
                return;
            };
            let Some(args) = reference_row_args(ctx, op, target, errors) else {
                return;
            };
            let Some(SubstitutedSignature {
                result,
                inputs: params,
                ..
            }) = substituted_parts(ctx, target_ty, &args)
            else {
                return;
            };
            if !types_match(&value_types(ctx, ctx.op_operands(op)), &params) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "tail call argument types do not match the logical signature",
                );
            }
            let Ok((owner, owner_ty)) = tail_call_owner(ctx, op) else {
                return;
            };
            if func_sig_parts(ctx, owner_ty)
                .is_some_and(|(owner_result, _)| Some(owner_result) != result)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "callee result type does not match enclosing callable result",
                );
            }
            if let (Some(callee_cc), Some(owner_cc)) = (
                tribute_cps::declared_convention(ctx, target),
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
}

/// Resolve every function reference in the module tree from its nearest module.
pub(super) fn validate_module_symbols(
    ctx: &IrContext,
    module: Module,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    for ops in funcs.duplicates() {
        let symbol = qualified_name(ctx, ops[0]).unwrap_or_else(SymbolPath::root);
        for &op in &ops[1..] {
            push_op_error(
                ctx,
                op,
                errors,
                format!("duplicate function symbol @{symbol}"),
            );
        }
    }
    validate_symbol_uses(ctx, module, funcs, errors);
}

fn collect_external_references(
    ctx: &IrContext,
    region: RegionRef,
    defined: &HashSet<ValueRef>,
    external: &mut HashSet<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            for operand in ctx.op_operands(op).iter().copied() {
                if !defined.contains(&operand) {
                    external.insert(operand);
                }
            }
            if SourceFunc::is_set(ctx, op) {
                continue;
            }
            for nested in ctx.op_regions(op) {
                collect_external_references(ctx, nested, defined, external);
            }
        }
    }
}

pub(super) fn validate_lambda_captures(
    ctx: &IrContext,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        if !SourceLambda::is_set(ctx, op) {
            return;
        }
        let captures = ctx.op_operands(op);
        let capture_set: HashSet<ValueRef> = captures.iter().copied().collect();
        if capture_set.len() != captures.len() {
            push_op_error(ctx, op, errors, "capture list contains duplicate values");
        }
        // The bodies share the captures, which every body's references
        // outside it make up together.
        let mut external = HashSet::default();
        for region in ctx.op_regions(op) {
            let mut defined = HashSet::default();
            collect_region_values(ctx, region, &mut defined);
            collect_external_references(ctx, region, &defined, &mut external);
        }
        for missing in external.difference(&capture_set) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("capture list is missing external value {missing}"),
            );
        }
        for excess in capture_set.difference(&external) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("capture list contains unused external value {excess}"),
            );
        }
    });
}
