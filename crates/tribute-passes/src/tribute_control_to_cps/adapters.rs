//! Block, closure, and tail-transfer builders that `tribute_control_to_cps`
//! and `lower_continuation_frames` share.

use super::*;

pub(crate) fn make_block(ctx: &mut IrContext, location: Location, types: &[TypeRef]) -> BlockRef {
    ctx.create_block(BlockData {
        location,
        args: types
            .iter()
            .copied()
            .map(|ty| BlockArgData {
                ty,
                attrs: AttributeMap::new(),
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    })
}

pub(crate) fn single_block_region(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
) -> RegionRef {
    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// The name of the `index`th helper function of kind `prefix`.
pub(crate) fn helper_symbol(prefix: &str, index: u32) -> Symbol {
    Symbol::new(&format!("__tribute_{prefix}_{index}"))
}

/// Build the closure of `region`, capturing the values it uses from outside
/// in their order of first use.
pub(crate) fn closure_over(
    ctx: &mut IrContext,
    location: Location,
    region: RegionRef,
    closure_type: TypeRef,
    convention: CallingConvention,
) -> closure::Lambda {
    let lambda = closure::Lambda::operands(ordered_external_values(ctx, region))
        .results(closure_type)
        .regions(region)
        .build(ctx, location);
    set_calling_convention(ctx, lambda.op_ref(), convention);
    lambda
}

/// Emit a final CPS tail transfer from the callee's exact typed closure
/// contract. This must not reconstruct a signature from physical operands:
/// closure extraction interposes an environment later and preserves this
/// contract on the resulting indirect transfer.
pub(crate) fn emit_cps_tail_call_indirect(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    callee: ValueRef,
    args: impl IntoIterator<Item = ValueRef>,
) -> Result<OpRef, TributeControlToCpsError> {
    let args = args.into_iter().collect::<Vec<_>>();
    let closure_type = ctx.value_ty(callee);
    let signature = cps_closure_function_type(ctx, closure_type).ok_or_else(|| {
        TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail callee has no exact provenance-bearing closure contract",
        )
    })?;
    let callable = func::FuncSig::from_type_ref(ctx, signature).ok_or_else(|| {
        TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail callee contract is not func.func_sig",
        )
    })?;
    let never = core::never(ctx).as_type_ref();
    if callable.results(ctx) != [never]
        || callable.inputs(ctx).len() != args.len()
        || callable
            .inputs(ctx)
            .iter()
            .zip(&args)
            .any(|(expected, actual)| *expected != ctx.value_ty(*actual))
    {
        return Err(TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail operands differ from the exact closure contract",
        ));
    }
    let tail = func::TailCallIndirect::operands(callee, args)
        .signature(signature)
        .build(ctx, location);
    set_calling_convention(ctx, tail.op_ref(), CallingConvention::Cps);
    ctx.push_op(block, tail.op_ref());
    Ok(tail.op_ref())
}

fn collect_defined_values(ctx: &IrContext, region: RegionRef, defined: &mut HashSet<ValueRef>) {
    for block in ctx.region(region).blocks.iter().copied() {
        defined.extend(ctx.block_args(block).iter().copied());
        for op in ctx.block(block).ops.iter().copied() {
            defined.extend(ctx.op_results(op).iter().copied());
            for nested in ctx.op_regions(op) {
                collect_defined_values(ctx, nested, defined);
            }
        }
    }
}

fn collect_external_in_order(
    ctx: &IrContext,
    region: RegionRef,
    defined: &HashSet<ValueRef>,
    seen: &mut HashSet<ValueRef>,
    external: &mut Vec<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            for operand in ctx.op_operands(op).iter().copied() {
                if !defined.contains(&operand) && seen.insert(operand) {
                    external.push(operand);
                }
            }
            for nested in ctx.op_regions(op) {
                collect_external_in_order(ctx, nested, defined, seen, external);
            }
        }
    }
}

fn ordered_external_values(ctx: &IrContext, region: RegionRef) -> Vec<ValueRef> {
    let mut defined = HashSet::default();
    collect_defined_values(ctx, region, &mut defined);
    let mut seen = HashSet::default();
    let mut external = Vec::new();
    collect_external_in_order(ctx, region, &defined, &mut seen, &mut external);
    external
}
