//! Lower structured control flow (scf) to CFG-based control flow (cf) for arena IR.
//!
//! Converts `scf.if`, `scf.loop`, and `scf.switch` operations into explicit
//! `cf.br` and `cf.cond_br` branch operations with basic block splitting.
//!
//! Unlike the Salsa-based version, this pass uses RAUW (`replace_all_uses`)
//! to map scf op results to merge block arguments, eliminating the need for
//! manual value remapping.
//!
//! ## Transformations
//!
//! ### scf.if
//! ```text
//! ^bb0:
//!   %0 = op_before(...)
//!   %1 = scf.if(%cond) -> T { scf.yield(%a) } { scf.yield(%b) }
//!   %2 = op_after(%1)
//! ```
//! becomes:
//! ```text
//! ^bb0:
//!   %0 = op_before(...)
//!   cf.cond_br(%cond) -> ^then, ^else
//! ^then:
//!   cf.br(%a) -> ^merge
//! ^else:
//!   cf.br(%b) -> ^merge
//! ^merge(%1: T):
//!   %2 = op_after(%1)
//! ```

use smallvec::SmallVec;

use crate::context::{BlockArgData, BlockData, IrContext};
use crate::dialect::{arith, cf, func, scf};
use crate::op_interface::{RegionBranchOps, RegionBranchPoint, RegionSuccessor};
use crate::ops::DialectOp;
use crate::pass::{Pass, pass_fn};
use crate::refs::{BlockRef, OpRef, RegionRef, ValueRef};
use crate::rewrite::Module;
use crate::rewrite::helpers::{inline_region_blocks, split_block};
use crate::symbol::Symbol;
use crate::types::{Attribute, Location};

/// Lower all `scf` operations in a module to `cf` operations.
pub fn lower_scf_to_cf(ctx: &mut IrContext, module: Module) {
    let body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };
    transform_region(ctx, body);
}

/// Lower all `scf` operations in one function to `cf` operations.
pub fn lower_scf_to_cf_func(ctx: &mut IrContext, func: func::Func) {
    let Some(body) = func.body_if_present(ctx) else {
        return;
    };
    transform_region(ctx, body);
}

/// Build a function-anchored SCF-to-CF lowering pass.
pub fn scf_to_cf_pass() -> impl Pass<Target = func::Func> {
    pass_fn("scf-to-cf-func", |ctx, target| {
        lower_scf_to_cf_func(ctx, target);
        Ok(())
    })
}

/// Transform all blocks in a region, lowering scf ops to cf.
fn transform_region(ctx: &mut IrContext, region: RegionRef) {
    // We iterate blocks by index because new blocks may be inserted.
    // Process each block: if it contains an scf op, split and expand.
    let mut i = 0;
    loop {
        let blocks = ctx.region(region).blocks.to_vec();
        if i >= blocks.len() {
            break;
        }
        let block = blocks[i];
        transform_block(ctx, block);
        i += 1;
    }
}

/// Transform a single block, looking for the first scf op to lower.
///
/// If found, the block is split and expanded. The merge block (containing
/// operations after the scf op) will be processed in a subsequent iteration
/// of the region loop.
fn transform_block(ctx: &mut IrContext, block: BlockRef) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    // First, recursively transform nested regions in non-scf ops
    for &op in &ops {
        if is_scf_control_flow(ctx, op) {
            continue;
        }
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            transform_region(ctx, region);
        }
    }

    // Find the first scf control-flow op
    let scf_idx = ops.iter().position(|&op| is_scf_control_flow(ctx, op));
    let Some(scf_idx) = scf_idx else {
        return;
    };

    let scf_op = ops[scf_idx];
    let loc = ctx.op(scf_op).location;

    if scf::If::matches(ctx, scf_op) {
        lower_scf_if(ctx, block, scf_op, loc);
    } else if scf::Loop::matches(ctx, scf_op) {
        lower_scf_loop(ctx, block, scf_op, loc);
    } else if scf::Switch::matches(ctx, scf_op) {
        lower_scf_switch(ctx, block, scf_op, loc);
    }
}

/// Check if an op is an scf control-flow op (if/loop/switch).
fn is_scf_control_flow(ctx: &IrContext, op: OpRef) -> bool {
    let d = ctx.op(op).dialect;
    if d != Symbol::new("scf") {
        return false;
    }
    let n = ctx.op(op).name;
    n == Symbol::new("if") || n == Symbol::new("loop") || n == Symbol::new("switch")
}

/// Whether a one-block region has only terminal structured control or a
/// proper-tail transfer, so lowering it cannot need a continuation block.
fn is_terminal_region(ctx: &IrContext, region: RegionRef) -> bool {
    let [branch] = ctx.region(region).blocks.as_slice() else {
        return false;
    };
    let Some(&terminator) = ctx.block(*branch).ops.last() else {
        return false;
    };
    if scf::If::matches(ctx, terminator) {
        is_terminal_never_if(ctx, *branch, terminator)
    } else if scf::Switch::matches(ctx, terminator) {
        ctx.op_results(terminator).is_empty()
            && has_only_terminal_region_successors(ctx, terminator)
    } else {
        !is_scf_control_flow(ctx, terminator)
            && crate::validation::is_proper_tail_terminator(ctx, terminator)
    }
}

/// Whether every semantic entry successor of a structured operation is a
/// terminal region. Missing, incomplete, or parent-returning mappings are
/// nonterminal by construction.
fn has_only_terminal_region_successors(ctx: &IrContext, op: OpRef) -> bool {
    let Some(interface) = RegionBranchOps::get(ctx, op) else {
        return false;
    };
    let Ok(successors) = interface.successors(ctx, op, RegionBranchPoint::Parent) else {
        return false;
    };
    !successors.as_slice().is_empty()
        && successors.as_slice().iter().all(|successor| {
            matches!(successor, RegionSuccessor::Region(region) if is_terminal_region(ctx, *region))
        })
}

/// Whether this is a terminal `scf.if : core.never` whose branches each
/// already transfer control. Such an if has no continuation to merge into.
fn is_terminal_never_if(ctx: &IrContext, block: BlockRef, scf_op: OpRef) -> bool {
    let [result] = ctx.op_results(scf_op) else {
        return false;
    };
    let result_ty = ctx.types.get(ctx.value_ty(*result));
    if result_ty.dialect != Symbol::new("core")
        || result_ty.name != Symbol::new("never")
        || ctx.has_uses(*result)
        || ctx.block(block).ops.last() != Some(&scf_op)
    {
        return false;
    }

    has_only_terminal_region_successors(ctx, scf_op)
}

/// Lower a terminal `scf.if : core.never` without creating a merge block.
fn lower_terminal_never_if(
    ctx: &mut IrContext,
    block: BlockRef,
    scf_op: OpRef,
    loc: Location,
    cond: ValueRef,
    then_region: RegionRef,
    else_region: RegionRef,
) {
    let parent_region = ctx.block(block).parent_region.unwrap();
    let blocks = ctx.region(parent_region).blocks.to_vec();
    let insert_before = blocks
        .iter()
        .position(|&candidate| candidate == block)
        .and_then(|index| blocks.get(index + 1).copied());

    ctx.detach_op(scf_op);
    let then_blocks = inline_region_blocks(ctx, then_region, parent_region, insert_before);
    let else_blocks = inline_region_blocks(ctx, else_region, parent_region, insert_before);
    ctx.remove_op(scf_op);

    let cond_br = cf::cond_br(ctx, loc, cond, then_blocks[0], else_blocks[0]);
    ctx.push_op(block, cond_br.op_ref());

    for &branch in then_blocks.iter().chain(&else_blocks) {
        transform_block(ctx, branch);
    }
}

/// Lower `scf.if` to cf.cond_br + then/else/merge blocks.
fn lower_scf_if(ctx: &mut IrContext, block: BlockRef, scf_op: OpRef, loc: Location) {
    let if_op = scf::If::from_op(ctx, scf_op).unwrap();
    let cond = if_op.cond(ctx);
    let then_region = if_op.then_region(ctx);
    let else_region = if_op.else_region(ctx);

    if is_terminal_never_if(ctx, block, scf_op) {
        lower_terminal_never_if(ctx, block, scf_op, loc, cond, then_region, else_region);
        return;
    }

    // Only results with users need a CFG merge value. This intentionally
    // applies to every result type: an unused result does not need a block
    // argument or branch operands merely because the structured op has one.
    let result = match ctx.op_results(scf_op) {
        [] => None,
        [result] if ctx.has_uses(*result) => Some((*result, ctx.value_ty(*result))),
        [_] => None,
        _ => return,
    };

    // Split block at the scf op: ops after scf_op go to merge block
    let merge_block = split_block(ctx, block, scf_op);

    // Add a merge block argument and RAUW only a used result.
    if let Some((if_result, ty)) = result {
        let merge_arg = ctx.add_block_arg(
            merge_block,
            BlockArgData {
                ty,
                attrs: Default::default(),
            },
        );
        // RAUW: replace all uses of scf.if result with merge block arg
        ctx.replace_all_uses(if_result, merge_arg);
    }

    // Remove the scf op (split_block moved it to merge_block)
    ctx.detach_op(scf_op);

    // Inline then/else regions into the parent region
    let parent_region = ctx.block(block).parent_region.unwrap();
    let then_blocks = inline_region_blocks(ctx, then_region, parent_region, Some(merge_block));
    let else_blocks = inline_region_blocks(ctx, else_region, parent_region, Some(merge_block));
    ctx.remove_op(scf_op);

    let then_entry = then_blocks[0];
    let else_entry = else_blocks[0];

    // Replace scf.yield in then/else blocks with cf.br to merge
    replace_yield_with_br(ctx, &then_blocks, merge_block, loc);
    replace_yield_with_br(ctx, &else_blocks, merge_block, loc);

    // Add cf.cond_br to the original block
    let cond_br = cf::cond_br(ctx, loc, cond, then_entry, else_entry);
    ctx.push_op(block, cond_br.op_ref());

    // Recursively transform then/else blocks (they may contain nested scf ops)
    for &b in &then_blocks {
        transform_block(ctx, b);
    }
    for &b in &else_blocks {
        transform_block(ctx, b);
    }
}

/// Lower `scf.loop` to cf header + exit blocks.
fn lower_scf_loop(ctx: &mut IrContext, block: BlockRef, scf_op: OpRef, loc: Location) {
    let loop_op = scf::Loop::from_op(ctx, scf_op).unwrap();
    let init_values: Vec<_> = loop_op.init(ctx).to_vec();
    let body_region = loop_op.body(ctx);

    // Determine result type
    let result_ty = match ctx.op_results(scf_op) {
        [] => None,
        [result] => Some(ctx.value_ty(*result)),
        _ => return,
    };

    // Split block at scf op: ops after go to exit block
    let exit_block = split_block(ctx, block, scf_op);

    // Remove the scf op (split_block moved it to exit_block)
    ctx.detach_op(scf_op);

    // Add exit block argument for the result (if any)
    if let Some(ty) = result_ty {
        let exit_arg = ctx.add_block_arg(
            exit_block,
            BlockArgData {
                ty,
                attrs: Default::default(),
            },
        );
        let loop_result = ctx.op_results(scf_op)[0];
        ctx.replace_all_uses(loop_result, exit_arg);
    }

    // Inline body region blocks into parent region (before exit block)
    let parent_region = ctx.block(block).parent_region.unwrap();
    let body_blocks = inline_region_blocks(ctx, body_region, parent_region, Some(exit_block));
    ctx.remove_op(scf_op);

    // The first body block is the header (loop entry point)
    let header_block = body_blocks[0];

    // Replace scf.continue with cf.br to header, scf.break with cf.br to exit
    replace_continue_break(ctx, &body_blocks, header_block, exit_block, loc);

    // Add cf.br from entry block to header with init values
    let br_to_header = cf::br(ctx, loc, init_values, header_block);
    ctx.push_op(block, br_to_header.op_ref());

    // Recursively transform body blocks
    for &b in &body_blocks {
        transform_block(ctx, b);
    }
}

struct SwitchArms {
    discriminant: ValueRef,
    cases: Vec<(Attribute, RegionRef)>,
    default_region: Option<RegionRef>,
}

/// Immutable switch dispatch inputs after all arm blocks have been inlined.
struct SwitchDispatch<'a> {
    discriminant: ValueRef,
    cases: &'a [(Attribute, RegionRef)],
    case_entries: &'a [BlockRef],
    default_entry: BlockRef,
    parent_region: RegionRef,
    insert_before: Option<BlockRef>,
}

/// Collect a well-formed resultless switch's discriminant and arms.
///
/// The lowering intentionally leaves malformed switches untouched so the
/// operation and interface verifiers can report the source shape.
fn switch_arms(ctx: &IrContext, scf_op: OpRef) -> Option<SwitchArms> {
    if !ctx.op_results(scf_op).is_empty() {
        return None;
    }

    let [discriminant] = ctx.op_operands(scf_op) else {
        return None;
    };
    let [body_region] = ctx.op(scf_op).regions.as_slice() else {
        return None;
    };
    let [body_block] = ctx.region(*body_region).blocks.as_slice() else {
        return None;
    };

    let mut cases = Vec::new();
    let mut default_region = None;
    for &arm in &ctx.block(*body_block).ops {
        let arm_data = ctx.op(arm);
        let [arm_region] = arm_data.regions.as_slice() else {
            return None;
        };
        if ctx.region(*arm_region).blocks.is_empty() {
            return None;
        }
        if scf::Case::matches(ctx, arm) {
            cases.push((arm_data.attributes.get("value")?.clone(), *arm_region));
        } else if scf::Default::matches(ctx, arm) {
            if default_region.replace(*arm_region).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(SwitchArms {
        discriminant: *discriminant,
        cases,
        default_region,
    })
}

/// Whether a resultless switch is the final operation in its block and every
/// selectable arm transfers control, leaving no continuation to merge into.
fn is_terminal_resultless_switch(
    ctx: &IrContext,
    block: BlockRef,
    scf_op: OpRef,
    cases: &[(Attribute, RegionRef)],
    default_region: Option<RegionRef>,
) -> bool {
    ctx.block(block).ops.last() == Some(&scf_op)
        && default_region.is_some_and(|region| is_terminal_region(ctx, region))
        && cases
            .iter()
            .all(|(_, region)| is_terminal_region(ctx, *region))
}

/// Lower a terminal resultless switch without manufacturing an unreachable
/// merge block. The explicit default is required because an unmatched switch
/// must also transfer control.
fn lower_terminal_resultless_switch(
    ctx: &mut IrContext,
    block: BlockRef,
    scf_op: OpRef,
    loc: Location,
    discriminant: ValueRef,
    cases: Vec<(Attribute, RegionRef)>,
    default_region: RegionRef,
) {
    let parent_region = ctx.block(block).parent_region.unwrap();
    let blocks = ctx.region(parent_region).blocks.to_vec();
    let insert_before = blocks
        .iter()
        .position(|&candidate| candidate == block)
        .and_then(|index| blocks.get(index + 1).copied());

    ctx.detach_op(scf_op);
    let mut case_entries = Vec::new();
    let mut all_inlined = Vec::new();
    for (_, case_region) in &cases {
        let inlined = inline_region_blocks(ctx, *case_region, parent_region, insert_before);
        case_entries.push(inlined[0]);
        all_inlined.push(inlined);
    }
    let default_blocks = inline_region_blocks(ctx, default_region, parent_region, insert_before);
    let default_entry = default_blocks[0];
    all_inlined.push(default_blocks);
    ctx.remove_op(scf_op);

    SwitchDispatch {
        discriminant,
        cases: &cases,
        case_entries: &case_entries,
        default_entry,
        parent_region,
        insert_before,
    }
    .append(ctx, block, loc);

    for group in &all_inlined {
        for &branch in group {
            transform_block(ctx, branch);
        }
    }
}

/// Lower `scf.switch` to chained cond_br comparisons.
fn lower_scf_switch(ctx: &mut IrContext, block: BlockRef, scf_op: OpRef, loc: Location) {
    let Some(arms) = switch_arms(ctx, scf_op) else {
        return;
    };

    if is_terminal_resultless_switch(ctx, block, scf_op, &arms.cases, arms.default_region) {
        lower_terminal_resultless_switch(
            ctx,
            block,
            scf_op,
            loc,
            arms.discriminant,
            arms.cases,
            arms.default_region.unwrap(),
        );
        return;
    }

    let SwitchArms {
        discriminant,
        cases,
        default_region,
    } = arms;

    // Split block at scf op: ops after go to merge block
    let merge_block = split_block(ctx, block, scf_op);

    // Remove the scf op (split_block moved it to merge_block)
    ctx.detach_op(scf_op);

    let parent_region = ctx.block(block).parent_region.unwrap();

    // Inline and transform case regions
    let mut case_entries: Vec<BlockRef> = Vec::new();
    let mut all_inlined: Vec<Vec<BlockRef>> = Vec::new();

    for (_, case_region) in &cases {
        let inlined = inline_region_blocks(ctx, *case_region, parent_region, Some(merge_block));
        replace_yield_with_br(ctx, &inlined, merge_block, loc);
        case_entries.push(inlined[0]);
        all_inlined.push(inlined);
    }

    // Inline default region
    let default_entry = if let Some(def_region) = default_region {
        let inlined = inline_region_blocks(ctx, def_region, parent_region, Some(merge_block));
        replace_yield_with_br(ctx, &inlined, merge_block, loc);
        let entry = inlined[0];
        all_inlined.push(inlined);
        entry
    } else {
        // No default: create a block that branches to merge with no args
        let default_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });
        let br = cf::br(
            ctx,
            loc,
            std::iter::empty::<crate::refs::ValueRef>(),
            merge_block,
        );
        ctx.push_op(default_block, br.op_ref());
        // Insert into parent region before merge
        let merge_pos = ctx
            .region(parent_region)
            .blocks
            .iter()
            .position(|&b| b == merge_block)
            .unwrap();
        ctx.region_mut(parent_region)
            .blocks
            .insert(merge_pos, default_block);
        ctx.block_mut(default_block).parent_region = Some(parent_region);
        default_block
    };
    ctx.remove_op(scf_op);

    SwitchDispatch {
        discriminant,
        cases: &cases,
        case_entries: &case_entries,
        default_entry,
        parent_region,
        insert_before: Some(merge_block),
    }
    .append(ctx, block, loc);

    // Recursively transform inlined blocks
    for group in &all_inlined {
        for &b in group {
            transform_block(ctx, b);
        }
    }
}

/// Add the CFG dispatch chain for a switch whose arm entries are already in
/// the parent region.
impl SwitchDispatch<'_> {
    fn append(self, ctx: &mut IrContext, block: BlockRef, loc: Location) {
        let Self {
            discriminant,
            cases,
            case_entries,
            default_entry,
            parent_region,
            insert_before,
        } = self;

        // Get the discriminant type for comparisons
        let disc_ty = ctx.value_ty(discriminant);

        if cases.is_empty() {
            // No cases: branch directly to default
            let br = cf::br(
                ctx,
                loc,
                std::iter::empty::<crate::refs::ValueRef>(),
                default_entry,
            );
            ctx.push_op(block, br.op_ref());
        } else {
            // Build chained comparisons
            // We'll use the entry block for the first comparison, and create
            // new blocks for subsequent comparisons.
            let mut current_block = block;

            for (i, ((case_attr, _), &case_entry)) in cases.iter().zip(case_entries).enumerate() {
                let is_last = i == cases.len() - 1;

                // Create comparison: discriminant == case_value
                let case_const = arith::r#const(ctx, loc, disc_ty, case_attr.clone());
                ctx.push_op(current_block, case_const.op_ref());

                let i1_ty = ctx.types.intern(
                    crate::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1"))
                        .build(),
                );
                let cmp = arith::cmpi(
                    ctx,
                    loc,
                    discriminant,
                    case_const.result(ctx),
                    i1_ty,
                    Symbol::new("eq"),
                );
                ctx.push_op(current_block, cmp.op_ref());

                let else_target = if is_last {
                    default_entry
                } else {
                    // Create next check block
                    let next_block = ctx.create_block(BlockData {
                        location: loc,
                        args: vec![],
                        ops: SmallVec::new(),
                        parent_region: None,
                    });
                    // Insert subsequent checks alongside the already-inlined arms.
                    let insert_pos = insert_before
                        .and_then(|boundary| {
                            ctx.region(parent_region)
                                .blocks
                                .iter()
                                .position(|&candidate| candidate == boundary)
                        })
                        .unwrap_or(ctx.region(parent_region).blocks.len());
                    ctx.region_mut(parent_region)
                        .blocks
                        .insert(insert_pos, next_block);
                    ctx.block_mut(next_block).parent_region = Some(parent_region);
                    next_block
                };

                let cond_br = cf::cond_br(ctx, loc, cmp.result(ctx), case_entry, else_target);
                ctx.push_op(current_block, cond_br.op_ref());

                if !is_last {
                    current_block = else_target;
                }
            }
        }
    }
}

/// Replace `scf.yield` ops in the given blocks with `cf.br` to the target block.
///
/// Only passes as many yield values as the target block expects arguments,
/// ensuring branch arg counts always match the target block's block args.
fn replace_yield_with_br(
    ctx: &mut IrContext,
    blocks: &[BlockRef],
    target: BlockRef,
    loc: Location,
) {
    let target_arg_count = ctx.block_args(target).len();
    for &block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if scf::Yield::matches(ctx, op) {
                let yield_op = scf::Yield::from_op(ctx, op).unwrap();
                let values: Vec<_> = yield_op
                    .values(ctx)
                    .iter()
                    .copied()
                    .take(target_arg_count)
                    .collect();
                let br = cf::br(ctx, loc, values, target);

                // Replace yield with br in-place
                crate::rewrite::erase_op(ctx, op);
                ctx.push_op(block, br.op_ref());
            }
        }
    }
}

/// Replace `scf.continue` and `scf.break` ops in blocks with `cf.br`.
///
/// This operates only on the immediate blocks. Nested scf.loop regions
/// are left alone (their continue/break are handled when that loop is lowered).
fn replace_continue_break(
    ctx: &mut IrContext,
    blocks: &[BlockRef],
    header: BlockRef,
    exit: BlockRef,
    loc: Location,
) {
    let exit_arg_count = ctx.block_args(exit).len();
    for &block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if scf::Continue::matches(ctx, op) {
                let cont_op = scf::Continue::from_op(ctx, op).unwrap();
                let values: Vec<_> = cont_op.values(ctx).to_vec();
                let br = cf::br(ctx, loc, values, header);
                crate::rewrite::erase_op(ctx, op);
                ctx.push_op(block, br.op_ref());
            } else if scf::Break::matches(ctx, op) {
                let break_op = scf::Break::from_op(ctx, op).unwrap();
                let value = break_op.value(ctx);
                let values = (exit_arg_count != 0).then_some(value);
                let br = cf::br(ctx, loc, values, exit);
                crate::rewrite::erase_op(ctx, op);
                ctx.push_op(block, br.op_ref());
            } else {
                // Recurse into nested regions for continue/break replacement,
                // but skip nested scf.loop ops (their continue/break are theirs).
                if scf::Loop::matches(ctx, op) {
                    continue;
                }
                let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
                for region in regions {
                    let region_blocks = ctx.region(region).blocks.to_vec();
                    replace_continue_break(ctx, &region_blocks, header, exit, loc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core, func, scf};
    use crate::location::Span;
    use crate::refs::ValueRef;
    use crate::symbol::Symbol;
    use crate::*;
    use smallvec::smallvec;
    use std::ops::ControlFlow;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn i1_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build())
    }

    fn nil_type(ctx: &mut IrContext) -> TypeRef {
        core::nil(ctx).as_type_ref()
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        let nil_ty = crate::dialect::core::nil(ctx).as_type_ref();
        crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref()
    }

    fn build_module(ctx: &mut IrContext, loc: Location, func_ops: Vec<OpRef>) -> Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in func_ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).unwrap()
    }

    /// Collect all op names from a region (dialect.name format).
    fn collect_op_names(ctx: &IrContext, region: RegionRef) -> Vec<String> {
        let mut names = Vec::new();
        let _ = crate::walk::walk_region::<()>(ctx, region, &mut |op| {
            let d = ctx.op(op).dialect;
            let n = ctx.op(op).name;
            d.with_str(|ds| n.with_str(|ns| names.push(format!("{ds}.{ns}"))));
            ControlFlow::Continue(WalkAction::Advance)
        });
        names
    }

    /// Count blocks in a region.
    fn count_blocks(ctx: &IrContext, region: RegionRef) -> usize {
        ctx.region(region).blocks.len()
    }

    fn resultless_if(
        ctx: &mut IrContext,
        loc: Location,
        cond: ValueRef,
        then_region: RegionRef,
        else_region: RegionRef,
    ) -> scf::If {
        let data = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond)
            .region(then_region)
            .region(else_region)
            .build(ctx);
        let op = ctx.create_op(data);
        scf::If::from_op(ctx, op).unwrap()
    }

    fn resultless_loop(
        ctx: &mut IrContext,
        loc: Location,
        init: impl IntoIterator<Item = ValueRef>,
        body: RegionRef,
    ) -> scf::Loop {
        let data = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("loop"))
            .operands(init)
            .region(body)
            .build(ctx);
        let op = ctx.create_op(data);
        scf::Loop::from_op(ctx, op).unwrap()
    }

    fn build_void_if_func(ctx: &mut IrContext, loc: Location, name: &'static str) -> func::Func {
        let i1_ty = i1_type(ctx);
        let fn_ty = fn_type(ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let cond_const = arith::r#const(ctx, loc, i1_ty, Attribute::Bool(true));
        ctx.push_op(entry, cond_const.op_ref());

        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let then_yield = scf::r#yield(ctx, loc, std::iter::empty());
        ctx.push_op(then_block, then_yield.op_ref());
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let else_yield = scf::r#yield(ctx, loc, std::iter::empty());
        ctx.push_op(else_block, else_yield.op_ref());
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        let if_op = resultless_if(ctx, loc, cond_const.result(ctx), then_region, else_region);
        ctx.push_op(entry, if_op.op_ref());

        let ret = func::r#return(ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, Symbol::new(name), fn_ty, body_region)
    }

    #[test]
    fn lower_scf_if_basic() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i1_ty = i1_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        // %cond = arith.const true
        let cond_const = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Bool(true));
        ctx.push_op(entry, cond_const.op_ref());

        // then region: yield 42
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let then_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let then_v = then_val.result(&ctx);
        ctx.push_op(then_block, then_val.op_ref());
        let then_yield = scf::r#yield(&mut ctx, loc, [then_v]);
        ctx.push_op(then_block, then_yield.op_ref());
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        // else region: yield 0
        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let else_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        let else_v = else_val.result(&ctx);
        ctx.push_op(else_block, else_val.op_ref());
        let else_yield = scf::r#yield(&mut ctx, loc, [else_v]);
        ctx.push_op(else_block, else_yield.op_ref());
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        // scf.if
        let cond_v = cond_const.result(&ctx);
        let if_op = scf::r#if(&mut ctx, loc, cond_v, i32_ty, then_region, else_region);
        ctx.push_op(entry, if_op.op_ref());

        // Use the if result
        let if_result = if_op.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [if_result]);
        ctx.push_op(entry, ret.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, body_region);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        // Lower scf to cf
        lower_scf_to_cf(&mut ctx, module);
        let use_chains = crate::validation::validate_use_chains(&ctx, module);
        assert!(
            use_chains.is_ok(),
            "lowering must preserve use-chain consistency: {use_chains}"
        );

        // Verify: no scf ops remain
        let func_body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, func_body);
        assert!(
            !names.iter().any(|n| n.starts_with("scf.")),
            "scf ops remain: {names:?}"
        );

        // Should have cf.cond_br and cf.br ops
        assert!(
            names.iter().any(|n| n == "cf.cond_br"),
            "missing cf.cond_br: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "cf.br"),
            "missing cf.br: {names:?}"
        );

        // Should have 4 blocks: entry, then, else, merge
        assert_eq!(count_blocks(&ctx, func_body), 4);
    }

    #[test]
    fn lower_scf_to_cf_func_rewrites_only_selected_function() {
        let (mut ctx, loc) = test_ctx();
        let selected = build_void_if_func(&mut ctx, loc, "selected");
        let untouched = build_void_if_func(&mut ctx, loc, "untouched");
        let _module = build_module(&mut ctx, loc, vec![selected.op_ref(), untouched.op_ref()]);

        lower_scf_to_cf_func(&mut ctx, selected);

        let selected_names = collect_op_names(&ctx, selected.body(&ctx));
        let untouched_names = collect_op_names(&ctx, untouched.body(&ctx));

        assert!(
            !selected_names.iter().any(|n| n.starts_with("scf.")),
            "selected function still has scf ops: {selected_names:?}"
        );
        assert!(
            selected_names.iter().any(|n| n == "cf.cond_br"),
            "selected function should contain cf.cond_br: {selected_names:?}"
        );
        assert!(
            untouched_names.iter().any(|n| n.starts_with("scf.")),
            "untouched function should retain scf ops: {untouched_names:?}"
        );
    }

    #[test]
    fn scf_to_cf_pass_leaves_bodyless_func_unchanged() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);
        let func_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("external")))
            .attr("type", Attribute::Type(fn_ty))
            .build(&mut ctx);
        let func_op = ctx.create_op(func_data);
        let module = build_module(&mut ctx, loc, vec![func_op]);
        let before = crate::printer::print_module(&ctx, module.op());
        let func = func::Func::from_op(&ctx, func_op).unwrap();

        scf_to_cf_pass().run(&mut ctx, func).unwrap();

        assert!(func.body_if_present(&ctx).is_none());
        assert!(ctx.op(func_op).regions.is_empty());
        assert_eq!(crate::printer::print_module(&ctx, module.op()), before);
    }

    #[test]
    fn lower_scf_if_void() {
        let (mut ctx, loc) = test_ctx();
        let i1_ty = i1_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let cond_const = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Bool(true));
        ctx.push_op(entry, cond_const.op_ref());

        // then/else: yield nothing
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let then_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(then_block, then_yield.op_ref());
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let else_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(else_block, else_yield.op_ref());
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        let cond_v = cond_const.result(&ctx);
        let if_op = resultless_if(&mut ctx, loc, cond_v, then_region, else_region);
        ctx.push_op(entry, if_op.op_ref());

        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, body_region);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);
        let use_chains = crate::validation::validate_use_chains(&ctx, module);
        assert!(
            use_chains.is_ok(),
            "void lowering must preserve use-chain consistency: {use_chains}"
        );

        let func_body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, func_body);
        assert!(!names.iter().any(|n| n.starts_with("scf.")));
        // Nil-typed scf.if results are not materialized as CFG values.
        let blocks = ctx.region(func_body).blocks.to_vec();
        let merge = blocks.last().unwrap();
        assert_eq!(ctx.block_args(*merge).len(), 0);
    }

    fn assert_lowered_unit_tail_transfer(input: &str, expected_nil_merge_args: usize) {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let operation_verifiers = crate::validation::validate_operation_verifiers(&ctx, module);
        assert!(operation_verifiers.is_ok(), "{operation_verifiers}");

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, body);
        assert!(!names.iter().any(|name| name.starts_with("scf.")));
        let use_chains = crate::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let nil_ty = nil_type(&mut ctx);
        let nil_merge_args = ctx
            .region(body)
            .blocks
            .iter()
            .skip(1)
            .flat_map(|&block| ctx.block_args(block))
            .filter(|&&arg| ctx.value_ty(arg) == nil_ty)
            .count();
        assert_eq!(nil_merge_args, expected_nil_merge_args);

        let printed = crate::printer::print_module(&ctx, module.op());
        assert!(
            printed.contains("!t0 = func.func_sig<(core.nil) -> core.never>"),
            "{printed}"
        );
        assert!(printed.contains("signature = !t0"), "{printed}");
        assert!(printed.contains("tribute.calling_convention = 2"));
    }

    #[test]
    fn lower_scf_if_preserves_used_nil_result_for_tail_transfer() {
        assert_lowered_unit_tail_transfer(
            r#"core.module @test {
  func.func @main(%cond: core.i1, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    %selected = scf.if %cond : core.nil {
      scf.yield %unit
    } {
      scf.yield %unit
    }
    func.tail_call_indirect %callee, %selected {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
  }
}"#,
            1,
        );
    }

    #[test]
    fn lower_scf_if_drops_unused_never_result_for_tail_transfers() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    %discarded = scf.if %cond : core.never {
      func.unreachable
    } {
      func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let operation_verifiers = crate::validation::validate_operation_verifiers(&ctx, module);
        assert!(operation_verifiers.is_ok(), "{operation_verifiers}");

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, body);
        assert!(!names.iter().any(|name| name.starts_with("scf.")));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "func.tail_call_indirect")
                .count(),
            1
        );
        assert!(names.iter().any(|name| name == "func.unreachable"));
        assert!(crate::validation::validate_use_chains(&ctx, module).is_ok());
        assert!(
            crate::validation::validate_operation_verifiers(&ctx, module).is_ok(),
            "terminal lowering must keep the CFG well-formed"
        );

        let has_non_entry_never_arg = ctx
            .region(body)
            .blocks
            .iter()
            .skip(1)
            .flat_map(|&block| ctx.block_args(block))
            .any(|&arg| {
                let ty = ctx.types.get(ctx.value_ty(arg));
                ty.dialect == Symbol::new("core") && ty.name == Symbol::new("never")
            });
        assert!(
            !has_non_entry_never_arg,
            "unused scf.if result must not create a core.never CFG block argument"
        );
        assert_eq!(
            count_blocks(&ctx, body),
            3,
            "a terminal core.never scf.if must not leave an empty merge block"
        );
    }

    #[test]
    fn lower_terminal_scf_switch_without_empty_merge_block() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        func.unreachable
      }
      scf.default {
        func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
      }
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, body);
        assert!(
            !names.iter().any(|name| name.starts_with("scf.")),
            "terminal switch must be fully lowered: {names:?}"
        );
        assert!(names.iter().any(|name| name == "cf.cond_br"));
        assert!(names.iter().any(|name| name == "func.unreachable"));
        assert!(names.iter().any(|name| name == "func.tail_call_indirect"));
        assert_eq!(
            count_blocks(&ctx, body),
            3,
            "terminal switch needs only the dispatch and two terminal arms"
        );
        assert!(
            ctx.region(body)
                .blocks
                .iter()
                .all(|&block| !ctx.block(block).ops.is_empty()),
            "terminal switch must not leave an empty CFG block"
        );
        let use_chains = crate::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let operation_verifiers = crate::validation::validate_operation_verifiers(&ctx, module);
        assert!(operation_verifiers.is_ok(), "{operation_verifiers}");
    }

    #[test]
    fn lower_terminal_scf_switch_with_nested_never_if_has_no_merge_block() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32, %cond: core.i1, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        %discarded = scf.if %cond : core.never {
          func.unreachable
        } {
          func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
        }
      }
      scf.default {
        func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
      }
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let blocks = ctx.region(body).blocks.to_vec();
        let names = collect_op_names(&ctx, body);
        assert!(!names.iter().any(|name| name.starts_with("scf.")));
        assert_eq!(count_blocks(&ctx, body), 5);
        for &block in blocks.iter().skip(1) {
            assert!(
                !ctx.block(block).ops.is_empty(),
                "terminal lowering left an empty block: {block}"
            );
            assert!(
                blocks.iter().any(|&candidate| {
                    ctx.block(candidate)
                        .ops
                        .iter()
                        .any(|&op| ctx.op(op).successors.contains(&block))
                }),
                "terminal lowering left a predecessor-free block: {block}"
            );
        }
        assert!(crate::validation::validate_use_chains(&ctx, module).is_ok());
        assert!(crate::validation::validate_operation_verifiers(&ctx, module).is_ok());
    }

    #[test]
    fn lower_terminal_scf_switch_with_nested_terminal_switch_has_no_empty_merge_block() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32, %nested_choice: core.i32, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        scf.switch %nested_choice {
          scf.case {value = 0} {
            func.unreachable
          }
          scf.default {
            func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
          }
        }
      }
      scf.default {
        func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
      }
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let blocks = ctx.region(body).blocks.to_vec();
        let names = collect_op_names(&ctx, body);
        assert!(!names.iter().any(|name| name.starts_with("scf.")));
        assert_eq!(count_blocks(&ctx, body), 5);
        assert!(
            blocks.iter().skip(1).all(|&block| {
                !ctx.block(block).ops.is_empty()
                    && blocks.iter().any(|&candidate| {
                        ctx.block(candidate)
                            .ops
                            .iter()
                            .any(|&op| ctx.op(op).successors.contains(&block))
                    })
            }),
            "terminal switches must not leave a predecessor-free empty merge block"
        );
        assert!(crate::validation::validate_use_chains(&ctx, module).is_ok());
        assert!(crate::validation::validate_operation_verifiers(&ctx, module).is_ok());
    }

    #[test]
    fn lower_nonterminal_scf_switch_keeps_merge_continuation() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.case {value = 0} {
        scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    %unit = arith.const {value = 0} : core.nil
    func.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let blocks = ctx.region(body).blocks.to_vec();
        let continuation = *blocks.last().unwrap();
        let names = collect_op_names(&ctx, body);
        assert!(!names.iter().any(|name| name.starts_with("scf.")));
        assert_eq!(count_blocks(&ctx, body), 4);
        assert!(
            ctx.block(continuation).ops.iter().any(|&op| {
                ctx.op(op).dialect == Symbol::new("func")
                    && ctx.op(op).name == Symbol::new("return")
            }),
            "ordinary switch continuation must remain in its merge block"
        );
        assert_eq!(
            blocks
                .iter()
                .flat_map(|&block| ctx.block(block).ops.iter())
                .filter(|&&op| ctx.op(op).successors.as_slice() == [continuation])
                .count(),
            2,
            "both yielding arms must branch to the merge continuation"
        );
        assert!(crate::validation::validate_use_chains(&ctx, module).is_ok());
        assert!(crate::validation::validate_operation_verifiers(&ctx, module).is_ok());
    }

    #[test]
    fn lower_nonterminal_scf_switch_with_multiblock_arm_keeps_merge_continuation() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.case {value = 0} {
        ^first:
          cf.br [^second]
        ^second:
          scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    %unit = arith.const {value = 0} : core.nil
    func.return %unit
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let func_op = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let blocks = ctx.region(body).blocks.to_vec();
        let continuation = *blocks.last().unwrap();
        let names = collect_op_names(&ctx, body);
        assert!(
            !names.iter().any(|name| name.starts_with("scf.")),
            "nonterminal multiblock switch must be fully lowered: {names:?}"
        );
        assert!(
            ctx.block(continuation).ops.iter().any(|&op| {
                ctx.op(op).dialect == Symbol::new("func")
                    && ctx.op(op).name == Symbol::new("return")
            }),
            "ordinary switch continuation must remain in its merge block"
        );
        assert_eq!(
            blocks
                .iter()
                .flat_map(|&block| ctx.block(block).ops.iter())
                .filter(|&&op| ctx.op(op).successors.as_slice() == [continuation])
                .count(),
            2,
            "both yielding paths must branch to the merge continuation"
        );
        assert!(crate::validation::validate_use_chains(&ctx, module).is_ok());
        assert!(crate::validation::validate_operation_verifiers(&ctx, module).is_ok());
    }

    #[test]
    fn lower_nested_nil_results_preserves_each_merge_value() {
        assert_lowered_unit_tail_transfer(
            r#"core.module @test {
  func.func @main(%cond: core.i1, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    %inner = scf.if %cond : core.nil {
      scf.yield %unit
    } {
      scf.yield %unit
    }
    %outer = scf.if %cond : core.nil {
      scf.yield %inner
    } {
      scf.yield %inner
    }
    func.tail_call_indirect %callee, %outer {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
  }
}"#,
            2,
        );
    }

    #[test]
    fn lower_scf_loop_basic() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        // init value
        let init = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        ctx.push_op(entry, init.op_ref());

        // Loop body: loop_arg -> break(loop_arg)
        let body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let loop_arg = ctx.block_arg(body_block, 0);
        let break_op = scf::r#break(&mut ctx, loc, loop_arg);
        ctx.push_op(body_block, break_op.op_ref());
        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![body_block],
            parent_op: None,
        });

        let init_v = init.result(&ctx);
        let loop_op = scf::r#loop(&mut ctx, loc, [init_v], i32_ty, body_region);
        let loop_result = loop_op.result(&ctx);
        ctx.push_op(entry, loop_op.op_ref());

        let ret = func::r#return(&mut ctx, loc, [loop_result]);
        ctx.push_op(entry, ret.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, func_body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, body);
        assert!(
            !names.iter().any(|n| n.starts_with("scf.")),
            "scf ops remain: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "cf.br"),
            "missing cf.br: {names:?}"
        );

        // Should have: entry, header (body), exit
        assert_eq!(count_blocks(&ctx, body), 3);
    }

    #[test]
    fn lower_scf_loop_resultless_drops_break_value() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let init = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        ctx.push_op(entry, init.op_ref());

        let body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let loop_arg = ctx.block_arg(body_block, 0);
        let break_op = scf::r#break(&mut ctx, loc, loop_arg);
        ctx.push_op(body_block, break_op.op_ref());
        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![body_block],
            parent_op: None,
        });

        let init_v = init.result(&ctx);
        let loop_op = resultless_loop(&mut ctx, loc, [init_v], body_region);
        ctx.push_op(entry, loop_op.op_ref());
        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, func_body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let blocks = ctx.region(body).blocks.to_vec();
        assert_eq!(blocks.len(), 3);
        assert!(
            !collect_op_names(&ctx, body)
                .iter()
                .any(|name| name.starts_with("scf."))
        );
        let exit = *blocks
            .iter()
            .find(|&&block| {
                ctx.block(block).ops.iter().any(|&op| {
                    ctx.op(op).dialect == Symbol::new("func")
                        && ctx.op(op).name == Symbol::new("return")
                })
            })
            .unwrap();
        assert!(ctx.block_args(exit).is_empty());
        for &block in &blocks {
            for &op in &ctx.block(block).ops {
                if ctx.op(op).successors.as_slice() == [exit] {
                    assert!(
                        ctx.op_operands(op).is_empty(),
                        "{} has operands {:?}",
                        ctx.op(op).name,
                        ctx.op_operands(op)
                    );
                }
            }
        }
    }

    #[test]
    fn lower_scf_switch_basic() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let disc = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(entry, disc.op_ref());

        // Case 0: resultless yield
        let case0_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let case0_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(case0_block, case0_yield.op_ref());
        let case0_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![case0_block],
            parent_op: None,
        });
        let case0_op = scf::case(&mut ctx, loc, Attribute::Int(0), case0_region);

        // Case 1: resultless yield
        let case1_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let case1_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(case1_block, case1_yield.op_ref());
        let case1_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![case1_block],
            parent_op: None,
        });
        let case1_op = scf::case(&mut ctx, loc, Attribute::Int(1), case1_region);

        // Default: resultless yield
        let default_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let default_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(default_block, default_yield.op_ref());
        let default_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![default_block],
            parent_op: None,
        });
        let default_op = scf::default(&mut ctx, loc, default_region);

        // Switch body region containing case and default ops
        let switch_body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(switch_body_block, case0_op.op_ref());
        ctx.push_op(switch_body_block, case1_op.op_ref());
        ctx.push_op(switch_body_block, default_op.op_ref());
        let switch_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![switch_body_block],
            parent_op: None,
        });

        let disc_v = disc.result(&ctx);
        let switch = scf::switch(&mut ctx, loc, disc_v, switch_body);
        ctx.push_op(entry, switch.op_ref());

        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, func_body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, body);
        assert!(
            !names.iter().any(|n| n.starts_with("scf.")),
            "scf ops remain: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "cf.cond_br"),
            "missing cf.cond_br: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "arith.cmpi"),
            "missing arith.cmpi: {names:?}"
        );
    }

    #[test]
    fn lower_scf_switch_no_result() {
        // scf.switch doesn't produce a result, so case regions yield no values.
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let disc = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(entry, disc.op_ref());

        // Case 0: void yield (no values)
        let case0_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let case0_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(case0_block, case0_yield.op_ref());
        let case0_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![case0_block],
            parent_op: None,
        });
        let case0_op = scf::case(&mut ctx, loc, Attribute::Int(0), case0_region);

        // Case 1: resultless yield
        let case1_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let case1_yield = scf::r#yield(&mut ctx, loc, std::iter::empty());
        ctx.push_op(case1_block, case1_yield.op_ref());
        let case1_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![case1_block],
            parent_op: None,
        });
        let case1_op = scf::case(&mut ctx, loc, Attribute::Int(1), case1_region);

        // Switch body
        let switch_body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(switch_body_block, case0_op.op_ref());
        ctx.push_op(switch_body_block, case1_op.op_ref());
        let switch_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![switch_body_block],
            parent_op: None,
        });

        let disc_v = disc.result(&ctx);
        let switch = scf::switch(&mut ctx, loc, disc_v, switch_body);
        ctx.push_op(entry, switch.op_ref());

        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, func_body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        let body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, body);
        assert!(
            !names.iter().any(|n| n.starts_with("scf.")),
            "scf ops remain: {names:?}"
        );

        // scf.switch has no result, so merge block should have 0 args
        let blocks = ctx.region(body).blocks.to_vec();
        let merge = blocks.last().unwrap();
        let merge_args = ctx.block_args(*merge);
        assert_eq!(
            merge_args.len(),
            0,
            "merge block should have 0 args since scf.switch has no result"
        );
    }

    #[test]
    fn lower_scf_switch_with_result_and_empty_body_fails_closed() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let disc = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        ctx.push_op(entry, disc.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![],
            parent_op: None,
        });
        let switch_data = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("switch"))
            .operand(disc.result(&ctx))
            .result(i32_ty)
            .region(body)
            .build(&mut ctx);
        let switch = ctx.create_op(switch_data);
        ctx.push_op(entry, switch);
        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, func_body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        let names = collect_op_names(&ctx, func_op.body(&ctx));
        assert!(names.iter().any(|name| name == "scf.switch"));
        assert!(!names.iter().any(|name| name == "cf.cond_br"));
    }

    #[test]
    fn no_scf_ops_is_noop() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        let val_v = val.result(&ctx);
        ctx.push_op(entry, val.op_ref());
        let ret = func::r#return(&mut ctx, loc, [val_v]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        // Should remain unchanged
        let func_body = func_op.body(&ctx);
        assert_eq!(count_blocks(&ctx, func_body), 1);
    }

    #[test]
    fn scf_if_result_rauw() {
        // Verify that RAUW correctly replaces scf.if result with merge block arg
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i1_ty = i1_type(&mut ctx);
        let fn_ty = fn_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Bool(true));
        ctx.push_op(entry, cond.op_ref());

        // then: yield 1
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let t_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        let t_v = t_val.result(&ctx);
        ctx.push_op(then_block, t_val.op_ref());
        let t_yield = scf::r#yield(&mut ctx, loc, [t_v]);
        ctx.push_op(then_block, t_yield.op_ref());
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        // else: yield 2
        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let e_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(2));
        let e_v = e_val.result(&ctx);
        ctx.push_op(else_block, e_val.op_ref());
        let e_yield = scf::r#yield(&mut ctx, loc, [e_v]);
        ctx.push_op(else_block, e_yield.op_ref());
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        let cond_v = cond.result(&ctx);
        let if_op = scf::r#if(&mut ctx, loc, cond_v, i32_ty, then_region, else_region);
        let if_result = if_op.result(&ctx);
        ctx.push_op(entry, if_op.op_ref());

        // Use the if result in an add
        let add = arith::addi(&mut ctx, loc, if_result, if_result, i32_ty);
        let add_result = add.result(&ctx);
        ctx.push_op(entry, add.op_ref());

        let ret = func::r#return(&mut ctx, loc, [add_result]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("test"), fn_ty, body);
        let module = build_module(&mut ctx, loc, vec![func_op.op_ref()]);

        lower_scf_to_cf(&mut ctx, module);

        // Verify: the add op's operands should now reference the merge block arg,
        // not the old if_op result.
        let func_body = func_op.body(&ctx);
        let blocks = ctx.region(func_body).blocks.to_vec();
        let merge = blocks.last().unwrap();
        let merge_args = ctx.block_args(*merge);
        assert_eq!(merge_args.len(), 1, "merge block should have 1 arg");

        // The add op should use the merge block arg
        let merge_ops: Vec<OpRef> = ctx.block(*merge).ops.to_vec();
        let add_op = merge_ops
            .iter()
            .find(|&&op| {
                ctx.op(op).dialect == Symbol::new("arith") && ctx.op(op).name == Symbol::new("addi")
            })
            .unwrap();
        let operands = ctx.op_operands(*add_op);
        assert_eq!(operands[0], merge_args[0]);
        assert_eq!(operands[1], merge_args[0]);
    }
}
