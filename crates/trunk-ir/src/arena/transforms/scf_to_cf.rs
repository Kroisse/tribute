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

use std::collections::BTreeMap;

use smallvec::SmallVec;

use crate::arena::context::{BlockArgData, BlockData, IrContext};
use crate::arena::dialect::{arith, cf, scf};
use crate::arena::ops::ArenaDialectOp;
use crate::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef};
use crate::arena::rewrite::ArenaModule;
use crate::arena::rewrite::helpers::{inline_region_blocks, split_block};
use crate::arena::types::{Attribute, Location};
use crate::ir::Symbol;

/// Lower all `scf` operations in a module to `cf` operations.
pub fn lower_scf_to_cf(ctx: &mut IrContext, module: ArenaModule) {
    let body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };
    transform_region(ctx, body);
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

/// Lower `scf.if` to cf.cond_br + then/else/merge blocks.
fn lower_scf_if(ctx: &mut IrContext, block: BlockRef, scf_op: OpRef, loc: Location) {
    let if_op = scf::If::from_op(ctx, scf_op).unwrap();
    let cond = if_op.cond(ctx);
    let then_region = if_op.then_region(ctx);
    let else_region = if_op.else_region(ctx);

    // Determine result type (if any)
    let results = ctx.op_results(scf_op);
    let result_ty = if results.is_empty() {
        None
    } else {
        let ty = ctx.value_ty(results[0]);
        let ty_data = ctx.types.get(ty);
        // Skip nil type (void)
        if ty_data.dialect == Symbol::new("core") && ty_data.name == Symbol::new("nil") {
            None
        } else {
            Some(ty)
        }
    };

    // Split block at the scf op: ops after scf_op go to merge block
    let merge_block = split_block(ctx, block, scf_op);

    // Remove the scf op from the original block
    ctx.remove_op_from_block(block, scf_op);

    // Add merge block argument for the result (if any)
    if let Some(ty) = result_ty {
        let merge_arg = ctx.add_block_arg(
            merge_block,
            BlockArgData {
                ty,
                attrs: BTreeMap::new(),
            },
        );
        // RAUW: replace all uses of scf.if result with merge block arg
        let if_result = ctx.op_results(scf_op)[0];
        ctx.replace_all_uses(if_result, merge_arg);
    }

    // Remove the scf op (split_block moved it to merge_block)
    ctx.detach_op(scf_op);

    // Inline then/else regions into the parent region
    let parent_region = ctx.block(block).parent_region.unwrap();
    let then_blocks = inline_region_blocks(ctx, then_region, parent_region, Some(merge_block));
    let else_blocks = inline_region_blocks(ctx, else_region, parent_region, Some(merge_block));

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
    let results = ctx.op_results(scf_op);
    let result_ty = if results.is_empty() {
        None
    } else {
        Some(ctx.value_ty(results[0]))
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
                attrs: BTreeMap::new(),
            },
        );
        let loop_result = ctx.op_results(scf_op)[0];
        ctx.replace_all_uses(loop_result, exit_arg);
    }

    // Inline body region blocks into parent region (before exit block)
    let parent_region = ctx.block(block).parent_region.unwrap();
    let body_blocks = inline_region_blocks(ctx, body_region, parent_region, Some(exit_block));

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

/// Lower `scf.switch` to chained cond_br comparisons.
fn lower_scf_switch(ctx: &mut IrContext, block: BlockRef, scf_op: OpRef, loc: Location) {
    let switch_op = scf::Switch::from_op(ctx, scf_op).unwrap();
    let discriminant = switch_op.discriminant(ctx);
    let body_region = switch_op.body(ctx);

    // Collect cases and default from the body region
    let body_blocks_vec = ctx.region(body_region).blocks.to_vec();
    let body_block = body_blocks_vec[0];
    let body_ops: Vec<OpRef> = ctx.block(body_block).ops.to_vec();

    let mut cases: Vec<(Attribute, RegionRef)> = Vec::new();
    let mut default_region: Option<RegionRef> = None;

    for &op in &body_ops {
        if let Ok(case_op) = scf::Case::from_op(ctx, op) {
            cases.push((case_op.value(ctx), case_op.body(ctx)));
        } else if scf::Default::matches(ctx, op) {
            let default_op = scf::Default::from_op(ctx, op).unwrap();
            default_region = Some(default_op.body(ctx));
        }
    }

    // Find result type from yielded values (if any)
    let result_ty = find_yield_type(ctx, cases.first().map(|(_, r)| *r))
        .or_else(|| find_yield_type(ctx, default_region));

    // Split block at scf op: ops after go to merge block
    let merge_block = split_block(ctx, block, scf_op);

    // Remove the scf op (split_block moved it to merge_block)
    ctx.detach_op(scf_op);

    // Add merge block argument for the result (if any)
    if let Some(ty) = result_ty {
        let merge_arg = ctx.add_block_arg(
            merge_block,
            BlockArgData {
                ty,
                attrs: BTreeMap::new(),
            },
        );
        let results = ctx.op_results(scf_op);
        if !results.is_empty() {
            ctx.replace_all_uses(results[0], merge_arg);
        }
    }

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
            std::iter::empty::<crate::arena::refs::ValueRef>(),
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

    // Get the discriminant type for comparisons
    let disc_ty = ctx.value_ty(discriminant);

    if cases.is_empty() {
        // No cases: branch directly to default
        let br = cf::br(
            ctx,
            loc,
            std::iter::empty::<crate::arena::refs::ValueRef>(),
            default_entry,
        );
        ctx.push_op(block, br.op_ref());
    } else {
        // Build chained comparisons
        // We'll use the entry block for the first comparison, and create
        // new blocks for subsequent comparisons.
        let mut current_block = block;

        for (i, ((case_attr, _), &case_entry)) in cases.iter().zip(case_entries.iter()).enumerate()
        {
            let is_last = i == cases.len() - 1;

            // Create comparison: discriminant == case_value
            let case_const = arith::r#const(ctx, loc, disc_ty, case_attr.clone());
            ctx.push_op(current_block, case_const.op_ref());

            let i1_ty = ctx.types.intern(
                crate::arena::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1"))
                    .build(),
            );
            let cmp = arith::cmp_eq(ctx, loc, discriminant, case_const.result(ctx), i1_ty);
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
                // Insert into parent region before merge
                let merge_pos = ctx
                    .region(parent_region)
                    .blocks
                    .iter()
                    .position(|&b| b == merge_block)
                    .unwrap();
                ctx.region_mut(parent_region)
                    .blocks
                    .insert(merge_pos, next_block);
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

    // Recursively transform inlined blocks
    for group in &all_inlined {
        for &b in group {
            transform_block(ctx, b);
        }
    }
}

/// Replace `scf.yield` ops in the given blocks with `cf.br` to the target block.
fn replace_yield_with_br(
    ctx: &mut IrContext,
    blocks: &[BlockRef],
    target: BlockRef,
    loc: Location,
) {
    for &block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if scf::Yield::matches(ctx, op) {
                let yield_op = scf::Yield::from_op(ctx, op).unwrap();
                let values: Vec<_> = yield_op.values(ctx).to_vec();
                let br = cf::br(ctx, loc, values, target);

                // Replace yield with br in-place
                ctx.remove_op_from_block(block, op);
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
    for &block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if scf::Continue::matches(ctx, op) {
                let cont_op = scf::Continue::from_op(ctx, op).unwrap();
                let values: Vec<_> = cont_op.values(ctx).to_vec();
                let br = cf::br(ctx, loc, values, header);
                ctx.remove_op_from_block(block, op);
                ctx.push_op(block, br.op_ref());
            } else if scf::Break::matches(ctx, op) {
                let break_op = scf::Break::from_op(ctx, op).unwrap();
                let value = break_op.value(ctx);
                let br = cf::br(ctx, loc, [value], exit);
                ctx.remove_op_from_block(block, op);
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

/// Find the type of values yielded from a region.
fn find_yield_type(ctx: &IrContext, region: Option<RegionRef>) -> Option<TypeRef> {
    let region = region?;
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if scf::Yield::matches(ctx, op) {
                let yield_op = scf::Yield::from_op(ctx, op).unwrap();
                let values = yield_op.values(ctx);
                if !values.is_empty() {
                    return Some(ctx.value_ty(values[0]));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::dialect::{arith, func, scf};
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
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
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn")).build())
    }

    fn build_module(ctx: &mut IrContext, loc: Location, func_ops: Vec<OpRef>) -> ArenaModule {
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
        ArenaModule::new(ctx, module_op).unwrap()
    }

    /// Collect all op names from a region (dialect.name format).
    fn collect_op_names(ctx: &IrContext, region: RegionRef) -> Vec<String> {
        let mut names = Vec::new();
        let _ = crate::arena::walk::walk_region::<()>(ctx, region, &mut |op| {
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
        let then_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
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
        let else_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
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
    fn lower_scf_if_void() {
        let (mut ctx, loc) = test_ctx();
        let nil_ty = nil_type(&mut ctx);
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
        let if_op = scf::r#if(&mut ctx, loc, cond_v, nil_ty, then_region, else_region);
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

        let func_body = func_op.body(&ctx);
        let names = collect_op_names(&ctx, func_body);
        assert!(!names.iter().any(|n| n.starts_with("scf.")));
        // Merge block should have no args (void if)
        let blocks = ctx.region(func_body).blocks.to_vec();
        let merge = blocks.last().unwrap();
        assert_eq!(ctx.block_args(*merge).len(), 0);
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
        let init = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
        ctx.push_op(entry, init.op_ref());

        // Loop body: loop_arg -> break(loop_arg)
        let body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
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

        let disc = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        ctx.push_op(entry, disc.op_ref());

        // Case 0: yield 10
        let case0_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let case0_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(10));
        let case0_v = case0_val.result(&ctx);
        ctx.push_op(case0_block, case0_val.op_ref());
        let case0_yield = scf::r#yield(&mut ctx, loc, [case0_v]);
        ctx.push_op(case0_block, case0_yield.op_ref());
        let case0_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![case0_block],
            parent_op: None,
        });
        let case0_op = scf::case(&mut ctx, loc, Attribute::IntBits(0), case0_region);

        // Case 1: yield 20
        let case1_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let case1_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(20));
        let case1_v = case1_val.result(&ctx);
        ctx.push_op(case1_block, case1_val.op_ref());
        let case1_yield = scf::r#yield(&mut ctx, loc, [case1_v]);
        ctx.push_op(case1_block, case1_yield.op_ref());
        let case1_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![case1_block],
            parent_op: None,
        });
        let case1_op = scf::case(&mut ctx, loc, Attribute::IntBits(1), case1_region);

        // Default: yield 0
        let default_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let default_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
        let default_v = default_val.result(&ctx);
        ctx.push_op(default_block, default_val.op_ref());
        let default_yield = scf::r#yield(&mut ctx, loc, [default_v]);
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
            names.iter().any(|n| n == "arith.cmp_eq"),
            "missing arith.cmp_eq: {names:?}"
        );
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
        let val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
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
        let t_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
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
        let e_val = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(2));
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
        let add = arith::add(&mut ctx, loc, if_result, if_result, i32_ty);
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
                ctx.op(op).dialect == Symbol::new("arith") && ctx.op(op).name == Symbol::new("add")
            })
            .unwrap();
        let operands = ctx.op_operands(*add_op);
        assert_eq!(operands[0], merge_args[0]);
        assert_eq!(operands[1], merge_args[0]);
    }
}
