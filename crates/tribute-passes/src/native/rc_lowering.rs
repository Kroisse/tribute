//! RC lowering pass: lower `tribute_rt.retain`/`release` to inline `clif.*` ops.
//!
//! ## Pipeline Position
//!
//! Runs after `resolve_unrealized_casts` (Phase 3) and before `emit_module_to_native`
//! (Phase 4). At this point all types are resolved, and after lowering no
//! `tribute_rt.*` ops should remain.
//!
//! ## Retain Lowering (null-guarded, block split)
//!
//! ```text
//! // --- entry block ---
//! %null    = clif.iconst(0) : core.ptr
//! %is_null = clif.icmp(ptr, %null, eq) : core.i8
//! clif.brif(%is_null, ^skip_block, ^do_retain_block)
//!
//! ^do_retain_block:
//!   %hdr_sz  = clif.iconst(8) : core.i64
//!   %rc_addr = clif.isub(ptr, %hdr_sz) : core.ptr
//!   %rc      = clif.load(%rc_addr, offset=0) : core.i32
//!   %one     = clif.iconst(1) : core.i32
//!   %new_rc  = clif.iadd(%rc, %one) : core.i32
//!   clif.store(%new_rc, %rc_addr, offset=0)
//!   clif.jump(^skip_block)
//!
//! ^skip_block:
//!   // remaining ops (retain result remapped to original ptr)
//! ```
//!
//! ## Release Lowering (null-guarded, block split)
//!
//! ```text
//! // --- entry block ---
//! %null    = clif.iconst(0) : core.ptr
//! %is_null = clif.icmp(ptr, %null, eq) : core.i8
//! clif.brif(%is_null, ^skip_block, ^do_release_block)
//!
//! ^do_release_block:
//!   %hdr_sz  = clif.iconst(8) : core.i64
//!   %rc_addr = clif.isub(ptr, %hdr_sz) : core.ptr
//!   %rc      = clif.load(%rc_addr, offset=0) : core.i32
//!   %one     = clif.iconst(1) : core.i32
//!   %new_rc  = clif.isub(%rc, %one) : core.i32
//!   clif.store(%new_rc, %rc_addr, offset=0)
//!   %zero    = clif.iconst(0) : core.i32
//!   %is_zero = clif.icmp(%new_rc, %zero, eq) : core.i8
//!   clif.brif(%is_zero, ^free_block, ^skip_block)
//!
//! ^free_block:
//!   clif.call @__tribute_deep_release(ptr, size)
//!   clif.jump(^skip_block)
//!
//! ^skip_block:
//!   // remaining ops
//! ```

use std::collections::HashSet;

use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::arena::rewrite::helpers::erase_op;
use trunk_ir::arena::{BlockData, BlockRef, OpRef, RegionRef, TypeRef, ValueRef};

use tribute_ir::arena::dialect::tribute_rt;

/// Name of the deep release dispatch function.
const DEEP_RELEASE_FN: &str = "__tribute_deep_release";

/// Lower all `tribute_rt.retain` and `tribute_rt.release` operations to
/// inline `clif.*` operations.
pub fn lower_rc(ctx: &mut IrContext, module: ArenaModule) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in module_ops {
        if let Ok(_func_op) = clif::Func::from_op(ctx, op) {
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for region in regions {
                lower_rc_in_region(ctx, region);
            }
        }
    }
}

/// Recursively lower RC ops in a region.
fn lower_rc_in_region(ctx: &mut IrContext, region: RegionRef) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.clone().to_vec();
    let original_blocks: HashSet<BlockRef> = blocks.iter().copied().collect();

    for block in blocks {
        // Step 1: Lower retain/release ops (may split block → add blocks to region)
        lower_rc_in_block(ctx, region, block);

        // Step 2: Recursively process nested regions in each block's ops
        // (need to re-read ops since block may have been modified)
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            let nested: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for nested_region in nested {
                lower_rc_in_region(ctx, nested_region);
            }
        }
    }

    // Also process nested regions in newly-created blocks (do_retain, do_release, free)
    // These blocks were added to the region during step 1
    let all_blocks: Vec<BlockRef> = ctx.region(region).blocks.clone().to_vec();
    for block in &all_blocks {
        if original_blocks.contains(block) {
            continue;
        }
        let ops: Vec<OpRef> = ctx.block(*block).ops.to_vec();
        for op in ops {
            let nested: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for nested_region in nested {
                lower_rc_in_region(ctx, nested_region);
            }
        }
    }
}

/// Lower RC ops in a single block, potentially splitting it.
///
/// Uses a forward scan approach:
/// - When a retain is found: split block, create do_retain + skip blocks
/// - When a release is found: split block, create do_release + free + skip blocks
/// - Subsequent ops move to the skip/continue block
fn lower_rc_in_block(ctx: &mut IrContext, region: RegionRef, block: BlockRef) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let loc = ctx.block(block).location;

    // Quick check: any RC ops?
    let has_rc_ops = ops.iter().any(|&op| {
        tribute_rt::Retain::from_op(ctx, op).is_ok()
            || tribute_rt::Release::from_op(ctx, op).is_ok()
    });

    if !has_rc_ops {
        return;
    }

    let ptr_ty = intern_type(ctx, "core", "ptr");
    let i64_ty = intern_type(ctx, "core", "i64");
    let i32_ty = intern_type(ctx, "core", "i32");
    let i8_ty = intern_type(ctx, "core", "i8");
    let nil_ty = intern_type(ctx, "core", "nil");

    // Process the block: iterate ops, when we hit an RC op, split.
    // `current_block` is where ops before the RC op live.
    // After splitting, `current_block` is the skip/continue block.
    let current_block = block;
    let mut i = 0;

    while i < ops.len() {
        let op = ops[i];

        if let Ok(_retain_op) = tribute_rt::Retain::from_op(ctx, op) {
            let ptr = ctx.op_operands(op).to_vec()[0];
            let retain_result = ctx.op_result(op, 0);

            // Create skip block (continuation)
            let skip_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });

            // Create do_retain block
            let do_retain_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });

            // Generate null check ops in current_block
            let null_val = clif::iconst(ctx, loc, ptr_ty, 0);
            ctx.push_op(current_block, null_val.op_ref());
            let is_null = clif::icmp(
                ctx,
                loc,
                ptr,
                null_val.result(ctx),
                i8_ty,
                Symbol::new("eq"),
            );
            ctx.push_op(current_block, is_null.op_ref());

            // brif: if null → skip, else → do_retain
            let brif_op = clif::brif(ctx, loc, is_null.result(ctx), skip_block, do_retain_block);
            ctx.push_op(current_block, brif_op.op_ref());

            // Generate retain RC ops in do_retain_block
            gen_retain_rc_ops(ctx, loc, do_retain_block, ptr, ptr_ty, i64_ty, i32_ty);

            // Jump from do_retain to skip
            let jump = clif::jump(ctx, loc, [], skip_block);
            ctx.push_op(do_retain_block, jump.op_ref());

            // Move remaining ops (after retain) to skip block
            for &remaining_op in &ops[i + 1..] {
                ctx.detach_op(remaining_op);
                ctx.push_op(skip_block, remaining_op);
            }

            // Erase the retain op
            erase_op(ctx, op);

            // RAUW: retain result → original ptr
            ctx.replace_all_uses(retain_result, ptr);

            // Add new blocks to region (after current_block)
            let region_blocks = &mut ctx.region_mut(region).blocks;
            let pos = region_blocks
                .iter()
                .position(|&b| b == current_block)
                .unwrap();
            region_blocks.insert(pos + 1, do_retain_block);
            region_blocks.insert(pos + 2, skip_block);
            ctx.block_mut(do_retain_block).parent_region = Some(region);
            ctx.block_mut(skip_block).parent_region = Some(region);

            // Now recursively process the skip block for remaining RC ops
            lower_rc_in_block(ctx, region, skip_block);
            return;
        } else if let Ok(release_op) = tribute_rt::Release::from_op(ctx, op) {
            let ptr = ctx.op_operands(op).to_vec()[0];
            let alloc_size = release_op.alloc_size(ctx);

            // Create skip block (continuation)
            let skip_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });

            // Create do_release block
            let do_release_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });

            // Create free block
            let free_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });

            // Generate null check ops in current_block
            let null_val = clif::iconst(ctx, loc, ptr_ty, 0);
            ctx.push_op(current_block, null_val.op_ref());
            let is_null = clif::icmp(
                ctx,
                loc,
                ptr,
                null_val.result(ctx),
                i8_ty,
                Symbol::new("eq"),
            );
            ctx.push_op(current_block, is_null.op_ref());

            // brif: if null → skip, else → do_release
            let brif_op = clif::brif(ctx, loc, is_null.result(ctx), skip_block, do_release_block);
            ctx.push_op(current_block, brif_op.op_ref());

            // Generate release decrement ops in do_release_block
            let is_zero_val = gen_release_decrement(
                ctx,
                loc,
                do_release_block,
                ptr,
                ptr_ty,
                i64_ty,
                i32_ty,
                i8_ty,
            );

            // brif: if zero → free, else → skip
            let zero_brif = clif::brif(ctx, loc, is_zero_val, free_block, skip_block);
            ctx.push_op(do_release_block, zero_brif.op_ref());

            // Generate free block ops
            gen_deep_release_call(
                ctx, loc, free_block, ptr, alloc_size, skip_block, i64_ty, nil_ty,
            );

            // Move remaining ops (after release) to skip block
            for &remaining_op in &ops[i + 1..] {
                ctx.detach_op(remaining_op);
                ctx.push_op(skip_block, remaining_op);
            }

            // Erase the release op
            erase_op(ctx, op);

            // Add new blocks to region (after current_block)
            let region_blocks = &mut ctx.region_mut(region).blocks;
            let pos = region_blocks
                .iter()
                .position(|&b| b == current_block)
                .unwrap();
            region_blocks.insert(pos + 1, do_release_block);
            region_blocks.insert(pos + 2, free_block);
            region_blocks.insert(pos + 3, skip_block);
            ctx.block_mut(do_release_block).parent_region = Some(region);
            ctx.block_mut(free_block).parent_region = Some(region);
            ctx.block_mut(skip_block).parent_region = Some(region);

            // Recursively process skip block for remaining RC ops
            lower_rc_in_block(ctx, region, skip_block);
            return;
        }

        i += 1;
    }
}

/// Intern a type in the arena context.
fn intern_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    use trunk_ir::arena::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new(dialect), Symbol::new(name)).build())
}

/// Generate retain RC ops (load, increment, store) in a block.
fn gen_retain_rc_ops(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::Location,
    block: BlockRef,
    ptr: ValueRef,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) {
    // rc_addr = ptr - RC_HEADER_SIZE
    let hdr_sz = clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
    ctx.push_op(block, hdr_sz.op_ref());
    let rc_addr = clif::isub(ctx, loc, ptr, hdr_sz.result(ctx), ptr_ty);
    ctx.push_op(block, rc_addr.op_ref());

    // rc = load(rc_addr, offset=0)
    let rc = clif::load(ctx, loc, rc_addr.result(ctx), i32_ty, 0);
    ctx.push_op(block, rc.op_ref());

    // new_rc = rc + 1
    let one = clif::iconst(ctx, loc, i32_ty, 1);
    ctx.push_op(block, one.op_ref());
    let new_rc = clif::iadd(ctx, loc, rc.result(ctx), one.result(ctx), i32_ty);
    ctx.push_op(block, new_rc.op_ref());

    // store(new_rc, rc_addr, offset=0)
    let store = clif::store(ctx, loc, new_rc.result(ctx), rc_addr.result(ctx), 0);
    ctx.push_op(block, store.op_ref());
}

/// Generate release decrement ops in a block.
/// Returns the `is_zero` value.
#[allow(clippy::too_many_arguments)]
fn gen_release_decrement(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::Location,
    block: BlockRef,
    ptr: ValueRef,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
    i8_ty: TypeRef,
) -> ValueRef {
    // rc_addr = ptr - RC_HEADER_SIZE
    let hdr_sz = clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
    ctx.push_op(block, hdr_sz.op_ref());
    let rc_addr = clif::isub(ctx, loc, ptr, hdr_sz.result(ctx), ptr_ty);
    ctx.push_op(block, rc_addr.op_ref());

    // rc = load(rc_addr, offset=0)
    let rc = clif::load(ctx, loc, rc_addr.result(ctx), i32_ty, 0);
    ctx.push_op(block, rc.op_ref());

    // new_rc = rc - 1
    let one = clif::iconst(ctx, loc, i32_ty, 1);
    ctx.push_op(block, one.op_ref());
    let new_rc = clif::isub(ctx, loc, rc.result(ctx), one.result(ctx), i32_ty);
    ctx.push_op(block, new_rc.op_ref());

    // store(new_rc, rc_addr, offset=0)
    let store = clif::store(ctx, loc, new_rc.result(ctx), rc_addr.result(ctx), 0);
    ctx.push_op(block, store.op_ref());

    // is_zero = icmp(new_rc, 0, eq)
    let zero = clif::iconst(ctx, loc, i32_ty, 0);
    ctx.push_op(block, zero.op_ref());
    let is_zero = clif::icmp(
        ctx,
        loc,
        new_rc.result(ctx),
        zero.result(ctx),
        i8_ty,
        Symbol::new("eq"),
    );
    ctx.push_op(block, is_zero.op_ref());

    is_zero.result(ctx)
}

/// Generate deep release call + jump in the free block.
#[allow(clippy::too_many_arguments)]
fn gen_deep_release_call(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::Location,
    block: BlockRef,
    payload_ptr: ValueRef,
    alloc_size: u64,
    continue_block: BlockRef,
    i64_ty: TypeRef,
    nil_ty: TypeRef,
) {
    // size = iconst(alloc_size)
    let size = clif::iconst(ctx, loc, i64_ty, alloc_size as i64);
    ctx.push_op(block, size.op_ref());

    // call @__tribute_deep_release(payload_ptr, size)
    let call = clif::call(
        ctx,
        loc,
        [payload_ptr, size.result(ctx)],
        nil_ty,
        Symbol::new(DEEP_RELEASE_FN),
    );
    ctx.push_op(block, call.op_ref());

    // jump to continue block
    let jump = clif::jump(ctx, loc, [], continue_block);
    ctx.push_op(block, jump.op_ref());
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::arena::context::IrContext;
    use trunk_ir::arena::parser::parse_test_module;
    use trunk_ir::arena::printer::print_module;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower_rc(&mut ctx, module);
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_snapshot_retain() {
        // retain result is unused (matches real pipeline behavior:
        // insert_rc adds retain for side-effect only)
        let result = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.ptr {
    %1 = tribute_rt.retain %0 : core.ptr
    clif.return %0
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_snapshot_release() {
        let result = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_snapshot_retain_and_release() {
        let result = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
    %1 = tribute_rt.retain %0 : core.ptr
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_chained_retains() {
        // Two independent retains on same pointer (retain result unused)
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.ptr {
    %1 = tribute_rt.retain %0 : core.ptr
    %2 = tribute_rt.retain %0 : core.ptr
    clif.return %0
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt."),
            "no tribute_rt ops should remain after lowering"
        );
        // Each retain produces one iadd (rc + 1) in its do_retain block
        let iadd_count = output.matches("clif.iadd").count();
        assert_eq!(iadd_count, 2, "expected 2 RC increments, got {iadd_count}");
    }

    #[test]
    fn test_multiple_functions() {
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.ptr {
    %1 = tribute_rt.retain %0 : core.ptr
    clif.return %0
  }
  clif.func @g(%0: core.ptr) -> core.nil {
    tribute_rt.release %0 {alloc_size = 16}
    clif.return
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt."),
            "no tribute_rt ops should remain"
        );
    }

    #[test]
    fn test_no_rc_ops_noop() {
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.i32) -> core.i32 {
    clif.return %0
  }
}"#,
        );
        // No RC ops → no null checks or block splits
        assert!(
            !output.contains("clif.icmp"),
            "no null checks should be inserted without RC ops"
        );
        assert!(
            !output.contains("clif.brif"),
            "no branch splits should occur without RC ops"
        );
    }

    #[test]
    fn test_release_successors() {
        let mut ctx = IrContext::new();
        let ir = r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#;
        let module = parse_test_module(&mut ctx, ir);
        lower_rc(&mut ctx, module);

        // After lowering a single release:
        // entry_block (null check + brif) → do_release_block → free_block → skip_block
        let func_ops = module.ops(&ctx);
        let func_op = func_ops[0];
        let regions = ctx.op(func_op).regions.to_vec();
        let body = regions[0];
        let block_count = ctx.region(body).blocks.len();
        assert_eq!(
            block_count, 4,
            "expected 4 blocks after release lowering, got {block_count}"
        );
    }

    #[test]
    fn test_retain_produces_null_guard() {
        // Verify retain lowering produces the expected null-guard structure:
        // iconst(0), icmp eq, brif to skip/do_retain
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.ptr {
    %1 = tribute_rt.retain %0 : core.ptr
    clif.return %0
  }
}"#,
        );
        assert!(
            output.contains("clif.icmp"),
            "null check comparison should be present"
        );
        assert!(
            output.contains("clif.brif"),
            "conditional branch should be present"
        );
        assert!(
            output.contains("clif.iadd"),
            "RC increment should be present"
        );
    }
}
