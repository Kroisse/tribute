//! RC lowering pass: lower `tribute_rt.retain`/`release` to inline `clif.*` ops.
//!
//! ## Pipeline Position
//!
//! Runs after unrealized cast conversion and reconciliation (Phase 3) and before `emit_module_to_native`
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
//!   %one     = clif.iconst(1) : core.i32
//!   %old_rc  = clif.atomic_rmw(%rc_addr, %one, bin_op="add", offset=0) : core.i32
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
//!   %hdr_sz     = clif.iconst(8) : core.i64
//!   %rc_addr    = clif.isub(ptr, %hdr_sz) : core.ptr
//!   %one        = clif.iconst(1) : core.i32
//!   %old_rc     = clif.atomic_rmw(%rc_addr, %one, bin_op="sub", offset=0) : core.i32
//!   %is_last    = clif.icmp(%old_rc, %one, eq) : core.i8
//!   clif.brif(%is_last, ^free_block, ^skip_block)
//!
//! ^free_block:
//!   clif.call @__tribute_deep_release(ptr, size)
//!   clif.jump(^skip_block)
//!
//! ^skip_block:
//!   // remaining ops
//! ```

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::Module;
use trunk_ir::rewrite::helpers::erase_op;
use trunk_ir::{BlockArgData, BlockData, BlockRef, OpRef, RegionData, RegionRef, TypeRef, ValueRef};

use tribute_ir::dialect::tribute_rt;

use super::rtti::{DEALLOC_FN, DEEP_RELEASE_FN};
use trunk_ir::SymbolPath;
use trunk_ir::dialect::core;

/// Lower all `tribute_rt.retain` and `tribute_rt.release` operations to
/// inline `clif.*` operations.
fn lower_rc(ctx: &mut IrContext, module: Module) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    outline_moving_releases(ctx, first_block);
    let module_ops = ctx.block(first_block).ops.clone();

    for op in module_ops {
        if let Ok(_func_op) = clif::Func::from_op(ctx, op) {
            let regions = ctx.op_regions(op).collect::<trunk_ir::RegionList>();
            for region in regions {
                lower_rc_in_region(ctx, region);
            }
        }
    }
}

/// Name prefix for the functions that hold a moving release.
const RELEASE_MOVING_FN_PREFIX: &str = "__tribute_release_moving_";

/// Replace each `tribute_rt.release_moving` with a call to a function that
/// performs it, one per allocation size, released offsets and number of
/// moved values. Expanding both of its paths at every release would grow the
/// code each owner's death compiles to.
fn outline_moving_releases(ctx: &mut IrContext, module_block: BlockRef) {
    let mut releases = Vec::<OpRef>::new();
    for &op in &ctx.block(module_block).ops {
        if clif::Func::matches(ctx, op) {
            for region in ctx.op_regions(op) {
                for &block in &ctx.region(region).blocks {
                    releases.extend(
                        ctx.block(block)
                            .ops
                            .iter()
                            .copied()
                            .filter(|&op| tribute_rt::ReleaseMoving::matches(ctx, op)),
                    );
                }
            }
        }
    }
    if releases.is_empty() {
        return;
    }
    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
    let mut functions = HashMap::<(u64, Vec<i32>, usize), SymbolPath>::default();
    for op in releases {
        let release = tribute_rt::ReleaseMoving::from_op(ctx, op)
            .expect("collected moving release");
        let alloc_size = release.alloc_size(ctx);
        let release_offsets = release.release_offsets(ctx).collect::<Vec<_>>();
        let operands = std::iter::once(release.ptr(ctx))
            .chain(release.moved(ctx).iter().copied())
            .collect::<Vec<_>>();
        let key = (alloc_size, release_offsets, operands.len() - 1);
        let callee = match functions.get(&key) {
            Some(callee) => callee.clone(),
            None => {
                let callee = SymbolPath::new([format!(
                    "{RELEASE_MOVING_FN_PREFIX}{}",
                    functions.len()
                )
                .as_str()]);
                let location = ctx.op(op).location;
                let function =
                    moving_release_function(ctx, &callee, &key, location, ptr_ty, nil_ty);
                ctx.push_op(module_block, function);
                functions.insert(key, callee.clone());
                callee
            }
        };
        let location = ctx.op(op).location;
        let block = ctx.op(op).parent_block.expect("moving release in a block");
        let call = clif::Call::operands(operands)
            .callee(callee)
            .results([nil_ty])
            .build(ctx, location);
        ctx.insert_op_before(block, op, call.op_ref());
        erase_op(ctx, op);
    }
}

/// `fn(owner, moved...)` performing one moving release of `key`'s shape.
fn moving_release_function(
    ctx: &mut IrContext,
    name: &SymbolPath,
    (alloc_size, release_offsets, moved): &(u64, Vec<i32>, usize),
    location: trunk_ir::Location,
    ptr_ty: TypeRef,
    nil_ty: TypeRef,
) -> OpRef {
    let inputs = vec![ptr_ty; moved + 1];
    let entry = ctx.create_block(BlockData {
        location,
        args: inputs
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    });
    let args = ctx.block_args(entry).to_vec();
    let release = tribute_rt::ReleaseMoving::operands(args[0], args[1..].to_vec())
        .alloc_size(*alloc_size)
        .release_offsets(release_offsets.iter().copied())
        .build(ctx, location);
    ctx.push_op(entry, release.op_ref());
    let ret = clif::Return::operands([]).build(ctx, location);
    ctx.push_op(entry, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry],
        parent_op: None,
    });
    let func_ty = clif::func_sig(ctx, inputs, [nil_ty]).as_type_ref();
    clif::Func::operands()
        .sym_name(name.leaf().expect("generated function name").to_string())
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location)
        .op_ref()
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
        let ops = ctx.block(block).ops.clone();
        for op in ops {
            let nested = ctx.op_regions(op).collect::<trunk_ir::RegionList>();
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
        let ops = ctx.block(*block).ops.clone();
        for op in ops {
            let nested = ctx.op_regions(op).collect::<trunk_ir::RegionList>();
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
    let ops = ctx.block(block).ops.clone();
    let loc = ctx.block(block).location;

    // Quick check: any RC ops?
    let has_rc_ops = ops.iter().any(|&op| {
        tribute_rt::Retain::from_op(ctx, op).is_ok()
            || tribute_rt::Release::from_op(ctx, op).is_ok()
            || tribute_rt::ReleaseMoving::matches(ctx, op)
    });

    if !has_rc_ops {
        return;
    }

    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let i64_ty = core::I64::type_ref(ctx);
    let i32_ty = core::I32::type_ref(ctx);
    let i8_ty = core::I8::type_ref(ctx);
    let nil_ty = core::Nil::params().intern(ctx).as_type_ref();

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
            let null_val = clif::Iconst::operands()
                .value(0)
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(current_block, null_val.op_ref());
            let is_null = clif::Icmp::operands(ptr, null_val.result(ctx))
                .cond("eq")
                .results(i8_ty)
                .build(ctx, loc);
            ctx.push_op(current_block, is_null.op_ref());

            // brif: if null → skip, else → do_retain
            let brif_op = clif::Brif::operands(is_null.result(ctx))
                .successors(skip_block, do_retain_block)
                .build(ctx, loc);
            ctx.push_op(current_block, brif_op.op_ref());

            // Generate retain RC ops in do_retain_block
            gen_retain_rc_ops(ctx, loc, do_retain_block, ptr, ptr_ty, i64_ty, i32_ty);

            // Jump from do_retain to skip
            let jump = clif::Jump::operands([])
                .successors(skip_block)
                .build(ctx, loc);
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
            let null_val = clif::Iconst::operands()
                .value(0)
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(current_block, null_val.op_ref());
            let is_null = clif::Icmp::operands(ptr, null_val.result(ctx))
                .cond("eq")
                .results(i8_ty)
                .build(ctx, loc);
            ctx.push_op(current_block, is_null.op_ref());

            // brif: if null → skip, else → do_release
            let brif_op = clif::Brif::operands(is_null.result(ctx))
                .successors(skip_block, do_release_block)
                .build(ctx, loc);
            ctx.push_op(current_block, brif_op.op_ref());

            // Generate release decrement ops in do_release_block
            let is_last_val = gen_release_decrement(
                ctx,
                loc,
                do_release_block,
                ptr,
                ptr_ty,
                i64_ty,
                i32_ty,
                i8_ty,
            );

            // brif: if last ref → free, else → skip
            let last_brif = clif::Brif::operands(is_last_val)
                .successors(free_block, skip_block)
                .build(ctx, loc);
            ctx.push_op(do_release_block, last_brif.op_ref());

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
        } else if let Ok(release) = tribute_rt::ReleaseMoving::from_op(ctx, op) {
            let ptr = release.ptr(ctx);
            let moved = release.moved(ctx).to_vec();
            let alloc_size = release.alloc_size(ctx);
            let release_offsets = release.release_offsets(ctx).collect::<Vec<_>>();
            let new_block = |ctx: &mut IrContext| {
                ctx.create_block(BlockData {
                    location: loc,
                    args: vec![],
                    ops: Default::default(),
                    parent_region: None,
                })
            };
            let check_block = new_block(ctx);
            let unique_block = new_block(ctx);
            let shared_block = new_block(ctx);
            let skip_block = new_block(ctx);

            let null_val = clif::Iconst::operands()
                .value(0)
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(current_block, null_val.op_ref());
            let is_null = clif::Icmp::operands(ptr, null_val.result(ctx))
                .cond("eq")
                .results(i8_ty)
                .build(ctx, loc);
            ctx.push_op(current_block, is_null.op_ref());
            let brif_op = clif::Brif::operands(is_null.result(ctx))
                .successors(skip_block, check_block)
                .build(ctx, loc);
            ctx.push_op(current_block, brif_op.op_ref());

            // A count of one is the caller's own unit, which no other holder
            // can change, so a plain load decides uniqueness.
            let hdr_sz = clif::Iconst::operands()
                .value(RC_HEADER_SIZE as i64)
                .results(i64_ty)
                .build(ctx, loc);
            ctx.push_op(check_block, hdr_sz.op_ref());
            let raw_ptr = clif::Isub::operands(ptr, hdr_sz.result(ctx))
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(check_block, raw_ptr.op_ref());
            let rc = clif::Load::operands(raw_ptr.result(ctx))
                .offset(0)
                .results(i32_ty)
                .build(ctx, loc);
            ctx.push_op(check_block, rc.op_ref());
            let one = clif::Iconst::operands()
                .value(1)
                .results(i32_ty)
                .build(ctx, loc);
            ctx.push_op(check_block, one.op_ref());
            let is_unique = clif::Icmp::operands(rc.result(ctx), one.result(ctx))
                .cond("eq")
                .results(i8_ty)
                .build(ctx, loc);
            ctx.push_op(check_block, is_unique.op_ref());
            let unique_brif = clif::Brif::operands(is_unique.result(ctx))
                .successors(unique_block, shared_block)
                .build(ctx, loc);
            ctx.push_op(check_block, unique_brif.op_ref());

            // Unique: the moved values keep their fields' units; release the
            // other managed fields and free the shell.
            for offset in release_offsets {
                let field = clif::Load::operands(ptr)
                    .offset(offset)
                    .results(ptr_ty)
                    .build(ctx, loc);
                ctx.push_op(unique_block, field.op_ref());
                let field_release = tribute_rt::Release::operands(field.result(ctx))
                    .alloc_size(0)
                    .build(ctx, loc);
                ctx.push_op(unique_block, field_release.op_ref());
            }
            let size = clif::Iconst::operands()
                .value(alloc_size as i64)
                .results(i64_ty)
                .build(ctx, loc);
            ctx.push_op(unique_block, size.op_ref());
            let dealloc = clif::Call::operands([raw_ptr.result(ctx), size.result(ctx)])
                .callee(SymbolPath::new([DEALLOC_FN]))
                .results([nil_ty])
                .build(ctx, loc);
            ctx.push_op(unique_block, dealloc.op_ref());
            let unique_jump = clif::Jump::operands([])
                .successors(skip_block)
                .build(ctx, loc);
            ctx.push_op(unique_block, unique_jump.op_ref());

            // Shared: acquire a unit for each moved value, then release the
            // owner as usual; another holder may release it concurrently.
            for value in moved {
                let ty = ctx.value_ty(value);
                let retain = tribute_rt::Retain::operands(value)
                    .results(ty)
                    .build(ctx, loc);
                ctx.push_op(shared_block, retain.op_ref());
            }
            let owner_release = tribute_rt::Release::operands(ptr)
                .alloc_size(alloc_size)
                .build(ctx, loc);
            ctx.push_op(shared_block, owner_release.op_ref());
            let shared_jump = clif::Jump::operands([])
                .successors(skip_block)
                .build(ctx, loc);
            ctx.push_op(shared_block, shared_jump.op_ref());

            for &remaining_op in &ops[i + 1..] {
                ctx.detach_op(remaining_op);
                ctx.push_op(skip_block, remaining_op);
            }
            erase_op(ctx, op);

            let region_blocks = &mut ctx.region_mut(region).blocks;
            let pos = region_blocks
                .iter()
                .position(|&b| b == current_block)
                .unwrap();
            for (offset, new) in [check_block, unique_block, shared_block, skip_block]
                .into_iter()
                .enumerate()
            {
                region_blocks.insert(pos + 1 + offset, new);
            }
            for new in [check_block, unique_block, shared_block, skip_block] {
                ctx.block_mut(new).parent_region = Some(region);
            }

            lower_rc_in_block(ctx, region, unique_block);
            lower_rc_in_block(ctx, region, shared_block);
            lower_rc_in_block(ctx, region, skip_block);
            return;
        }

        i += 1;
    }
}

/// Generate retain RC ops (atomic increment) in a block.
fn gen_retain_rc_ops(
    ctx: &mut IrContext,
    loc: trunk_ir::Location,
    block: BlockRef,
    ptr: ValueRef,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) {
    // rc_addr = ptr - RC_HEADER_SIZE
    let hdr_sz = clif::Iconst::operands()
        .value(RC_HEADER_SIZE as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ctx.push_op(block, hdr_sz.op_ref());
    let rc_addr = clif::Isub::operands(ptr, hdr_sz.result(ctx))
        .results(ptr_ty)
        .build(ctx, loc);
    ctx.push_op(block, rc_addr.op_ref());

    // atomic_rmw add: atomically increment RC
    let one = clif::Iconst::operands()
        .value(1)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(block, one.op_ref());
    let _old_rc = clif::AtomicRmw::operands(rc_addr.result(ctx), one.result(ctx))
        .bin_op("add")
        .offset(0)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(block, _old_rc.op_ref());
}

/// Generate release decrement ops in a block.
/// Returns a flag indicating the old refcount was 1 (i.e., now zero after atomic decrement).
#[allow(clippy::too_many_arguments)]
fn gen_release_decrement(
    ctx: &mut IrContext,
    loc: trunk_ir::Location,
    block: BlockRef,
    ptr: ValueRef,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
    i8_ty: TypeRef,
) -> ValueRef {
    // rc_addr = ptr - RC_HEADER_SIZE
    let hdr_sz = clif::Iconst::operands()
        .value(RC_HEADER_SIZE as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ctx.push_op(block, hdr_sz.op_ref());
    let rc_addr = clif::Isub::operands(ptr, hdr_sz.result(ctx))
        .results(ptr_ty)
        .build(ctx, loc);
    ctx.push_op(block, rc_addr.op_ref());

    // atomic_rmw sub: atomically decrement RC, returns old value
    let one = clif::Iconst::operands()
        .value(1)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(block, one.op_ref());
    let old_rc = clif::AtomicRmw::operands(rc_addr.result(ctx), one.result(ctx))
        .bin_op("sub")
        .offset(0)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(block, old_rc.op_ref());

    // is_last = (old_rc == 1) means refcount was 1 before decrement, now 0
    let is_last = clif::Icmp::operands(old_rc.result(ctx), one.result(ctx))
        .cond("eq")
        .results(i8_ty)
        .build(ctx, loc);
    ctx.push_op(block, is_last.op_ref());

    is_last.result(ctx)
}

/// Generate deep release call + jump in the free block.
#[allow(clippy::too_many_arguments)]
fn gen_deep_release_call(
    ctx: &mut IrContext,
    loc: trunk_ir::Location,
    block: BlockRef,
    payload_ptr: ValueRef,
    alloc_size: u64,
    continue_block: BlockRef,
    i64_ty: TypeRef,
    nil_ty: TypeRef,
) {
    // size = iconst(alloc_size)
    let size = clif::Iconst::operands()
        .value(alloc_size as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ctx.push_op(block, size.op_ref());

    // call @__tribute_deep_release(payload_ptr, size)
    let call = clif::Call::operands([payload_ptr, size.result(ctx)])
        .callee(SymbolPath::new([DEEP_RELEASE_FN]))
        .results([nil_ty])
        .build(ctx, loc);
    ctx.push_op(block, call.op_ref());

    // jump to continue block
    let jump = clif::Jump::operands([])
        .successors(continue_block)
        .build(ctx, loc);
    ctx.push_op(block, jump.op_ref());
}

/// Pass form of [`lower_rc`].
pub struct RcLowering;

impl trunk_ir::pass::Pass for RcLowering {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "rc-lowering"
    }

    fn run(
        &mut self,
        ctx: &mut trunk_ir::context::IrContext,
        target: trunk_ir::dialect::core::Module,
        _analyses: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        lower_rc(ctx, target.into());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower_rc(&mut ctx, module);
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_snapshot_retain() {
        // The materializer adds retains for their side effect; their result is unused.
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
    fn test_snapshot_release_moving() {
        // The unique path releases the unmoved field at offset 8 and frees
        // the owner; the shared path acquires the moved value and releases
        // the owner.
        let result = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.ptr {
    %1 = clif.load %0 {offset = 0} : core.ptr
    tribute_rt.release_moving %0, %1 {alloc_size = 24, release_offsets = [8]}
    clif.return %1
  }
}"#,
        );
        assert!(!result.contains("tribute_rt."), "{result}");
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
        // Each retain produces one atomic_rmw in its do_retain block
        let atomic_count = output.matches("clif.atomic_rmw").count();
        assert_eq!(
            atomic_count, 2,
            "expected 2 atomic RC increments, got {atomic_count}"
        );
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
        let body = ctx.op_region(func_op, 0).unwrap();
        let block_count = ctx.region(body).blocks.len();
        assert_eq!(
            block_count, 4,
            "expected 4 blocks after release lowering, got {block_count}"
        );
    }
}
