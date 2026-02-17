//! RC lowering pass: lower `tribute_rt.retain`/`release` to inline `clif.*` ops.
//!
//! ## Pipeline Position
//!
//! Runs after `resolve_unrealized_casts` (Phase 3) and before `emit_module_to_native`
//! (Phase 4). At this point all types are resolved, and after lowering no
//! `tribute_rt.*` ops should remain.
//!
//! ## Retain Lowering (inline, no block split)
//!
//! ```text
//! %hdr_sz   = clif.iconst(8) : core.i64
//! %rc_addr  = clif.isub(ptr, %hdr_sz) : core.ptr
//! %rc       = clif.load(%rc_addr, offset=0) : core.i32
//! %one      = clif.iconst(1) : core.i32
//! %new_rc   = clif.iadd(%rc, %one) : core.i32
//! clif.store(%new_rc, %rc_addr, offset=0)
//! %zero_ptr = clif.iconst(0) : core.ptr
//! %result   = clif.iadd(ptr, %zero_ptr) : core.ptr   ← maps to retain result
//! ```
//!
//! ## Release Lowering (block split)
//!
//! ```text
//! // --- current block tail ---
//! %hdr_sz   = clif.iconst(8) : core.i64
//! %rc_addr  = clif.isub(ptr, %hdr_sz) : core.ptr
//! %rc       = clif.load(%rc_addr, offset=0) : core.i32
//! %one      = clif.iconst(1) : core.i32
//! %new_rc   = clif.isub(%rc, %one) : core.i32
//! clif.store(%new_rc, %rc_addr, offset=0)
//! %zero     = clif.iconst(0) : core.i32
//! %is_zero  = clif.icmp(%new_rc, %zero, cond="eq") : core.i8
//! clif.brif(%is_zero, ^free_block, ^continue_block)
//!
//! ^free_block:
//!   %size = clif.iconst(<alloc_size>) : core.i64
//!   clif.call(ptr, %size, callee=@__tribute_deep_release)
//!   clif.jump(^continue_block)
//!
//! ^continue_block:
//!   // remaining ops from original block
//! ```

use std::collections::HashMap;

use tribute_ir::dialect::tribute_rt::{self, RC_HEADER_SIZE};
use trunk_ir::dialect::{clif, core};
use trunk_ir::{Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Value};

/// Name of the deep release dispatch function.
const DEEP_RELEASE_FN: &str = "__tribute_deep_release";

/// Lower all `tribute_rt.retain` and `tribute_rt.release` operations to
/// inline `clif.*` operations.
#[salsa::tracked]
pub fn lower_rc<'db>(db: &'db dyn salsa::Database, module: core::Module<'db>) -> core::Module<'db> {
    let body = module.body(db);
    let mut module_changed = false;

    let new_blocks: IdVec<Block<'db>> = body
        .blocks(db)
        .iter()
        .map(|top_op_block| {
            let mut block_changed = false;
            let new_ops: IdVec<Operation<'db>> = top_op_block
                .operations(db)
                .iter()
                .map(|op| {
                    if let Ok(func_op) = clif::Func::from_operation(db, *op) {
                        let func_body = func_op.body(db);
                        let new_body = lower_rc_in_region(db, &func_body);
                        if new_body != func_body {
                            block_changed = true;
                            op.modify(db).regions(IdVec::from(vec![new_body])).build()
                        } else {
                            *op
                        }
                    } else {
                        *op
                    }
                })
                .collect();

            if block_changed {
                module_changed = true;
                Block::new(
                    db,
                    top_op_block.id(db),
                    top_op_block.location(db),
                    top_op_block.args(db).clone(),
                    new_ops,
                )
            } else {
                *top_op_block
            }
        })
        .collect();

    if !module_changed {
        return module;
    }

    let new_body = Region::new(db, body.location(db), new_blocks);
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Recursively lower RC ops in a region.
///
/// Processes all blocks in the region (lowering retain/release ops and splitting
/// blocks for release), then recursively descends into nested regions within
/// operations (e.g., `scf.if`, `cont.push_prompt`).
fn lower_rc_in_region<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> Region<'db> {
    let mut all_new_blocks: Vec<Block<'db>> = Vec::new();
    let mut changed = false;

    for block in region.blocks(db).iter() {
        // Step 1: Lower retain/release ops (may split block)
        let lowered_blocks = lower_rc_in_block(db, block);
        if lowered_blocks.len() != 1 || lowered_blocks[0] != *block {
            changed = true;
        }

        // Step 2: Recursively process nested regions in each resulting block
        for b in lowered_blocks {
            let new_block = lower_nested_regions(db, &b, &mut changed);
            all_new_blocks.push(new_block);
        }
    }

    if !changed {
        return *region;
    }

    Region::new(
        db,
        region.location(db),
        all_new_blocks.into_iter().collect(),
    )
}

/// Recursively lower RC ops in nested regions within a block's operations.
fn lower_nested_regions<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    changed: &mut bool,
) -> Block<'db> {
    let mut block_changed = false;
    let new_ops: IdVec<Operation<'db>> = block
        .operations(db)
        .iter()
        .map(|op| {
            let regions = op.regions(db);
            if regions.is_empty() {
                return *op;
            }

            let mut any_region_changed = false;
            let new_regions: IdVec<Region<'db>> = regions
                .iter()
                .map(|r| {
                    let new_r = lower_rc_in_region(db, r);
                    if new_r != *r {
                        any_region_changed = true;
                    }
                    new_r
                })
                .collect();

            if any_region_changed {
                block_changed = true;
                op.modify(db).regions(new_regions).build()
            } else {
                *op
            }
        })
        .collect();

    if block_changed {
        *changed = true;
        Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            new_ops,
        )
    } else {
        *block
    }
}

/// Info collected for each release op during Phase 1.
struct ReleaseInfo<'db> {
    is_zero_val: Value<'db>,
    payload_ptr_val: Value<'db>,
    alloc_size: u64,
    free_block_id: BlockId,
}

/// A segment of operations between release points.
struct Segment<'db> {
    ops: Vec<Operation<'db>>,
    block_id: BlockId,
    block_args: IdVec<trunk_ir::BlockArg<'db>>,
    /// If this segment ends with a release, stores info for the block split.
    release: Option<ReleaseInfo<'db>>,
}

/// Lower RC ops in a single block, potentially splitting it into multiple blocks.
///
/// Returns a Vec of blocks. If the block contains no release ops, returns the
/// original block unchanged. Each release op causes a block split.
///
/// Uses a two-phase algorithm to ensure `clif.brif` and `clif.jump` operations
/// reference the actual destination blocks (not empty placeholders):
///
/// - **Phase 1**: Iterate ops and collect segments separated by release points.
/// - **Phase 2**: Build blocks from last to first, so each `brif`/`jump` can
///   reference the already-created continue block.
fn lower_rc_in_block<'db>(db: &'db dyn salsa::Database, block: &Block<'db>) -> Vec<Block<'db>> {
    let ops = block.operations(db);
    let location = block.location(db);

    // Quick check: does this block contain any RC ops?
    let has_rc_ops = ops.iter().any(|op| {
        tribute_rt::Retain::from_operation(db, *op).is_ok()
            || tribute_rt::Release::from_operation(db, *op).is_ok()
    });

    if !has_rc_ops {
        return vec![*block];
    }

    // Phase 1: Collect segments separated by release points.
    // Also maintain a value remap so that uses of retain results are rewritten
    // to reference the identity op produced by the inline expansion.
    let mut segments: Vec<Segment<'db>> = Vec::new();
    let mut current_ops: Vec<Operation<'db>> = Vec::new();
    let mut current_id = block.id(db);
    let mut current_args = block.args(db).clone();
    let mut value_remap: HashMap<Value<'db>, Value<'db>> = HashMap::new();

    for op in ops.iter() {
        if let Ok(retain_op) = tribute_rt::Retain::from_operation(db, *op) {
            // Retain: inline refcount increment, no block split
            let ptr = remap_value(&value_remap, retain_op.ptr(db));
            let inline_ops = gen_retain_inline(db, location, ptr);

            // The last inline op's result replaces the original retain result
            let identity_result = inline_ops.last().unwrap().result(db, 0);
            let original_result = op.result(db, 0);
            value_remap.insert(original_result, identity_result);

            current_ops.extend(inline_ops);
        } else if let Ok(release_op) = tribute_rt::Release::from_operation(db, *op) {
            // Release: refcount decrement + conditional free + block split
            let ptr = remap_value(&value_remap, release_op.ptr(db));
            let alloc_size = release_op.alloc_size(db);

            let continue_block_id = BlockId::fresh();
            let free_block_id = BlockId::fresh();

            // Generate decrement ops (brif will be appended in Phase 2)
            let (decrement_ops, is_zero_val, _rc_addr_val) =
                gen_release_decrement(db, location, ptr);
            current_ops.extend(decrement_ops);

            // Finish this segment
            segments.push(Segment {
                ops: std::mem::take(&mut current_ops),
                block_id: current_id,
                block_args: current_args,
                release: Some(ReleaseInfo {
                    is_zero_val,
                    payload_ptr_val: ptr,
                    alloc_size,
                    free_block_id,
                }),
            });

            // Next segment starts as the continue block
            current_id = continue_block_id;
            current_args = IdVec::new();
        } else {
            // Remap operands for non-RC ops
            current_ops.push(remap_op_operands(db, op, &value_remap));
        }
    }

    // Push the final segment (no release)
    segments.push(Segment {
        ops: current_ops,
        block_id: current_id,
        block_args: current_args,
        release: None,
    });

    // Phase 2: Build blocks from last to first.
    // The last segment has no release, so it's just a plain block.
    // Working backwards, each segment with a release creates:
    //   1. The continue block (already built from previous iteration)
    //   2. A free block (dealloc + jump to continue block)
    //   3. The current block (segment ops + brif to free/continue)
    let mut result_blocks: Vec<Block<'db>> = Vec::new();

    // Build the last segment first
    let last = segments.pop().expect("at least one segment");
    let mut continue_block = Block::new(
        db,
        last.block_id,
        location,
        last.block_args,
        last.ops.into_iter().collect(),
    );

    // Process remaining segments in reverse
    for segment in segments.into_iter().rev() {
        let release = segment.release.expect("non-last segment must have release");

        // Build free block: deep_release + jump to continue_block
        let free_ops = gen_deep_release_call(
            db,
            location,
            release.payload_ptr_val,
            release.alloc_size,
            continue_block,
        );
        let free_block = Block::new(
            db,
            release.free_block_id,
            location,
            IdVec::new(),
            free_ops.into_iter().collect(),
        );

        // Append brif to segment ops
        let mut ops = segment.ops;
        let brif_op = clif::brif(
            db,
            location,
            release.is_zero_val,
            free_block,
            continue_block,
        );
        ops.push(brif_op.as_operation());

        // Build current block
        let current_block = Block::new(
            db,
            segment.block_id,
            location,
            segment.block_args,
            ops.into_iter().collect(),
        );

        // Prepend free_block and continue_block (they come after current in forward order)
        result_blocks.push(continue_block);
        result_blocks.push(free_block);
        continue_block = current_block;
    }

    // The first block (entry or first segment)
    result_blocks.push(continue_block);

    // Reverse to get forward order
    result_blocks.reverse();

    result_blocks
}

/// Look up a value in the remap, returning the remapped value or the original.
fn remap_value<'db>(remap: &HashMap<Value<'db>, Value<'db>>, value: Value<'db>) -> Value<'db> {
    remap.get(&value).copied().unwrap_or(value)
}

/// Remap all operands of an operation using the value remap.
/// Returns a new operation if any operand changed, or the original otherwise.
fn remap_op_operands<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    remap: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    if remap.is_empty() {
        return *op;
    }

    let operands = op.operands(db);
    let mut new_operands: IdVec<Value<'db>> = IdVec::new();
    let mut changed = false;

    for &operand in operands.iter() {
        let mapped = remap_value(remap, operand);
        new_operands.push(mapped);
        if mapped != operand {
            changed = true;
        }
    }

    if !changed {
        return *op;
    }

    op.modify(db).operands(new_operands).build()
}

/// Generate inline retain ops: load RC, increment, store back, identity result.
fn gen_retain_inline<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    ptr: Value<'db>,
) -> Vec<Operation<'db>> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();

    let mut ops = Vec::new();

    // rc_addr = ptr - RC_HEADER_SIZE
    let hdr_sz = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_sz.as_operation());
    let rc_addr = clif::isub(db, location, ptr, hdr_sz.result(db), ptr_ty);
    ops.push(rc_addr.as_operation());

    // rc = load(rc_addr, offset=0)
    let rc = clif::load(db, location, rc_addr.result(db), i32_ty, 0);
    ops.push(rc.as_operation());

    // new_rc = rc + 1
    let one = clif::iconst(db, location, i32_ty, 1);
    ops.push(one.as_operation());
    let new_rc = clif::iadd(db, location, rc.result(db), one.result(db), i32_ty);
    ops.push(new_rc.as_operation());

    // store(new_rc, rc_addr, offset=0)
    let store = clif::store(db, location, new_rc.result(db), rc_addr.result(db), 0);
    ops.push(store.as_operation());

    // result = ptr (identity via iadd + iconst 0)
    let zero_ptr = clif::iconst(db, location, ptr_ty, 0);
    ops.push(zero_ptr.as_operation());
    let result = clif::iadd(db, location, ptr, zero_ptr.result(db), ptr_ty);
    ops.push(result.as_operation());

    ops
}

/// Generate release decrement ops.
///
/// Returns `(ops, is_zero_val, rc_addr_val)` where `is_zero_val` is the
/// comparison result and `rc_addr_val` is the address of the refcount field
/// (also the raw_ptr for deallocation).
fn gen_release_decrement<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    ptr: Value<'db>,
) -> (Vec<Operation<'db>>, Value<'db>, Value<'db>) {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();
    let i8_ty = core::I8::new(db).as_type();

    let mut ops = Vec::new();

    // rc_addr = ptr - RC_HEADER_SIZE
    let hdr_sz = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_sz.as_operation());
    let rc_addr = clif::isub(db, location, ptr, hdr_sz.result(db), ptr_ty);
    ops.push(rc_addr.as_operation());
    let rc_addr_val = rc_addr.result(db);

    // rc = load(rc_addr, offset=0)
    let rc = clif::load(db, location, rc_addr_val, i32_ty, 0);
    ops.push(rc.as_operation());

    // new_rc = rc - 1
    let one = clif::iconst(db, location, i32_ty, 1);
    ops.push(one.as_operation());
    let new_rc = clif::isub(db, location, rc.result(db), one.result(db), i32_ty);
    ops.push(new_rc.as_operation());

    // store(new_rc, rc_addr, offset=0)
    let store = clif::store(db, location, new_rc.result(db), rc_addr_val, 0);
    ops.push(store.as_operation());

    // is_zero = icmp(new_rc, 0, eq)
    let zero = clif::iconst(db, location, i32_ty, 0);
    ops.push(zero.as_operation());
    let is_zero = clif::icmp(
        db,
        location,
        new_rc.result(db),
        zero.result(db),
        i8_ty,
        Symbol::new("eq"),
    );
    ops.push(is_zero.as_operation());

    (ops, is_zero.result(db), rc_addr_val)
}

/// Generate deep release call ops for the free block.
///
/// Returns ops: `clif.iconst(alloc_size) + clif.call @__tribute_deep_release(payload_ptr, size) + clif.jump(continue)`.
fn gen_deep_release_call<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    payload_ptr: Value<'db>,
    alloc_size: u64,
    continue_block: Block<'db>,
) -> Vec<Operation<'db>> {
    let i64_ty = core::I64::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();

    let mut ops = Vec::new();

    // size = iconst(alloc_size)
    let size = clif::iconst(db, location, i64_ty, alloc_size as i64);
    ops.push(size.as_operation());

    // call @__tribute_deep_release(payload_ptr, size)
    let call = clif::call(
        db,
        location,
        [payload_ptr, size.result(db)],
        nil_ty,
        Symbol::new(DEEP_RELEASE_FN),
    );
    ops.push(call.as_operation());

    // jump to continue block
    let jump = clif::jump(db, location, [], continue_block);
    ops.push(jump.as_operation());

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_op;

    #[salsa::input]
    struct TextInput {
        #[returns(ref)]
        text: String,
    }

    #[salsa::tracked]
    fn do_lower_rc(db: &dyn salsa::Database, input: TextInput) -> core::Module<'_> {
        let module = parse_test_module(db, input.text(db));
        lower_rc(db, module)
    }

    fn run_rc_lowering(db: &salsa::DatabaseImpl, ir: &str) -> String {
        let input = TextInput::new(db, ir.to_string());
        let result = do_lower_rc(db, input);
        print_op(db, result.as_operation())
    }

    #[salsa_test]
    fn test_retain_inline(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.ptr, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %1 = tribute_rt.retain %p : core.ptr
                        clif.return %1
                }
            }
            "#,
        );

        // Should not contain tribute_rt ops
        assert!(
            !ir.contains("tribute_rt."),
            "no tribute_rt ops should remain: {ir}"
        );
        // Should contain inline RC ops
        assert!(ir.contains("clif.load"), "should have load for RC: {ir}");
        assert!(ir.contains("clif.iadd"), "should have iadd for RC: {ir}");
        assert!(ir.contains("clif.store"), "should have store for RC: {ir}");
        assert!(ir.contains("clif.return"), "should have return: {ir}");
    }

    #[salsa_test]
    fn test_release_block_split(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        tribute_rt.release %p {alloc_size = 12}
                        clif.return
                }
            }
            "#,
        );

        // Should not contain tribute_rt ops
        assert!(
            !ir.contains("tribute_rt."),
            "no tribute_rt ops should remain: {ir}"
        );
        // Should have block split (brif, jump, dealloc call)
        assert!(ir.contains("clif.brif"), "should have brif: {ir}");
        assert!(ir.contains("clif.jump"), "should have jump: {ir}");
        assert!(
            ir.contains("__tribute_deep_release"),
            "should have dealloc call: {ir}"
        );
        assert!(ir.contains("clif.isub"), "should have isub for RC: {ir}");
        assert!(ir.contains("clif.icmp"), "should have icmp: {ir}");
    }

    #[salsa_test]
    fn test_retain_and_release_mixed(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %1 = tribute_rt.retain %p : core.ptr
                        tribute_rt.release %1 {alloc_size = 0}
                        clif.return
                }
            }
            "#,
        );

        assert!(
            !ir.contains("tribute_rt."),
            "no tribute_rt ops should remain: {ir}"
        );
        // Both retain (iadd for increment) and release (isub for decrement)
        assert!(ir.contains("clif.isub"), "should have isub: {ir}");
        assert!(ir.contains("clif.brif"), "should have brif: {ir}");
    }

    #[salsa_test]
    fn test_two_releases_two_splits(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr, core.ptr)} {
                    ^entry(%a: core.ptr, %b: core.ptr):
                        tribute_rt.release %a {alloc_size = 12}
                        tribute_rt.release %b {alloc_size = 16}
                        clif.return
                }
            }
            "#,
        );

        assert!(
            !ir.contains("tribute_rt."),
            "no tribute_rt ops should remain: {ir}"
        );
        // Two releases → two block splits → at least 2 brif + 2 jump + 2 dealloc
        let brif_count = ir.matches("clif.brif").count();
        assert_eq!(brif_count, 2, "should have 2 brifs for 2 releases: {ir}");
    }

    #[salsa_test]
    fn test_no_rc_ops_noop(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.i32, core.i32)} {
                    ^entry(%x: core.i32):
                        clif.return %x
                }
            }
            "#,
        );

        // Should be unchanged
        assert!(!ir.contains("clif.brif"), "should not add brif: {ir}");
        assert!(ir.contains("clif.return"), "should have return: {ir}");
    }

    /// Verify that brif/jump successor blocks are the same Block values as the
    /// blocks in the function's region. This ensures the Cranelift backend can
    /// resolve successor references via its block_map (which uses Block identity).
    #[salsa_test]
    fn test_release_successors_match_region_blocks(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        tribute_rt.release %p {alloc_size = 12}
                        clif.return
                }
            }
            "#
            .to_string(),
        );
        let module = do_lower_rc(db, input);

        // Navigate to function body: module → block[0] → op[0] (clif.func) → body region
        let func_op = module.body(db).blocks(db)[0].operations(db)[0];
        let func = clif::Func::from_operation(db, func_op).unwrap();
        let body = func.body(db);
        let region_blocks: Vec<Block> = body.blocks(db).iter().copied().collect();

        // After release lowering: entry block, free block, continue block
        assert_eq!(region_blocks.len(), 3, "release should produce 3 blocks");

        // Find brif in entry block
        let brif_op = region_blocks[0]
            .operations(db)
            .iter()
            .find_map(|op| clif::Brif::from_operation(db, *op).ok())
            .expect("entry block should end with brif");

        // brif successors must be the actual region blocks, not placeholders
        assert_eq!(
            brif_op.then_dest(db),
            region_blocks[1],
            "brif.then_dest must be the free block in the region"
        );
        assert_eq!(
            brif_op.else_dest(db),
            region_blocks[2],
            "brif.else_dest must be the continue block in the region"
        );

        // Find jump in free block
        let jump_op = region_blocks[1]
            .operations(db)
            .iter()
            .find_map(|op| clif::Jump::from_operation(db, *op).ok())
            .expect("free block should end with jump");

        // jump dest must be the continue block in the region
        assert_eq!(
            jump_op.dest(db),
            region_blocks[2],
            "jump.dest must be the continue block in the region"
        );
    }

    /// Verify successor identity with two releases (chained block splits).
    #[salsa_test]
    fn test_two_releases_successors_match_region_blocks(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr, core.ptr)} {
                    ^entry(%a: core.ptr, %b: core.ptr):
                        tribute_rt.release %a {alloc_size = 12}
                        tribute_rt.release %b {alloc_size = 16}
                        clif.return
                }
            }
            "#
            .to_string(),
        );
        let module = do_lower_rc(db, input);

        let func_op = module.body(db).blocks(db)[0].operations(db)[0];
        let func = clif::Func::from_operation(db, func_op).unwrap();
        let body = func.body(db);
        let region_blocks: Vec<Block> = body.blocks(db).iter().copied().collect();

        // Two releases → 5 blocks: entry, free1, continue1, free2, continue2
        assert_eq!(
            region_blocks.len(),
            5,
            "two releases should produce 5 blocks"
        );

        // Verify all brif/jump successors are actual region blocks
        let region_set: std::collections::HashSet<Block> = region_blocks.iter().copied().collect();

        for (i, block) in region_blocks.iter().enumerate() {
            for op in block.operations(db).iter() {
                if let Ok(brif) = clif::Brif::from_operation(db, *op) {
                    assert!(
                        region_set.contains(&brif.then_dest(db)),
                        "block {i}: brif.then_dest not in region"
                    );
                    assert!(
                        region_set.contains(&brif.else_dest(db)),
                        "block {i}: brif.else_dest not in region"
                    );
                }
                if let Ok(jump) = clif::Jump::from_operation(db, *op) {
                    assert!(
                        region_set.contains(&jump.dest(db)),
                        "block {i}: jump.dest not in region"
                    );
                }
            }
        }
    }

    /// Verify that retain result is properly remapped in subsequent uses.
    /// Before the fix, this would produce `clif.return %?` (dangling reference).
    #[salsa_test]
    fn test_retain_result_used_by_subsequent_ops(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.ptr, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %retained = tribute_rt.retain %p : core.ptr
                        clif.return %retained
                }
            }
            "#,
        );

        // The return op must reference the identity result, not a dangling value
        assert!(
            !ir.contains("%?"),
            "should not have dangling value refs: {ir}"
        );
        assert!(!ir.contains("tribute_rt."), "no tribute_rt ops: {ir}");

        // The last value defined before clif.return should be used by return
        // (the identity iadd result from gen_retain_inline)
        assert!(
            ir.contains("clif.return %"),
            "return should reference a valid value: {ir}"
        );
    }

    /// Verify retain remap works when the retained value feeds into release.
    #[salsa_test]
    fn test_retain_result_feeds_into_release(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %r = tribute_rt.retain %p : core.ptr
                        tribute_rt.release %r {alloc_size = 12}
                        clif.return
                }
            }
            "#,
        );

        assert!(
            !ir.contains("%?"),
            "should not have dangling value refs: {ir}"
        );
        assert!(!ir.contains("tribute_rt."), "no tribute_rt ops: {ir}");
        // Should have both retain (iadd for increment) and release (isub + brif)
        assert!(
            ir.contains("clif.isub"),
            "should have release decrement: {ir}"
        );
        assert!(ir.contains("clif.brif"), "should have block split: {ir}");
    }

    /// Verify RC ops inside nested regions (e.g., scf.if) are lowered.
    #[salsa_test]
    fn test_nested_region_rc_ops_lowered(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.ptr, core.ptr, core.i8)} {
                    ^entry(%p: core.ptr, %cond: core.i8):
                        %result = scf.if %cond : core.ptr {
                            %r = tribute_rt.retain %p : core.ptr
                            scf.yield %r
                        } {
                            scf.yield %p
                        }
                        clif.return %result
                }
            }
            "#,
        );

        // tribute_rt.retain inside scf.if should be lowered
        assert!(
            !ir.contains("tribute_rt."),
            "nested retain should be lowered: {ir}"
        );
        // Should contain the inline RC ops from retain lowering
        assert!(ir.contains("clif.load"), "should have load for RC: {ir}");
        assert!(ir.contains("clif.store"), "should have store for RC: {ir}");
        // scf.if structure should be preserved
        assert!(ir.contains("scf.if"), "scf.if should remain: {ir}");
    }

    /// Verify release inside nested regions correctly splits blocks.
    #[salsa_test]
    fn test_nested_region_release_splits_blocks(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr, core.i8)} {
                    ^entry(%p: core.ptr, %cond: core.i8):
                        scf.if %cond : core.nil {
                            tribute_rt.release %p {alloc_size = 12}
                            scf.yield
                        } {
                            scf.yield
                        }
                        clif.return
                }
            }
            "#,
        );

        assert!(
            !ir.contains("tribute_rt."),
            "nested release should be lowered: {ir}"
        );
        // release causes block split → brif + dealloc inside the nested region
        assert!(
            ir.contains("clif.brif"),
            "should have brif from release: {ir}"
        );
        assert!(
            ir.contains("__tribute_deep_release"),
            "should have dealloc call: {ir}"
        );
    }

    /// Verify chained retains: retain(retain(p)) — remap must be transitive.
    #[salsa_test]
    fn test_chained_retains(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.ptr, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %r1 = tribute_rt.retain %p : core.ptr
                        %r2 = tribute_rt.retain %r1 : core.ptr
                        clif.return %r2
                }
            }
            "#,
        );

        assert!(!ir.contains("tribute_rt."), "no tribute_rt ops: {ir}");
        assert!(
            !ir.contains("%?"),
            "should not have dangling value refs: {ir}"
        );
        // Two retains → two load+iadd+store sequences (two RC increments)
        let load_count = ir.matches("clif.load").count();
        assert_eq!(load_count, 2, "should have 2 loads for 2 retains: {ir}");
    }

    /// Verify lowering processes all functions in a module.
    #[salsa_test]
    fn test_multiple_functions(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f1 {type = core.func(core.ptr, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %1 = tribute_rt.retain %p : core.ptr
                        clif.return %1
                }
                clif.func @f2 {type = core.func(core.nil, core.ptr)} {
                    ^entry(%q: core.ptr):
                        tribute_rt.release %q {alloc_size = 8}
                        clif.return
                }
            }
            "#,
        );

        assert!(
            !ir.contains("tribute_rt."),
            "no tribute_rt ops in any function: {ir}"
        );
        // f1 should have retain inline ops
        assert!(ir.contains("clif.load"), "f1 should have load: {ir}");
        // f2 should have release block split
        assert!(ir.contains("clif.brif"), "f2 should have brif: {ir}");
        assert!(
            ir.contains("__tribute_deep_release"),
            "f2 should have dealloc: {ir}"
        );
    }

    /// Release with alloc_size=0 (unknown allocation size).
    #[salsa_test]
    fn test_release_unknown_alloc_size(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        tribute_rt.release %p {alloc_size = 0}
                        clif.return
                }
            }
            "#,
        );

        assert!(
            !ir.contains("tribute_rt."),
            "no tribute_rt ops should remain: {ir}"
        );
        // Should still emit dealloc call with size 0
        assert!(
            ir.contains("__tribute_deep_release"),
            "should still call dealloc: {ir}"
        );
        assert!(ir.contains("clif.brif"), "should have brif: {ir}");
    }

    /// Verify a single retained value used by multiple subsequent ops.
    #[salsa_test]
    fn test_retain_result_multiple_uses(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.ptr, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %r = tribute_rt.retain %p : core.ptr
                        clif.store %r, %r {offset = 0}
                        clif.return %r
                }
            }
            "#,
        );

        assert!(!ir.contains("tribute_rt."), "no tribute_rt ops: {ir}");
        assert!(
            !ir.contains("%?"),
            "should not have dangling value refs: {ir}"
        );
        // store and return should both reference the same remapped value
        assert!(ir.contains("clif.store"), "should have store: {ir}");
        assert!(
            ir.contains("clif.return %"),
            "should have return with value: {ir}"
        );
    }

    /// Deeply nested regions: scf.if inside scf.if.
    #[salsa_test]
    fn test_deeply_nested_regions(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @f {type = core.func(core.ptr, core.ptr, core.i8)} {
                    ^entry(%p: core.ptr, %cond: core.i8):
                        %result = scf.if %cond : core.ptr {
                            %inner = scf.if %cond : core.ptr {
                                %r = tribute_rt.retain %p : core.ptr
                                scf.yield %r
                            } {
                                scf.yield %p
                            }
                            scf.yield %inner
                        } {
                            scf.yield %p
                        }
                        clif.return %result
                }
            }
            "#,
        );

        // retain inside nested scf.if should be lowered
        assert!(
            !ir.contains("tribute_rt."),
            "deeply nested retain should be lowered: {ir}"
        );
        // Both scf.if levels should be preserved
        let if_count = ir.matches("scf.if").count();
        assert_eq!(if_count, 2, "both scf.if should remain: {ir}");
    }

    /// Snapshot: retain + release combined lowering.
    #[salsa_test]
    fn test_snapshot_retain_and_release(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @retain_then_release {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %1 = tribute_rt.retain %p : core.ptr
                        tribute_rt.release %1 {alloc_size = 12}
                        clif.return
                }
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }

    #[salsa_test]
    fn test_snapshot_retain(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @retain_only {type = core.func(core.ptr, core.ptr)} {
                    ^entry(%p: core.ptr):
                        %1 = tribute_rt.retain %p : core.ptr
                        clif.return %1
                }
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }

    #[salsa_test]
    fn test_snapshot_release(db: &salsa::DatabaseImpl) {
        let ir = run_rc_lowering(
            db,
            r#"
            core.module @test {
                clif.func @release_only {type = core.func(core.nil, core.ptr)} {
                    ^entry(%p: core.ptr):
                        tribute_rt.release %p {alloc_size = 12}
                        clif.return
                }
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }
}
