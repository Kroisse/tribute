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
//! %neg8     = clif.iconst(-8) : core.i64
//! %rc_addr  = clif.iadd(ptr, %neg8) : core.ptr
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
//! %neg8     = clif.iconst(-8) : core.i64
//! %rc_addr  = clif.iadd(ptr, %neg8) : core.ptr
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
//!   clif.call(%rc_addr, %size, callee=@__tribute_dealloc)
//!   clif.jump(^continue_block)
//!
//! ^continue_block:
//!   // remaining ops from original block
//! ```

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::{clif, core};
use trunk_ir::{Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Value};

/// Name of the runtime deallocation function.
const DEALLOC_FN: &str = "__tribute_dealloc";

/// Lower all `tribute_rt.retain` and `tribute_rt.release` operations to
/// inline `clif.*` operations.
#[salsa::tracked]
pub fn lower_rc<'db>(db: &'db dyn salsa::Database, module: core::Module<'db>) -> core::Module<'db> {
    let body = module.body(db);
    let mut changed = false;

    let new_blocks: IdVec<Block<'db>> = body
        .blocks(db)
        .iter()
        .map(|top_op_block| {
            let new_ops: IdVec<Operation<'db>> = top_op_block
                .operations(db)
                .iter()
                .map(|op| {
                    if let Ok(func_op) = clif::Func::from_operation(db, *op) {
                        let func_body = func_op.body(db);
                        let new_body = lower_rc_in_function(db, &func_body);
                        if new_body != func_body {
                            changed = true;
                            op.modify(db).regions(IdVec::from(vec![new_body])).build()
                        } else {
                            *op
                        }
                    } else {
                        *op
                    }
                })
                .collect();

            if changed {
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

    if !changed {
        return module;
    }

    let new_body = Region::new(db, body.location(db), new_blocks);
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Lower RC ops in a function body.
fn lower_rc_in_function<'db>(db: &'db dyn salsa::Database, body: &Region<'db>) -> Region<'db> {
    let mut all_new_blocks: Vec<Block<'db>> = Vec::new();
    let mut changed = false;

    for block in body.blocks(db).iter() {
        let result = lower_rc_in_block(db, block);
        if result.len() != 1 || result[0] != *block {
            changed = true;
        }
        all_new_blocks.extend(result);
    }

    if !changed {
        return *body;
    }

    Region::new(db, body.location(db), all_new_blocks.into_iter().collect())
}

/// Lower RC ops in a single block, potentially splitting it into multiple blocks.
///
/// Returns a Vec of blocks. If the block contains no release ops, returns the
/// original block unchanged. Each release op causes a block split.
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

    let mut result_blocks: Vec<Block<'db>> = Vec::new();
    let mut current_ops: Vec<Operation<'db>> = Vec::new();
    let mut current_id = block.id(db);
    let mut current_args = block.args(db).clone();

    for op in ops.iter() {
        if let Ok(retain_op) = tribute_rt::Retain::from_operation(db, *op) {
            // Retain: inline refcount increment, no block split
            let ptr = retain_op.ptr(db);
            let inline_ops = gen_retain_inline(db, location, ptr);
            current_ops.extend(inline_ops);
        } else if let Ok(release_op) = tribute_rt::Release::from_operation(db, *op) {
            // Release: refcount decrement + conditional free + block split
            let ptr = release_op.ptr(db);
            let alloc_size = release_op.alloc_size(db);

            let continue_block_id = BlockId::fresh();
            let free_block_id = BlockId::fresh();

            // Generate decrement ops
            let (decrement_ops, is_zero_val, rc_addr_val) =
                gen_release_decrement(db, location, ptr);
            current_ops.extend(decrement_ops);

            // Create empty successor blocks for brif (they need to exist as Block values)
            let free_block_placeholder =
                Block::new(db, free_block_id, location, IdVec::new(), IdVec::new());
            let continue_block_placeholder =
                Block::new(db, continue_block_id, location, IdVec::new(), IdVec::new());

            // Terminate current block with brif
            let brif_op = clif::brif(
                db,
                location,
                is_zero_val,
                free_block_placeholder,
                continue_block_placeholder,
            );
            current_ops.push(brif_op.as_operation());

            // Finish current block
            result_blocks.push(Block::new(
                db,
                current_id,
                location,
                current_args,
                current_ops.into_iter().collect(),
            ));

            // Create free block: dealloc + jump to continue
            let free_ops =
                gen_dealloc_call(db, location, rc_addr_val, alloc_size, continue_block_id);
            let free_block = Block::new(
                db,
                free_block_id,
                location,
                IdVec::new(),
                free_ops.into_iter().collect(),
            );
            result_blocks.push(free_block);

            // Start a new continue block for remaining ops
            current_id = continue_block_id;
            current_args = IdVec::new();
            current_ops = Vec::new();
        } else {
            current_ops.push(*op);
        }
    }

    // Finish the last block
    result_blocks.push(Block::new(
        db,
        current_id,
        location,
        current_args,
        current_ops.into_iter().collect(),
    ));

    result_blocks
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

    // rc_addr = ptr + (-8)
    let neg8 = clif::iconst(db, location, i64_ty, -8);
    ops.push(neg8.as_operation());
    let rc_addr = clif::iadd(db, location, ptr, neg8.result(db), ptr_ty);
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

    // rc_addr = ptr + (-8)
    let neg8 = clif::iconst(db, location, i64_ty, -8);
    ops.push(neg8.as_operation());
    let rc_addr = clif::iadd(db, location, ptr, neg8.result(db), ptr_ty);
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

/// Generate dealloc call ops for the free block.
///
/// Returns ops: `clif.iconst(alloc_size) + clif.call @__tribute_dealloc + clif.jump(continue)`.
fn gen_dealloc_call<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    rc_addr: Value<'db>,
    alloc_size: i64,
    continue_block_id: BlockId,
) -> Vec<Operation<'db>> {
    let i64_ty = core::I64::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();

    let mut ops = Vec::new();

    // dealloc_size = iconst(alloc_size)
    let size = clif::iconst(db, location, i64_ty, alloc_size);
    ops.push(size.as_operation());

    // call @__tribute_dealloc(rc_addr, size)
    let call = clif::call(
        db,
        location,
        [rc_addr, size.result(db)],
        nil_ty,
        Symbol::new(DEALLOC_FN),
    );
    ops.push(call.as_operation());

    // jump to continue block
    let continue_block = Block::new(db, continue_block_id, location, IdVec::new(), IdVec::new());
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
            ir.contains("__tribute_dealloc"),
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
