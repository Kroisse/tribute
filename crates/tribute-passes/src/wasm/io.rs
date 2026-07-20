//! Lower target-independent output to WASI preview1.

use std::collections::BTreeMap;

use tribute_ir::dialect::tribute_io;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Location, TypeDataBuilder};
use trunk_ir_wasm_backend::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX};

const WRITE_HELPER: &str = "__tribute_wasi_write";
// WASI preview1 `errno::intr`.
const WASI_ERRNO_INTR: i32 = 27;
const PAGE_SIZE: i32 = 65_536;

const BYTES_DATA_FIELD: u32 = 0;
const BYTES_OFFSET_FIELD: u32 = 1;
const BYTES_LEN_FIELD: u32 = 2;

/// Linear-memory cells reserved for dynamic output.
pub struct IoAnalysis {
    pub needs_fd_write: bool,
    pub iovec_offset: u32,
    pub nwritten_offset: u32,
    pub scratch_offset: u32,
    pub total_size: u32,
}

impl IoAnalysis {
    pub fn analyze(ctx: &IrContext, module: Module, base_offset: u32) -> Self {
        let needs_fd_write = module.body(ctx).is_some_and(|body| {
            let mut found = false;
            walk_ops(ctx, body, &mut |ctx, op| {
                found |= tribute_io::Write::from_op(ctx, op).is_ok();
            });
            found
        });

        if !needs_fd_write {
            return Self {
                needs_fd_write: false,
                iovec_offset: base_offset,
                nwritten_offset: base_offset,
                scratch_offset: base_offset,
                total_size: 0,
            };
        }

        let iovec_offset = base_offset.div_ceil(4) * 4;
        let nwritten_offset = iovec_offset + 8;
        let scratch_offset = nwritten_offset + 4;
        Self {
            needs_fd_write: true,
            iovec_offset,
            nwritten_offset,
            scratch_offset,
            total_size: scratch_offset - base_offset,
        }
    }
}

fn walk_ops(ctx: &IrContext, region: RegionRef, callback: &mut impl FnMut(&IrContext, OpRef)) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            callback(ctx, op);
            for &nested in &ctx.op(op).regions {
                walk_ops(ctx, nested, callback);
            }
        }
    }
}

pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    analysis: &IoAnalysis,
) -> Result<(), ConversionError> {
    if analysis.needs_fd_write {
        let helper = build_write_helper(ctx, ctx.op(module.op()).location, analysis);
        let block = module
            .first_block(ctx)
            .expect("module should have a body block");
        ctx.push_op(block, helper);
    }

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(WritePattern)
        .with_target(ConversionTarget::new().illegal_dialect("tribute_io"))
        .apply_partial_conversion(ctx, module, "io-to-wasm")?;
    Ok(())
}

struct WritePattern;

impl RewritePattern for WritePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(write) = tribute_io::Write::from_op(ctx, op) else {
            return false;
        };
        let call = func::call(
            ctx,
            ctx.op(op).location,
            [write.bytes(ctx), write.newline(ctx)],
            ctx.op_result_types(op)[0],
            Symbol::new(WRITE_HELPER),
        );
        rewriter.replace_op(call.op_ref());
        true
    }
}

fn build_write_helper(ctx: &mut IrContext, loc: Location, analysis: &IoAnalysis) -> OpRef {
    let i32_ty = simple_type(ctx, "core", "i32");
    let bytes_ty = core::bytes(ctx).as_type_ref();
    let nil_ty = core::nil(ctx).as_type_ref();
    let body = ctx.create_block(BlockData {
        location: loc,
        args: vec![block_arg(bytes_ty), block_arg(i32_ty)],
        ops: smallvec![],
        parent_region: None,
    });
    let bytes = ctx.block_arg(body, 0);
    let newline = ctx.block_arg(body, 1);

    let i8_ty = simple_type(ctx, "core", "i8");
    let array_ty = core::array(ctx, i8_ty).as_type_ref();
    let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();
    let data = wasm_dialect::struct_get(
        ctx,
        loc,
        bytes,
        array_ref_ty,
        BYTES_STRUCT_IDX,
        BYTES_DATA_FIELD,
    );
    let offset = wasm_dialect::struct_get(
        ctx,
        loc,
        bytes,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_OFFSET_FIELD,
    );
    let len = wasm_dialect::struct_get(ctx, loc, bytes, i32_ty, BYTES_STRUCT_IDX, BYTES_LEN_FIELD);
    for op in [data.op_ref(), offset.op_ref(), len.op_ref()] {
        ctx.push_op(body, op);
    }

    let total = wasm_dialect::i32_add(ctx, loc, len.result(ctx), newline, i32_ty);
    ctx.push_op(body, total.op_ref());
    trap_if_less(ctx, body, loc, total.result(ctx), len.result(ctx), i32_ty);

    let scratch = i32_const(ctx, body, loc, i32_ty, analysis.scratch_offset as i32);
    let end = wasm_dialect::i32_add(ctx, loc, scratch, total.result(ctx), i32_ty);
    ctx.push_op(body, end.op_ref());
    trap_if_less(ctx, body, loc, end.result(ctx), total.result(ctx), i32_ty);
    ensure_memory(ctx, body, loc, end.result(ctx), i32_ty, nil_ty);

    let zero = i32_const(ctx, body, loc, i32_ty, 0);
    let copy = copy_loop(
        ctx,
        CopyLoopInput {
            loc,
            init: zero,
            data: data.result(ctx),
            offset: offset.result(ctx),
            len: len.result(ctx),
            scratch,
            i32_ty,
            nil_ty,
        },
    );
    ctx.push_op(body, copy);

    let newline_addr = wasm_dialect::i32_add(ctx, loc, scratch, len.result(ctx), i32_ty);
    ctx.push_op(body, newline_addr.op_ref());
    let newline_region = region(ctx, loc, |ctx, block| {
        let lf = i32_const(ctx, block, loc, i32_ty, 10);
        let store = wasm_dialect::i32_store8(ctx, loc, newline_addr.result(ctx), lf, 0, 0, 0);
        ctx.push_op(block, store.op_ref());
    });
    let empty_region = region(ctx, loc, |_, _| {});
    let append_newline =
        wasm_dialect::r#if(ctx, loc, newline, nil_ty, newline_region, empty_region);
    ctx.push_op(body, append_newline.op_ref());

    let writes = write_loop(ctx, loc, zero, total.result(ctx), analysis, i32_ty, nil_ty);
    ctx.push_op(body, writes);
    let ret = func::r#return(ctx, loc, []);
    ctx.push_op(body, ret.op_ref());

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![body],
        parent_op: None,
    });
    let fn_ty = core::func(ctx, nil_ty, [bytes_ty, i32_ty]).as_type_ref();
    func::func(ctx, loc, Symbol::new(WRITE_HELPER), fn_ty, body).op_ref()
}

fn ensure_memory(
    ctx: &mut IrContext,
    body: trunk_ir::BlockRef,
    loc: Location,
    end: ValueRef,
    i32_ty: TypeRef,
    nil_ty: TypeRef,
) {
    let one = i32_const(ctx, body, loc, i32_ty, 1);
    let end_minus_one = wasm_dialect::i32_sub(ctx, loc, end, one, i32_ty);
    ctx.push_op(body, end_minus_one.op_ref());
    let page_size = i32_const(ctx, body, loc, i32_ty, PAGE_SIZE);
    let quotient = wasm_dialect::i32_div_u(ctx, loc, end_minus_one.result(ctx), page_size, i32_ty);
    ctx.push_op(body, quotient.op_ref());
    let required = wasm_dialect::i32_add(ctx, loc, quotient.result(ctx), one, i32_ty);
    ctx.push_op(body, required.op_ref());
    let current = wasm_dialect::memory_size(ctx, loc, i32_ty, 0);
    ctx.push_op(body, current.op_ref());
    let needs_grow =
        wasm_dialect::i32_gt_u(ctx, loc, required.result(ctx), current.result(ctx), i32_ty);
    ctx.push_op(body, needs_grow.op_ref());

    let grow_region = region(ctx, loc, |ctx, block| {
        let delta =
            wasm_dialect::i32_sub(ctx, loc, required.result(ctx), current.result(ctx), i32_ty);
        ctx.push_op(block, delta.op_ref());
        let grown = wasm_dialect::memory_grow(ctx, loc, delta.result(ctx), i32_ty, 0);
        ctx.push_op(block, grown.op_ref());
        let failed = wasm_dialect::i32_const(ctx, loc, i32_ty, -1);
        ctx.push_op(block, failed.op_ref());
        let is_failed =
            wasm_dialect::i32_eq(ctx, loc, grown.result(ctx), failed.result(ctx), i32_ty);
        ctx.push_op(block, is_failed.op_ref());
        trap_if(ctx, block, loc, is_failed.result(ctx), nil_ty);
    });
    let no_grow = region(ctx, loc, |_, _| {});
    let grow_if = wasm_dialect::r#if(
        ctx,
        loc,
        needs_grow.result(ctx),
        nil_ty,
        grow_region,
        no_grow,
    );
    ctx.push_op(body, grow_if.op_ref());
}

struct CopyLoopInput {
    loc: Location,
    init: ValueRef,
    data: ValueRef,
    offset: ValueRef,
    len: ValueRef,
    scratch: ValueRef,
    i32_ty: TypeRef,
    nil_ty: TypeRef,
}

fn copy_loop(ctx: &mut IrContext, input: CopyLoopInput) -> OpRef {
    let CopyLoopInput {
        loc,
        init,
        data,
        offset,
        len,
        scratch,
        i32_ty,
        nil_ty,
    } = input;
    let loop_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![block_arg(i32_ty)],
        ops: smallvec![],
        parent_region: None,
    });
    let index = ctx.block_arg(loop_block, 0);
    let done = wasm_dialect::i32_ge_u(ctx, loc, index, len, i32_ty);
    ctx.push_op(loop_block, done.op_ref());
    let break_if_done = wasm_dialect::br_if(ctx, loc, done.result(ctx), 1);
    ctx.push_op(loop_block, break_if_done.op_ref());
    let source = wasm_dialect::i32_add(ctx, loc, offset, index, i32_ty);
    ctx.push_op(loop_block, source.op_ref());
    let byte =
        wasm_dialect::array_get_u(ctx, loc, data, source.result(ctx), i32_ty, BYTES_ARRAY_IDX);
    ctx.push_op(loop_block, byte.op_ref());
    let destination = wasm_dialect::i32_add(ctx, loc, scratch, index, i32_ty);
    ctx.push_op(loop_block, destination.op_ref());
    let store =
        wasm_dialect::i32_store8(ctx, loc, destination.result(ctx), byte.result(ctx), 0, 0, 0);
    ctx.push_op(loop_block, store.op_ref());
    let one = i32_const(ctx, loop_block, loc, i32_ty, 1);
    let next = wasm_dialect::i32_add(ctx, loc, index, one, i32_ty);
    ctx.push_op(loop_block, next.op_ref());
    let yield_next = wasm_dialect::r#yield(ctx, loc, next.result(ctx));
    ctx.push_op(loop_block, yield_next.op_ref());
    let continue_loop = wasm_dialect::br(ctx, loc, 0);
    ctx.push_op(loop_block, continue_loop.op_ref());
    loop_in_block(ctx, loc, init, loop_block, nil_ty)
}

fn write_loop(
    ctx: &mut IrContext,
    loc: Location,
    init: ValueRef,
    total: ValueRef,
    analysis: &IoAnalysis,
    i32_ty: TypeRef,
    nil_ty: TypeRef,
) -> OpRef {
    let loop_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![block_arg(i32_ty)],
        ops: smallvec![],
        parent_region: None,
    });
    let written = ctx.block_arg(loop_block, 0);
    let done = wasm_dialect::i32_ge_u(ctx, loc, written, total, i32_ty);
    ctx.push_op(loop_block, done.op_ref());
    let break_if_done = wasm_dialect::br_if(ctx, loc, done.result(ctx), 1);
    ctx.push_op(loop_block, break_if_done.op_ref());

    let scratch = i32_const(ctx, loop_block, loc, i32_ty, analysis.scratch_offset as i32);
    let ptr = wasm_dialect::i32_add(ctx, loc, scratch, written, i32_ty);
    ctx.push_op(loop_block, ptr.op_ref());
    let remaining = wasm_dialect::i32_sub(ctx, loc, total, written, i32_ty);
    ctx.push_op(loop_block, remaining.op_ref());
    let iovec = i32_const(ctx, loop_block, loc, i32_ty, analysis.iovec_offset as i32);
    let store_ptr = wasm_dialect::i32_store(ctx, loc, iovec, ptr.result(ctx), 0, 2, 0);
    ctx.push_op(loop_block, store_ptr.op_ref());
    let store_len = wasm_dialect::i32_store(ctx, loc, iovec, remaining.result(ctx), 4, 2, 0);
    ctx.push_op(loop_block, store_len.op_ref());

    let stdout = i32_const(ctx, loop_block, loc, i32_ty, 1);
    let one_iovec = i32_const(ctx, loop_block, loc, i32_ty, 1);
    let nwritten = i32_const(
        ctx,
        loop_block,
        loc,
        i32_ty,
        analysis.nwritten_offset as i32,
    );
    let call = wasm_dialect::call(
        ctx,
        loc,
        [stdout, iovec, one_iovec, nwritten],
        [i32_ty],
        Symbol::new("fd_write"),
    );
    ctx.push_op(loop_block, call.op_ref());
    let zero = i32_const(ctx, loop_block, loc, i32_ty, 0);
    let succeeded = wasm_dialect::i32_eq(ctx, loc, call.results(ctx)[0], zero, i32_ty);
    ctx.push_op(loop_block, succeeded.op_ref());

    let success = region(ctx, loc, |ctx, block| {
        let count = wasm_dialect::i32_load(ctx, loc, nwritten, i32_ty, 0, 2, 0);
        ctx.push_op(block, count.op_ref());
        let no_progress = wasm_dialect::i32_eq(ctx, loc, count.result(ctx), zero, i32_ty);
        ctx.push_op(block, no_progress.op_ref());
        let break_if_stalled = wasm_dialect::br_if(ctx, loc, no_progress.result(ctx), 2);
        ctx.push_op(block, break_if_stalled.op_ref());
        let next = wasm_dialect::i32_add(ctx, loc, written, count.result(ctx), i32_ty);
        ctx.push_op(block, next.op_ref());
        let yield_next = wasm_dialect::r#yield(ctx, loc, next.result(ctx));
        ctx.push_op(block, yield_next.op_ref());
        let continue_loop = wasm_dialect::br(ctx, loc, 1);
        ctx.push_op(block, continue_loop.op_ref());
    });
    let failure = region(ctx, loc, |ctx, block| {
        let intr = i32_const(ctx, block, loc, i32_ty, WASI_ERRNO_INTR);
        let interrupted = wasm_dialect::i32_eq(ctx, loc, call.results(ctx)[0], intr, i32_ty);
        ctx.push_op(block, interrupted.op_ref());
        let retry_if_interrupted = wasm_dialect::br_if(ctx, loc, interrupted.result(ctx), 1);
        ctx.push_op(block, retry_if_interrupted.op_ref());
        let stop = wasm_dialect::br(ctx, loc, 2);
        ctx.push_op(block, stop.op_ref());
    });
    let handle_result =
        wasm_dialect::r#if(ctx, loc, succeeded.result(ctx), nil_ty, success, failure);
    ctx.push_op(loop_block, handle_result.op_ref());
    loop_in_block(ctx, loc, init, loop_block, nil_ty)
}

fn loop_in_block(
    ctx: &mut IrContext,
    loc: Location,
    init: ValueRef,
    loop_block: trunk_ir::BlockRef,
    nil_ty: TypeRef,
) -> OpRef {
    let loop_region = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![loop_block],
        parent_op: None,
    });
    let loop_op = wasm_dialect::r#loop(ctx, loc, [init], nil_ty, loop_region);
    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(block, loop_op.op_ref());
    let region = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    });
    wasm_dialect::block(ctx, loc, nil_ty, region).op_ref()
}

fn trap_if_less(
    ctx: &mut IrContext,
    body: trunk_ir::BlockRef,
    loc: Location,
    lhs: ValueRef,
    rhs: ValueRef,
    i32_ty: TypeRef,
) {
    let overflow = wasm_dialect::i32_lt_u(ctx, loc, lhs, rhs, i32_ty);
    ctx.push_op(body, overflow.op_ref());
    let nil_ty = core::nil(ctx).as_type_ref();
    trap_if(ctx, body, loc, overflow.result(ctx), nil_ty);
}

fn trap_if(
    ctx: &mut IrContext,
    body: trunk_ir::BlockRef,
    loc: Location,
    condition: ValueRef,
    nil_ty: TypeRef,
) {
    let trap = region(ctx, loc, |ctx, block| {
        let unreachable = wasm_dialect::unreachable(ctx, loc);
        ctx.push_op(block, unreachable.op_ref());
    });
    let ok = region(ctx, loc, |_, _| {});
    let if_op = wasm_dialect::r#if(ctx, loc, condition, nil_ty, trap, ok);
    ctx.push_op(body, if_op.op_ref());
}

fn region(
    ctx: &mut IrContext,
    loc: Location,
    build: impl FnOnce(&mut IrContext, trunk_ir::BlockRef),
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    build(ctx, block);
    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn i32_const(
    ctx: &mut IrContext,
    block: trunk_ir::BlockRef,
    loc: Location,
    ty: TypeRef,
    value: i32,
) -> ValueRef {
    let op = wasm_dialect::i32_const(ctx, loc, ty, value);
    let result = op.result(ctx);
    ctx.push_op(block, op.op_ref());
    result
}

fn simple_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new(dialect), Symbol::new(name)).build())
}

fn block_arg(ty: TypeRef) -> BlockArgData {
    BlockArgData {
        ty,
        attrs: BTreeMap::new(),
    }
}
