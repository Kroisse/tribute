//! Lower target-independent output to WASI preview1.
//!
//! The lowering declares the module-level resources its write helper relies
//! on, the `fd_write` import and a linear memory holding the helper's cells,
//! so module assembly reads them from the IR.

use tribute_ir::dialect::tribute_io;
use trunk_ir::Symbol;
use trunk_ir::SymbolPath;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, IllegalOp, LegalityCheck, Module, PatternApplicator,
    PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir_wasm_backend::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX};

const IO_TO_WASM: &str = "io-to-wasm";
const WRITE_HELPER: &str = "__tribute_wasi_write";
const WASI_MODULE: &str = "wasi_snapshot_preview1";
const FD_WRITE: &str = "fd_write";
// WASI preview1 `errno::intr`.
const WASI_ERRNO_INTR: i32 = 27;
const PAGE_SIZE: i32 = 65_536;

// Linear-memory cells reserved at the start of memory 0: one iovec, the
// `nwritten` result, then the scratch buffer that grows on demand.
const IOVEC_OFFSET: i32 = 0;
const NWRITTEN_OFFSET: i32 = IOVEC_OFFSET + 8;
const SCRATCH_OFFSET: i32 = NWRITTEN_OFFSET + 4;

const BYTES_DATA_FIELD: u32 = 0;
const BYTES_OFFSET_FIELD: u32 = 1;
const BYTES_LEN_FIELD: u32 = 2;

fn has_write(ctx: &IrContext, module: Module) -> bool {
    module.body(ctx).is_some_and(|body| {
        let mut found = false;
        walk_ops(ctx, body, &mut |ctx, op| {
            found |= tribute_io::Write::from_op(ctx, op).is_ok();
        });
        found
    })
}

fn walk_ops(ctx: &IrContext, region: RegionRef, callback: &mut impl FnMut(&IrContext, OpRef)) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            callback(ctx, op);
            for nested in ctx.op_regions(op) {
                walk_ops(ctx, nested, callback);
            }
        }
    }
}

pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    if has_write(ctx, module) {
        let location = ctx.op(module.op()).location;
        let block = module
            .first_block(ctx)
            .expect("module should have a body block");
        declare_host_resources(ctx, block, location)?;
        let helper = build_write_helper(ctx, location);
        ctx.push_op(block, helper);
    }

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(WritePattern)
        .with_target(ConversionTarget::new().illegal_dialect("tribute_io"))
        .apply_partial_conversion(ctx, module, IO_TO_WASM)?;
    Ok(())
}

/// Declare the `fd_write` import and a memory that holds the reserved cells,
/// at the start of the module, unless the module already declares them.
///
/// An existing import under the `fd_write` symbol is reused only if it is the
/// WASI preview1 `fd_write` with the signature the helper calls; any other
/// import under that symbol is rejected. An existing memory is reused only if
/// it is a 32-bit memory, since the helper addresses it with `i32`.
fn declare_host_resources(
    ctx: &mut IrContext,
    block: trunk_ir::BlockRef,
    loc: Location,
) -> Result<(), ConversionError> {
    let i32_ty = simple_type(ctx, "core", "i32");
    let import_ty = wasm_dialect::func_sig(ctx, [i32_ty; 4], [i32_ty]).as_type_ref();

    let mut import = None;
    let mut memory = None;
    for &op in &ctx.block(block).ops {
        if let Ok(declared) = wasm_dialect::ImportFunc::from_op(ctx, op) {
            if declared.sym_name(ctx) == FD_WRITE {
                import = Some(declared);
            }
        } else if let Ok(declared) = wasm_dialect::Memory::from_op(ctx, op) {
            memory = Some(declared);
        }
    }

    if let Some(import) = import
        && (import.module(ctx) != WASI_MODULE
            || import.name(ctx) != FD_WRITE
            || import.r#type(ctx) != import_ty)
    {
        return Err(incompatible(
            ctx,
            import.op_ref(),
            format!(
                "`@{FD_WRITE}` must import `{WASI_MODULE}.{FD_WRITE}` with type \
                 `(i32, i32, i32, i32) -> i32`"
            ),
        ));
    }
    if let Some(memory) = memory
        && memory.memory64(ctx)
    {
        return Err(incompatible(
            ctx,
            memory.op_ref(),
            "output lowering addresses memory 0 with i32 and requires a 32-bit memory".to_owned(),
        ));
    }

    let required_pages = (SCRATCH_OFFSET as u32).div_ceil(PAGE_SIZE as u32).max(1);
    let mut preamble = Vec::new();
    if import.is_none() {
        let import = wasm_dialect::ImportFunc::operands()
            .module(WASI_MODULE)
            .name(FD_WRITE)
            .sym_name(FD_WRITE)
            .r#type(import_ty)
            .build(ctx, loc);
        preamble.push(import.op_ref());
    }
    match memory {
        Some(memory) if memory.min(ctx) < required_pages => {
            ctx.op_mut(memory.op_ref())
                .attributes
                .insert(Symbol::new("min"), Attribute::from(required_pages));
        }
        Some(_) => {}
        None => {
            let memory = wasm_dialect::Memory::operands()
                .min(required_pages)
                .max(0)
                .shared(false)
                .memory64(false)
                .build(ctx, loc);
            preamble.push(memory.op_ref());
        }
    }

    match ctx.block(block).ops.first().copied() {
        Some(first) => {
            for op in preamble {
                ctx.insert_op_before(block, first, op);
            }
        }
        None => {
            for op in preamble {
                ctx.push_op(block, op);
            }
        }
    }
    Ok(())
}

/// An `io-to-wasm` boundary error for a declaration the lowering cannot reuse.
fn incompatible(ctx: &IrContext, op: OpRef, reason: String) -> ConversionError {
    let data = ctx.op(op);
    let conflict = IllegalOp {
        op,
        dialect: data.dialect.clone(),
        name: data.name.clone(),
        legality: LegalityCheck::Illegal,
        reason: None,
    }
    .with_reason(reason);
    ConversionError::new(IO_TO_WASM, vec![conflict])
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
        let call = func::Call::operands([write.bytes(ctx), write.newline(ctx)])
            .callee(SymbolPath::from(WRITE_HELPER))
            .results([ctx.op_result_types(op)[0]])
            .build(ctx, ctx.op(op).location);
        rewriter.replace_op(call.op_ref());
        true
    }
}

fn build_write_helper(ctx: &mut IrContext, loc: Location) -> OpRef {
    let i32_ty = simple_type(ctx, "core", "i32");
    let bytes_ty = super::bytes::bytes_struct_type(ctx);
    let nil_ty = core::nil(ctx).as_type_ref();
    let body = ctx.create_block(BlockData {
        location: loc,
        args: vec![block_arg(bytes_ty), block_arg(i32_ty)],
        ops: smallvec![],
        parent_region: None,
    });
    let bytes = ctx.block_arg(body, 0);
    let newline = ctx.block_arg(body, 1);

    let array_ref_ty = super::bytes::bytes_data_type(ctx);
    let data = wasm_dialect::StructGet::operands(bytes)
        .type_idx(BYTES_STRUCT_IDX)
        .field_idx(BYTES_DATA_FIELD)
        .results(array_ref_ty)
        .build(ctx, loc);
    let offset = wasm_dialect::StructGet::operands(bytes)
        .type_idx(BYTES_STRUCT_IDX)
        .field_idx(BYTES_OFFSET_FIELD)
        .results(i32_ty)
        .build(ctx, loc);
    let len = wasm_dialect::StructGet::operands(bytes)
        .type_idx(BYTES_STRUCT_IDX)
        .field_idx(BYTES_LEN_FIELD)
        .results(i32_ty)
        .build(ctx, loc);
    for op in [data.op_ref(), offset.op_ref(), len.op_ref()] {
        ctx.push_op(body, op);
    }

    let total = wasm_dialect::I32Add::operands(len.result(ctx), newline).build(ctx, loc);
    ctx.push_op(body, total.op_ref());
    trap_if_less(ctx, body, loc, total.result(ctx), len.result(ctx), i32_ty);

    let scratch = i32_const(ctx, body, loc, i32_ty, SCRATCH_OFFSET);
    let end = wasm_dialect::I32Add::operands(scratch, total.result(ctx)).build(ctx, loc);
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

    let newline_addr = wasm_dialect::I32Add::operands(scratch, len.result(ctx)).build(ctx, loc);
    ctx.push_op(body, newline_addr.op_ref());
    let newline_region = region(ctx, loc, |ctx, block| {
        let lf = i32_const(ctx, block, loc, i32_ty, 10);
        let store = wasm_dialect::I32Store8::operands(newline_addr.result(ctx), lf)
            .offset(0)
            .align(0)
            .memory(0)
            .build(ctx, loc);
        ctx.push_op(block, store.op_ref());
    });
    let empty_region = region(ctx, loc, |_, _| {});
    let append_newline = wasm_dialect::If::operands(newline)
        .results([nil_ty])
        .regions(newline_region, empty_region)
        .build(ctx, loc);
    ctx.push_op(body, append_newline.op_ref());

    let writes = write_loop(ctx, loc, zero, total.result(ctx), i32_ty, nil_ty);
    ctx.push_op(body, writes);
    let ret = func::Return::operands([]).build(ctx, loc);
    ctx.push_op(body, ret.op_ref());

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![body],
        parent_op: None,
    });
    let fn_ty = func::func_sig(ctx, [bytes_ty, i32_ty], [nil_ty]).as_type_ref();
    func::Func::operands()
        .sym_name(WRITE_HELPER)
        .r#type(fn_ty)
        .regions(body)
        .build(ctx, loc)
        .op_ref()
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
    let end_minus_one = wasm_dialect::I32Sub::operands(end, one)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(body, end_minus_one.op_ref());
    let page_size = i32_const(ctx, body, loc, i32_ty, PAGE_SIZE);
    let quotient = wasm_dialect::I32DivU::operands(end_minus_one.result(ctx), page_size)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(body, quotient.op_ref());
    let required = wasm_dialect::I32Add::operands(quotient.result(ctx), one).build(ctx, loc);
    ctx.push_op(body, required.op_ref());
    let current = wasm_dialect::MemorySize::operands()
        .memory(0)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(body, current.op_ref());
    let needs_grow = wasm_dialect::I32GtU::operands(required.result(ctx), current.result(ctx))
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(body, needs_grow.op_ref());

    let grow_region = region(ctx, loc, |ctx, block| {
        let delta = wasm_dialect::I32Sub::operands(required.result(ctx), current.result(ctx))
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, delta.op_ref());
        let grown = wasm_dialect::MemoryGrow::operands(delta.result(ctx))
            .memory(0)
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, grown.op_ref());
        let failed = wasm_dialect::I32Const::operands()
            .value(-1)
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, failed.op_ref());
        let is_failed = wasm_dialect::I32Eq::operands(grown.result(ctx), failed.result(ctx))
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, is_failed.op_ref());
        trap_if(ctx, block, loc, is_failed.result(ctx), nil_ty);
    });
    let no_grow = region(ctx, loc, |_, _| {});
    let grow_if = wasm_dialect::If::operands(needs_grow.result(ctx))
        .results([nil_ty])
        .regions(grow_region, no_grow)
        .build(ctx, loc);
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
    let done = wasm_dialect::I32GeU::operands(index, len)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(loop_block, done.op_ref());
    let break_if_done = wasm_dialect::BrIf::operands(done.result(ctx))
        .target(1)
        .build(ctx, loc);
    ctx.push_op(loop_block, break_if_done.op_ref());
    let source = wasm_dialect::I32Add::operands(offset, index).build(ctx, loc);
    ctx.push_op(loop_block, source.op_ref());
    let byte = wasm_dialect::ArrayGetU::operands(data, source.result(ctx))
        .type_idx(BYTES_ARRAY_IDX)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(loop_block, byte.op_ref());
    let destination = wasm_dialect::I32Add::operands(scratch, index).build(ctx, loc);
    ctx.push_op(loop_block, destination.op_ref());
    let store = wasm_dialect::I32Store8::operands(destination.result(ctx), byte.result(ctx))
        .offset(0)
        .align(0)
        .memory(0)
        .build(ctx, loc);
    ctx.push_op(loop_block, store.op_ref());
    let one = i32_const(ctx, loop_block, loc, i32_ty, 1);
    let next = wasm_dialect::I32Add::operands(index, one).build(ctx, loc);
    ctx.push_op(loop_block, next.op_ref());
    let yield_next = wasm_dialect::Yield::operands(next.result(ctx)).build(ctx, loc);
    ctx.push_op(loop_block, yield_next.op_ref());
    let continue_loop = wasm_dialect::Br::operands().target(0).build(ctx, loc);
    ctx.push_op(loop_block, continue_loop.op_ref());
    loop_in_block(ctx, loc, init, loop_block, nil_ty)
}

fn write_loop(
    ctx: &mut IrContext,
    loc: Location,
    init: ValueRef,
    total: ValueRef,
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
    let done = wasm_dialect::I32GeU::operands(written, total)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(loop_block, done.op_ref());
    let break_if_done = wasm_dialect::BrIf::operands(done.result(ctx))
        .target(1)
        .build(ctx, loc);
    ctx.push_op(loop_block, break_if_done.op_ref());

    let scratch = i32_const(ctx, loop_block, loc, i32_ty, SCRATCH_OFFSET);
    let ptr = wasm_dialect::I32Add::operands(scratch, written).build(ctx, loc);
    ctx.push_op(loop_block, ptr.op_ref());
    let remaining = wasm_dialect::I32Sub::operands(total, written)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(loop_block, remaining.op_ref());
    let iovec = i32_const(ctx, loop_block, loc, i32_ty, IOVEC_OFFSET);
    let store_ptr = wasm_dialect::I32Store::operands(iovec, ptr.result(ctx))
        .offset(0)
        .align(2)
        .memory(0)
        .build(ctx, loc);
    ctx.push_op(loop_block, store_ptr.op_ref());
    let store_len = wasm_dialect::I32Store::operands(iovec, remaining.result(ctx))
        .offset(4)
        .align(2)
        .memory(0)
        .build(ctx, loc);
    ctx.push_op(loop_block, store_len.op_ref());

    let stdout = i32_const(ctx, loop_block, loc, i32_ty, 1);
    let one_iovec = i32_const(ctx, loop_block, loc, i32_ty, 1);
    let nwritten = i32_const(ctx, loop_block, loc, i32_ty, NWRITTEN_OFFSET);
    let call = wasm_dialect::Call::operands([stdout, iovec, one_iovec, nwritten])
        .callee(SymbolPath::from(FD_WRITE))
        .results([i32_ty])
        .build(ctx, loc);
    ctx.push_op(loop_block, call.op_ref());
    let zero = i32_const(ctx, loop_block, loc, i32_ty, 0);
    let succeeded = wasm_dialect::I32Eq::operands(call.results(ctx)[0], zero)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(loop_block, succeeded.op_ref());

    let success = region(ctx, loc, |ctx, block| {
        let count = wasm_dialect::I32Load::operands(nwritten)
            .offset(0)
            .align(2)
            .memory(0)
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, count.op_ref());
        let no_progress = wasm_dialect::I32Eq::operands(count.result(ctx), zero)
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, no_progress.op_ref());
        let break_if_stalled = wasm_dialect::BrIf::operands(no_progress.result(ctx))
            .target(2)
            .build(ctx, loc);
        ctx.push_op(block, break_if_stalled.op_ref());
        let next = wasm_dialect::I32Add::operands(written, count.result(ctx)).build(ctx, loc);
        ctx.push_op(block, next.op_ref());
        let yield_next = wasm_dialect::Yield::operands(next.result(ctx)).build(ctx, loc);
        ctx.push_op(block, yield_next.op_ref());
        let continue_loop = wasm_dialect::Br::operands().target(1).build(ctx, loc);
        ctx.push_op(block, continue_loop.op_ref());
    });
    let failure = region(ctx, loc, |ctx, block| {
        let intr = i32_const(ctx, block, loc, i32_ty, WASI_ERRNO_INTR);
        let interrupted = wasm_dialect::I32Eq::operands(call.results(ctx)[0], intr)
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(block, interrupted.op_ref());
        let retry_if_interrupted = wasm_dialect::BrIf::operands(interrupted.result(ctx))
            .target(1)
            .build(ctx, loc);
        ctx.push_op(block, retry_if_interrupted.op_ref());
        let stop = wasm_dialect::Br::operands().target(2).build(ctx, loc);
        ctx.push_op(block, stop.op_ref());
    });
    let handle_result = wasm_dialect::If::operands(succeeded.result(ctx))
        .results([nil_ty])
        .regions(success, failure)
        .build(ctx, loc);
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
    let loop_op = wasm_dialect::Loop::operands([init])
        .results([nil_ty])
        .regions(loop_region)
        .build(ctx, loc);
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
    wasm_dialect::Block::operands()
        .results([nil_ty])
        .regions(region)
        .build(ctx, loc)
        .op_ref()
}

fn trap_if_less(
    ctx: &mut IrContext,
    body: trunk_ir::BlockRef,
    loc: Location,
    lhs: ValueRef,
    rhs: ValueRef,
    i32_ty: TypeRef,
) {
    let overflow = wasm_dialect::I32LtU::operands(lhs, rhs)
        .results(i32_ty)
        .build(ctx, loc);
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
        let unreachable = wasm_dialect::Unreachable::operands().build(ctx, loc);
        ctx.push_op(block, unreachable.op_ref());
    });
    let ok = region(ctx, loc, |_, _| {});
    let if_op = wasm_dialect::If::operands(condition)
        .results([nil_ty])
        .regions(trap, ok)
        .build(ctx, loc);
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
    let op = wasm_dialect::I32Const::operands()
        .value(value)
        .results(ty)
        .build(ctx, loc);
    let result = op.result(ctx);
    ctx.push_op(block, op.op_ref());
    result
}

fn simple_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
}

fn block_arg(ty: TypeRef) -> BlockArgData {
    BlockArgData {
        ty,
        attrs: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn op_names(ctx: &IrContext, module: Module) -> Vec<String> {
        module
            .ops(ctx)
            .iter()
            .copied()
            .map(|op| format!("{}.{}", ctx.op(op).dialect, ctx.op(op).name))
            .collect()
    }

    #[test]
    fn lowering_without_writes_leaves_the_module_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil {
    func.return
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        lower(&mut ctx, module).expect("module without writes should lower");

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn lowering_declares_the_import_and_memory_the_helper_uses() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%bytes: core.bytes, %newline: core.i1) -> core.nil {
    %write = tribute_io.write %bytes, %newline : core.nil
    func.return
  }
}"#,
        );

        lower(&mut ctx, module).expect("write should lower");

        assert_eq!(
            op_names(&ctx, module),
            ["wasm.import_func", "wasm.memory", "func.func", "func.func"]
        );
        let ops = module.ops(&ctx);
        let import = wasm_dialect::ImportFunc::from_op(&ctx, ops[0]).expect("import");
        assert_eq!(import.module(&ctx), Symbol::new(WASI_MODULE));
        assert_eq!(import.sym_name(&ctx), FD_WRITE);
        let memory = wasm_dialect::Memory::from_op(&ctx, ops[1]).expect("memory");
        assert_eq!(memory.min(&ctx), 1);
        let helper = func::Func::from_op(&ctx, ops[3]).expect("write helper");
        assert_eq!(helper.sym_name(&ctx), WRITE_HELPER);
    }

    #[test]
    fn lowering_reuses_declared_import_and_memory() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.import_func {module = "wasi_snapshot_preview1", name = "fd_write", sym_name = "fd_write", type = wasm.func_sig<(core.i32, core.i32, core.i32, core.i32) -> core.i32>}
  wasm.memory {min = 0, max = 0, shared = false, memory64 = false}
  func.func @main(%bytes: core.bytes, %newline: core.i1) -> core.nil {
    %write = tribute_io.write %bytes, %newline : core.nil
    func.return
  }
}"#,
        );

        lower(&mut ctx, module).expect("write should lower");

        assert_eq!(
            op_names(&ctx, module),
            ["wasm.import_func", "wasm.memory", "func.func", "func.func"]
        );
        let memory = wasm_dialect::Memory::from_op(&ctx, module.ops(&ctx)[1]).expect("memory");
        assert_eq!(memory.min(&ctx), 1, "the reserved cells need one page");
    }

    #[test]
    fn lowering_rejects_a_conflicting_fd_write_import() {
        for import in [
            "wasm.import_func {module = \"env\", name = \"fd_write\", sym_name = \"fd_write\", type = wasm.func_sig<(core.i32, core.i32, core.i32, core.i32) -> core.i32>}",
            "wasm.import_func {module = \"wasi_snapshot_preview1\", name = \"fd_read\", sym_name = \"fd_write\", type = wasm.func_sig<(core.i32, core.i32, core.i32, core.i32) -> core.i32>}",
            "wasm.import_func {module = \"wasi_snapshot_preview1\", name = \"fd_write\", sym_name = \"fd_write\", type = wasm.func_sig<(core.i32) -> core.i32>}",
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  {import}
  func.func @main(%bytes: core.bytes, %newline: core.i1) -> core.nil {{
    %write = tribute_io.write %bytes, %newline : core.nil
    func.return
  }}
}}"#
                ),
            );
            let before = print_module(&ctx, module.op());

            let error = lower(&mut ctx, module).expect_err("conflicting import must be rejected");

            assert_eq!(error.boundary(), IO_TO_WASM);
            assert!(error.to_string().contains("must import"), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before, "{import}");
        }
    }

    #[test]
    fn lowering_rejects_a_64_bit_memory_and_reuses_a_shared_32_bit_memory() {
        let module_text = |memory64: bool| {
            format!(
                r#"core.module @test {{
  wasm.memory {{min = 1, max = 2, shared = true, memory64 = {memory64}}}
  func.func @main(%bytes: core.bytes, %newline: core.i1) -> core.nil {{
    %write = tribute_io.write %bytes, %newline : core.nil
    func.return
  }}
}}"#
            )
        };

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(true));
        let before = print_module(&ctx, module.op());
        let error = lower(&mut ctx, module).expect_err("memory64 must be rejected");
        assert_eq!(error.boundary(), IO_TO_WASM);
        assert!(error.to_string().contains("32-bit memory"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(false));
        lower(&mut ctx, module).expect("shared 32-bit memory should be reused");
        assert_eq!(
            op_names(&ctx, module),
            ["wasm.import_func", "wasm.memory", "func.func", "func.func"]
        );
    }
}
