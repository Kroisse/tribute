//! Memory operation handlers for wasm backend.
//!
//! This module handles memory management and linear memory access operations.

use trunk_ir::arena::IrContext;
use trunk_ir::arena::dialect::wasm as wasm_dialect;
use wasm_encoder::{Function, Instruction, MemArg};

use crate::CompilationResult;

use super::super::value_emission::emit_operands;
use super::super::{FunctionEmitContext, ModuleInfo, set_result_local};

/// Create a MemArg for load/store instructions.
///
/// The `natural_align` is the log2 of the natural alignment:
/// - 0 for 8-bit (2^0 = 1 byte)
/// - 1 for 16-bit (2^1 = 2 bytes)
/// - 2 for 32-bit (2^2 = 4 bytes)
/// - 3 for 64-bit (2^3 = 8 bytes)
///
/// The align parameter is clamped to not exceed natural_align, per WebAssembly spec.
fn make_memarg(offset: u32, align: u32, memory_index: u32, natural_align: u32) -> MemArg {
    MemArg {
        offset: offset as u64,
        align: if align == 0 {
            natural_align
        } else {
            align.min(natural_align)
        },
        memory_index,
    }
}

// === Memory Management ===

/// Handle memory.size operation
pub(crate) fn handle_memory_size(
    ctx: &IrContext,
    mem_size_op: wasm_dialect::MemorySize,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let memory = mem_size_op.memory(ctx);
    function.instruction(&Instruction::MemorySize(memory));
    set_result_local(ctx, mem_size_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle memory.grow operation
pub(crate) fn handle_memory_grow(
    ctx: &IrContext,
    mem_grow_op: wasm_dialect::MemoryGrow,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(mem_grow_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memory = mem_grow_op.memory(ctx);
    function.instruction(&Instruction::MemoryGrow(memory));
    set_result_local(ctx, mem_grow_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

// === Full-Width Loads ===

/// Handle i32.load operation
pub(crate) fn handle_i32_load(
    ctx: &IrContext,
    load_op: wasm_dialect::I32Load,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::I32Load(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.load operation
pub(crate) fn handle_i64_load(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        3,
    );
    function.instruction(&Instruction::I64Load(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle f32.load operation
pub(crate) fn handle_f32_load(
    ctx: &IrContext,
    load_op: wasm_dialect::F32Load,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::F32Load(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle f64.load operation
pub(crate) fn handle_f64_load(
    ctx: &IrContext,
    load_op: wasm_dialect::F64Load,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        3,
    );
    function.instruction(&Instruction::F64Load(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

// === Partial-Width Loads (i32) ===

/// Handle i32.load8_s operation
pub(crate) fn handle_i32_load8_s(
    ctx: &IrContext,
    load_op: wasm_dialect::I32Load8S,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        0,
    );
    function.instruction(&Instruction::I32Load8S(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i32.load8_u operation
pub(crate) fn handle_i32_load8_u(
    ctx: &IrContext,
    load_op: wasm_dialect::I32Load8U,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        0,
    );
    function.instruction(&Instruction::I32Load8U(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i32.load16_s operation
pub(crate) fn handle_i32_load16_s(
    ctx: &IrContext,
    load_op: wasm_dialect::I32Load16S,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        1,
    );
    function.instruction(&Instruction::I32Load16S(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i32.load16_u operation
pub(crate) fn handle_i32_load16_u(
    ctx: &IrContext,
    load_op: wasm_dialect::I32Load16U,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        1,
    );
    function.instruction(&Instruction::I32Load16U(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

// === Partial-Width Loads (i64) ===

/// Handle i64.load8_s operation
pub(crate) fn handle_i64_load8_s(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load8S,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        0,
    );
    function.instruction(&Instruction::I64Load8S(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.load8_u operation
pub(crate) fn handle_i64_load8_u(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load8U,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        0,
    );
    function.instruction(&Instruction::I64Load8U(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.load16_s operation
pub(crate) fn handle_i64_load16_s(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load16S,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        1,
    );
    function.instruction(&Instruction::I64Load16S(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.load16_u operation
pub(crate) fn handle_i64_load16_u(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load16U,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        1,
    );
    function.instruction(&Instruction::I64Load16U(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.load32_s operation
pub(crate) fn handle_i64_load32_s(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load32S,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::I64Load32S(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.load32_u operation
pub(crate) fn handle_i64_load32_u(
    ctx: &IrContext,
    load_op: wasm_dialect::I64Load32U,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(load_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        load_op.offset(ctx),
        load_op.align(ctx),
        load_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::I64Load32U(memarg));
    set_result_local(ctx, load_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

// === Full-Width Stores ===

/// Handle i32.store operation
pub(crate) fn handle_i32_store(
    ctx: &IrContext,
    store_op: wasm_dialect::I32Store,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::I32Store(memarg));
    Ok(())
}

/// Handle i64.store operation
pub(crate) fn handle_i64_store(
    ctx: &IrContext,
    store_op: wasm_dialect::I64Store,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        3,
    );
    function.instruction(&Instruction::I64Store(memarg));
    Ok(())
}

/// Handle f32.store operation
pub(crate) fn handle_f32_store(
    ctx: &IrContext,
    store_op: wasm_dialect::F32Store,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::F32Store(memarg));
    Ok(())
}

/// Handle f64.store operation
pub(crate) fn handle_f64_store(
    ctx: &IrContext,
    store_op: wasm_dialect::F64Store,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        3,
    );
    function.instruction(&Instruction::F64Store(memarg));
    Ok(())
}

// === Partial-Width Stores ===

/// Handle i32.store8 operation
pub(crate) fn handle_i32_store8(
    ctx: &IrContext,
    store_op: wasm_dialect::I32Store8,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        0,
    );
    function.instruction(&Instruction::I32Store8(memarg));
    Ok(())
}

/// Handle i32.store16 operation
pub(crate) fn handle_i32_store16(
    ctx: &IrContext,
    store_op: wasm_dialect::I32Store16,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        1,
    );
    function.instruction(&Instruction::I32Store16(memarg));
    Ok(())
}

/// Handle i64.store8 operation
pub(crate) fn handle_i64_store8(
    ctx: &IrContext,
    store_op: wasm_dialect::I64Store8,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        0,
    );
    function.instruction(&Instruction::I64Store8(memarg));
    Ok(())
}

/// Handle i64.store16 operation
pub(crate) fn handle_i64_store16(
    ctx: &IrContext,
    store_op: wasm_dialect::I64Store16,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        1,
    );
    function.instruction(&Instruction::I64Store16(memarg));
    Ok(())
}

/// Handle i64.store32 operation
pub(crate) fn handle_i64_store32(
    ctx: &IrContext,
    store_op: wasm_dialect::I64Store32,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(store_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;
    let memarg = make_memarg(
        store_op.offset(ctx),
        store_op.align(ctx),
        store_op.memory(ctx),
        2,
    );
    function.instruction(&Instruction::I64Store32(memarg));
    Ok(())
}
