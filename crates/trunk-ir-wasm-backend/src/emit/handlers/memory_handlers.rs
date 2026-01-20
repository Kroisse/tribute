//! Memory operation handlers for wasm backend.
//!
//! This module handles memory management and linear memory access operations.

use trunk_ir::dialect::wasm;
use wasm_encoder::{Function, Instruction, MemArg};

use crate::CompilationResult;

use super::super::{ModuleInfo, emit_operands, set_result_local};

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
pub(crate) fn handle_memory_size<'db>(
    db: &'db dyn salsa::Database,
    mem_size_op: wasm::MemorySize<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let memory = mem_size_op.memory(db);
    function.instruction(&Instruction::MemorySize(memory));
    set_result_local(db, &mem_size_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle memory.grow operation
pub(crate) fn handle_memory_grow<'db>(
    db: &'db dyn salsa::Database,
    mem_grow_op: wasm::MemoryGrow<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = mem_grow_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memory = mem_grow_op.memory(db);
    function.instruction(&Instruction::MemoryGrow(memory));
    set_result_local(db, &mem_grow_op.operation(), ctx, function)?;
    Ok(())
}

// === Full-Width Loads ===

/// Handle i32.load operation
pub(crate) fn handle_i32_load<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I32Load<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 2);
    function.instruction(&Instruction::I32Load(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i64.load operation
pub(crate) fn handle_i64_load<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 3);
    function.instruction(&Instruction::I64Load(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle f32.load operation
pub(crate) fn handle_f32_load<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::F32Load<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 2);
    function.instruction(&Instruction::F32Load(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle f64.load operation
pub(crate) fn handle_f64_load<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::F64Load<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 3);
    function.instruction(&Instruction::F64Load(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

// === Partial-Width Loads (i32) ===

/// Handle i32.load8_s operation
pub(crate) fn handle_i32_load8_s<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I32Load8S<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 0);
    function.instruction(&Instruction::I32Load8S(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i32.load8_u operation
pub(crate) fn handle_i32_load8_u<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I32Load8U<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 0);
    function.instruction(&Instruction::I32Load8U(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i32.load16_s operation
pub(crate) fn handle_i32_load16_s<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I32Load16S<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 1);
    function.instruction(&Instruction::I32Load16S(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i32.load16_u operation
pub(crate) fn handle_i32_load16_u<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I32Load16U<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 1);
    function.instruction(&Instruction::I32Load16U(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

// === Partial-Width Loads (i64) ===

/// Handle i64.load8_s operation
pub(crate) fn handle_i64_load8_s<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load8S<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 0);
    function.instruction(&Instruction::I64Load8S(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i64.load8_u operation
pub(crate) fn handle_i64_load8_u<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load8U<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 0);
    function.instruction(&Instruction::I64Load8U(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i64.load16_s operation
pub(crate) fn handle_i64_load16_s<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load16S<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 1);
    function.instruction(&Instruction::I64Load16S(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i64.load16_u operation
pub(crate) fn handle_i64_load16_u<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load16U<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 1);
    function.instruction(&Instruction::I64Load16U(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i64.load32_s operation
pub(crate) fn handle_i64_load32_s<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load32S<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 2);
    function.instruction(&Instruction::I64Load32S(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle i64.load32_u operation
pub(crate) fn handle_i64_load32_u<'db>(
    db: &'db dyn salsa::Database,
    load_op: wasm::I64Load32U<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = load_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(load_op.offset(db), load_op.align(db), load_op.memory(db), 2);
    function.instruction(&Instruction::I64Load32U(memarg));
    set_result_local(db, &load_op.operation(), ctx, function)?;
    Ok(())
}

// === Full-Width Stores ===

/// Handle i32.store operation
pub(crate) fn handle_i32_store<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I32Store<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        2,
    );
    function.instruction(&Instruction::I32Store(memarg));
    Ok(())
}

/// Handle i64.store operation
pub(crate) fn handle_i64_store<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I64Store<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        3,
    );
    function.instruction(&Instruction::I64Store(memarg));
    Ok(())
}

/// Handle f32.store operation
pub(crate) fn handle_f32_store<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::F32Store<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        2,
    );
    function.instruction(&Instruction::F32Store(memarg));
    Ok(())
}

/// Handle f64.store operation
pub(crate) fn handle_f64_store<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::F64Store<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        3,
    );
    function.instruction(&Instruction::F64Store(memarg));
    Ok(())
}

// === Partial-Width Stores ===

/// Handle i32.store8 operation
pub(crate) fn handle_i32_store8<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I32Store8<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        0,
    );
    function.instruction(&Instruction::I32Store8(memarg));
    Ok(())
}

/// Handle i32.store16 operation
pub(crate) fn handle_i32_store16<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I32Store16<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        1,
    );
    function.instruction(&Instruction::I32Store16(memarg));
    Ok(())
}

/// Handle i64.store8 operation
pub(crate) fn handle_i64_store8<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I64Store8<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        0,
    );
    function.instruction(&Instruction::I64Store8(memarg));
    Ok(())
}

/// Handle i64.store16 operation
pub(crate) fn handle_i64_store16<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I64Store16<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        1,
    );
    function.instruction(&Instruction::I64Store16(memarg));
    Ok(())
}

/// Handle i64.store32 operation
pub(crate) fn handle_i64_store32<'db>(
    db: &'db dyn salsa::Database,
    store_op: wasm::I64Store32<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = store_op.operation().operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let memarg = make_memarg(
        store_op.offset(db),
        store_op.align(db),
        store_op.memory(db),
        2,
    );
    function.instruction(&Instruction::I64Store32(memarg));
    Ok(())
}
