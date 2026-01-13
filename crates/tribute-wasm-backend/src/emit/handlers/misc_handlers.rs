//! Miscellaneous operation handlers for wasm backend.
//!
//! This module handles various WebAssembly operations that don't fit into other categories:
//! - wasm.bytes_from_data (create Bytes struct from passive data segment)

use trunk_ir::dialect::wasm;
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;
use crate::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX};

use super::super::{FunctionEmitContext, set_result_local};

/// Handle wasm.bytes_from_data operation
///
/// Compound operation: create Bytes struct from passive data segment
/// Stack operations:
///   i32.const <offset>    ; offset within data segment
///   i32.const <len>       ; number of bytes to copy
///   array.new_data $bytes_array <data_idx>
///   i32.const 0           ; offset field (we use the whole array)
///   i32.const <len>       ; len field
///   struct.new $bytes_struct
pub(crate) fn handle_bytes_from_data<'db>(
    db: &'db dyn salsa::Database,
    bytes_op: wasm::BytesFromData<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = bytes_op.operation();
    let data_idx = bytes_op.data_idx(db);
    let offset = bytes_op.offset(db);
    let len = bytes_op.len(db);

    // Push offset and length for array.new_data
    function.instruction(&Instruction::I32Const(offset as i32));
    function.instruction(&Instruction::I32Const(len as i32));
    function.instruction(&Instruction::ArrayNewData {
        array_type_index: BYTES_ARRAY_IDX,
        array_data_index: data_idx,
    });

    // Push struct fields: offset (0) and len
    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::I32Const(len as i32));
    function.instruction(&Instruction::StructNew(BYTES_STRUCT_IDX));

    set_result_local(db, &op, ctx, function)?;
    Ok(())
}
