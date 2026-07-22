//! Lower intrinsic calls to WASM operations.
//!
//! This pass transforms high-level intrinsic calls to low-level WASM instructions:
//! - `__bytes_len`, `__bytes_get_or_panic`, etc. -> WasmGC struct/array operations

use tribute_ir::ModulePathExt;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;

use trunk_ir_wasm_backend::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX};

// Bytes struct field indices (must match gc_types layout)
const BYTES_DATA_FIELD: u32 = 0; // ref (array i8)
const BYTES_OFFSET_FIELD: u32 = 1; // i32
const BYTES_LEN_FIELD: u32 = 2; // i32

/// Extracted Bytes struct fields: (data, offset, len) values.
struct BytesFields {
    data: ValueRef,
    offset: ValueRef,
    len: ValueRef,
}

/// Extract (data, offset, len) fields from a Bytes struct value.
///
/// Returns the extracted field values and the operations that produced them.
fn extract_bytes_fields(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    bytes_value: ValueRef,
) -> (BytesFields, Vec<OpRef>) {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let i8_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
    let array_ty = core::array(ctx, i8_ty).as_type_ref();
    let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();

    let get_data = wasm_dialect::struct_get(
        ctx,
        location,
        bytes_value,
        array_ref_ty,
        BYTES_STRUCT_IDX,
        BYTES_DATA_FIELD,
    );

    let get_offset = wasm_dialect::struct_get(
        ctx,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_OFFSET_FIELD,
    );

    let get_len = wasm_dialect::struct_get(
        ctx,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_LEN_FIELD,
    );

    let fields = BytesFields {
        data: get_data.result(ctx),
        offset: get_offset.result(ctx),
        len: get_len.result(ctx),
    };

    let ops = vec![get_data.op_ref(), get_offset.op_ref(), get_len.op_ref()];

    (fields, ops)
}

/// Lower target-independent Bytes intrinsic calls.
pub fn lower(ctx: &mut IrContext, module: Module) {
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(BytesLenPattern)
        .add_pattern(BytesGetOrPanicPattern)
        .add_pattern(BytesConcatPattern);

    applicator.apply_partial(ctx, module);
}

// =============================================================================
// Bytes intrinsic patterns
// =============================================================================

/// Check whether an operation calls one of the target-lowered Bytes helpers.
fn is_bytes_intrinsic_call(ctx: &IrContext, op: OpRef, intrinsic_names: &[&str]) -> bool {
    let Ok(call) = wasm_dialect::Call::from_op(ctx, op) else {
        return false;
    };
    let callee = call.callee(ctx).last_segment();
    intrinsic_names.iter().any(|name| callee == *name)
}

/// Pattern for `__bytes_len(bytes)` -> `struct.get $bytes 2`
///
/// Returns i32 directly since Nat is mapped to i32.
struct BytesLenPattern;

impl RewritePattern for BytesLenPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__bytes_len", "__tribute_bytes_len"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        let Some(bytes_ref) = operands.first().copied() else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        // struct.get to get len field (field 2)
        let get_len = wasm_dialect::struct_get(
            ctx,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_LEN_FIELD,
        );

        rewriter.replace_op(get_len.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesLenPattern"
    }
}

/// Pattern for `Bytes::get_or_panic(bytes, index)` -> array access with offset
///
/// Index is i32 (Nat), returns i32 (Nat, byte value 0-255).
struct BytesGetOrPanicPattern;

impl RewritePattern for BytesGetOrPanicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__bytes_get_or_panic"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 2 {
            return false;
        }
        let bytes_ref = operands[0];
        let index = operands[1]; // i32 (Nat)

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());

        // Get data array ref (field 0)
        let array_ty = core::array(ctx, i8_ty).as_type_ref();
        let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();
        let get_data = wasm_dialect::struct_get(
            ctx,
            location,
            bytes_ref,
            array_ref_ty,
            BYTES_STRUCT_IDX,
            BYTES_DATA_FIELD,
        );

        // Get offset (field 1)
        let get_offset = wasm_dialect::struct_get(
            ctx,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_OFFSET_FIELD,
        );

        // Add offset to index: actual_index = offset + index
        let add_offset =
            wasm_dialect::i32_add(ctx, location, get_offset.result(ctx), index, i32_ty);

        // array.get_u (unsigned extend to i32, for byte values 0-255)
        let array_get = wasm_dialect::array_get_u(
            ctx,
            location,
            get_data.result(ctx),
            add_offset.result(ctx),
            i32_ty,
            BYTES_ARRAY_IDX,
        );

        rewriter.insert_op(get_data.op_ref());
        rewriter.insert_op(get_offset.op_ref());
        rewriter.insert_op(add_offset.op_ref());
        rewriter.replace_op(array_get.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesGetOrPanicPattern"
    }
}

/// Pattern for `Bytes::concat(left, right)` -> allocate new array and copy both
struct BytesConcatPattern;

impl RewritePattern for BytesConcatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__bytes_concat", "__tribute_bytes_concat"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 2 {
            return false;
        }
        let left = operands[0];
        let right = operands[1];

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
        let bytes_ty = core::bytes(ctx).as_type_ref();
        let array_ty = core::array(ctx, i8_ty).as_type_ref();
        let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();

        // Extract fields from left and right Bytes structs
        let (left_fields, left_ops) = extract_bytes_fields(ctx, location, left);
        let (right_fields, right_ops) = extract_bytes_fields(ctx, location, right);

        // Calculate total_len = left.len + right.len
        let total_len =
            wasm_dialect::i32_add(ctx, location, left_fields.len, right_fields.len, i32_ty);

        // Allocate new array: array_new_default(total_len)
        let new_array = wasm_dialect::array_new_default(
            ctx,
            location,
            total_len.result(ctx),
            array_ref_ty,
            BYTES_ARRAY_IDX,
        );

        // Copy left bytes: array_copy(new_arr, 0, left.data, left.offset, left.len)
        let zero = wasm_dialect::i32_const(ctx, location, i32_ty, 0);

        let copy_left = wasm_dialect::array_copy(
            ctx,
            location,
            new_array.result(ctx),
            zero.result(ctx),
            left_fields.data,
            left_fields.offset,
            left_fields.len,
            BYTES_ARRAY_IDX,
            BYTES_ARRAY_IDX,
        );

        // Copy right bytes: array_copy(new_arr, left.len, right.data, right.offset, right.len)
        let copy_right = wasm_dialect::array_copy(
            ctx,
            location,
            new_array.result(ctx),
            left_fields.len,
            right_fields.data,
            right_fields.offset,
            right_fields.len,
            BYTES_ARRAY_IDX,
            BYTES_ARRAY_IDX,
        );

        // Create new Bytes struct: struct_new(new_arr, 0, total_len)
        let struct_new = wasm_dialect::struct_new(
            ctx,
            location,
            vec![
                new_array.result(ctx),
                zero.result(ctx),
                total_len.result(ctx),
            ],
            bytes_ty,
            BYTES_STRUCT_IDX,
        );

        // Combine all operations in order
        let mut ops = Vec::with_capacity(left_ops.len() + right_ops.len() + 6);
        ops.extend(left_ops);
        ops.extend(right_ops);
        ops.push(total_len.op_ref());
        ops.push(new_array.op_ref());
        ops.push(zero.op_ref());
        ops.push(copy_left.op_ref());
        ops.push(copy_right.op_ref());
        ops.push(struct_new.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }

    fn name(&self) -> &'static str {
        "BytesConcatPattern"
    }
}
