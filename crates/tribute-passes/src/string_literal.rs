//! The `adt` operations a string constant becomes on every target.
//!
//! An `adt.string_const` is the `Leaf` of the prelude `String` holding a
//! `Bytes` representation that views all of a byte array constant
//! (`design/string.md` § "Constant lowering"). Each target's constant
//! lowering builds these operations and then lowers the `adt.bytes_const`
//! among them as it lowers any other byte array constant.

use tribute_ir::dialect::adt;
use tribute_ir::metadata::StringLeaf;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::types::Location;

/// The operations that build the string `text` as a value of `result_ty`,
/// in order. The last one produces the value. `nat` builds the target's
/// constant of a `Nat` field type. `None`, with nothing built, when `leaf`
/// names no `Bytes` representation layout.
pub(crate) fn build(
    ctx: &mut IrContext,
    leaf: &StringLeaf,
    text: &[u8],
    result_ty: TypeRef,
    loc: Location,
    mut nat: impl FnMut(&mut IrContext, TypeRef, i64) -> (OpRef, ValueRef),
) -> Option<Vec<OpRef>> {
    let nat_ty = adt::field_type(ctx, leaf.bytes, leaf.start as usize)?;
    let array_ty = tribute_ir::runtime_layout::byte_array(ctx);
    let data = adt::BytesConst::operands()
        .value(text.to_vec().into())
        .results(array_ty)
        .build(ctx, loc);
    let (start_op, start) = nat(ctx, nat_ty, 0);
    let (len_op, len) = nat(ctx, nat_ty, text.len() as i64);
    let fields = leaf.bytes_fields(data.result(ctx), start, len);
    let bytes = adt::StructNew::operands(fields)
        .results(leaf.bytes)
        .build(ctx, loc);
    let string = adt::VariantNew::operands([bytes.result(ctx)])
        .tag(u64::from(leaf.leaf))
        .results(leaf.string)
        .build(ctx, loc);
    let mut ops = vec![
        data.op_ref(),
        start_op,
        len_op,
        bytes.op_ref(),
        string.op_ref(),
    ];
    if result_ty != leaf.string {
        let cast = core::UnrealizedConversionCast::operands(string.result(ctx))
            .results(result_ty)
            .build(ctx, loc);
        ops.push(cast.op_ref());
    }
    Some(ops)
}
