//! The Wasm `Bytes` layout and the in-boundary lowering of the bytes element
//! read intrinsic.
//!
//! A `Bytes` value is a struct of a backing byte array, a start offset, and a
//! length. Both types carry their runtime layout identifier (`@bytes`,
//! `@bytes_data`) and are built only by the constructors here. Wasm type
//! conversion maps `core.bytes` to the struct, so the view cast that the
//! lowering inserts folds away.

use tribute_ir::dialect::adt;
use tribute_ir::runtime_layout::{BYTES, BYTES_DATA, LAYOUT_ATTR};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{AttributeMap, TypeDataBuilder};

use crate::bytes_intrinsic::{self, BytesIntrinsicError};

/// Field index of the backing array in the bytes struct.
pub const DATA_FIELD: u32 = 0;
/// Field index of the start offset in the bytes struct.
pub const OFFSET_FIELD: u32 = 1;
/// Field index of the length in the bytes struct.
pub const LEN_FIELD: u32 = 2;

/// The bytes backing array: `core.array<core.i8, {layout = "bytes_data"}>`.
pub fn bytes_data_type(ctx: &mut IrContext) -> TypeRef {
    let i8_ty = ctx.intern_type(TypeDataBuilder::new("core", "i8").build());
    let layout = ctx.string_attr(BYTES_DATA);
    ctx.intern_type(
        TypeDataBuilder::new("core", "array")
            .param(i8_ty)
            .attr(LAYOUT_ATTR, layout)
            .build(),
    )
}

/// The bytes struct: backing array, start offset, and length, with
/// `layout = "bytes"`.
pub fn bytes_struct_type(ctx: &mut IrContext) -> TypeRef {
    let data_ty = bytes_data_type(ctx);
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let mut attrs = AttributeMap::new();
    attrs.insert(LAYOUT_ATTR, ctx.string_attr(BYTES));
    adt::struct_type(
        ctx,
        "_Bytes",
        [("data", data_ty), ("offset", i32_ty), ("len", i32_ty)],
        attrs,
    )
    .as_type_ref()
}

/// Lower calls to the bytes element read intrinsic to `adt` operations on
/// the Wasm bytes layout.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), BytesIntrinsicError> {
    bytes_intrinsic::lower_get_or_panic(ctx, module, lower_call)
}

/// Rewrite one bytes element read into reads of the bytes layout.
///
/// Emits:
///   %view   = core.unrealized_conversion_cast %bytes : !bytes
///   %data   = adt.struct_get %view {field = 0} : !bytes_data
///   %offset = adt.struct_get %view {field = 1} : core.i32
///   %at     = arith.addi %offset, %index : core.i32
///   %byte   = adt.array_get %data, %at : core.i8
///   %result = arith.extui %byte : core.i32
fn lower_call(ctx: &mut IrContext, call: OpRef) {
    let [bytes, index] = ctx.op_operands(call) else {
        unreachable!("call arity is validated before lowering")
    };
    let (bytes, index) = (*bytes, *index);
    let result_ty = ctx.op_result_types(call)[0];
    let loc = ctx.op(call).location;
    let bytes_ty = bytes_struct_type(ctx);
    let data_ty = bytes_data_type(ctx);
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());

    // View the opaque `core.bytes` as its Wasm layout. Wasm type conversion
    // maps both to the same type, so the cast folds away.
    let view = core::UnrealizedConversionCast::operands(bytes)
        .results(bytes_ty)
        .build(ctx, loc);
    let data = adt::StructGet::operands(view.result(ctx))
        .r#type(bytes_ty)
        .field(DATA_FIELD)
        .results(data_ty)
        .build(ctx, loc);
    let offset = adt::StructGet::operands(view.result(ctx))
        .r#type(bytes_ty)
        .field(OFFSET_FIELD)
        .results(i32_ty)
        .build(ctx, loc);
    let at = arith::Addi::operands(offset.result(ctx), index).build(ctx, loc);
    let byte = adt::ArrayGet::operands(data.result(ctx), at.result(ctx)).build(ctx, loc);
    // A byte is 0..=255: it widens with zero extension.
    let value = arith::Extui::operands(byte.result(ctx))
        .results(result_ty)
        .build(ctx, loc);
    bytes_intrinsic::replace_call(
        ctx,
        call,
        &[
            view.op_ref(),
            data.op_ref(),
            offset.op_ref(),
            at.op_ref(),
            byte.op_ref(),
            value.op_ref(),
        ],
        value.result(ctx),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tribute_ir::runtime_layout::has_runtime_layout;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn layout_types_carry_their_identifiers() {
        let mut ctx = IrContext::new();
        let bytes = bytes_struct_type(&mut ctx);
        let data = bytes_data_type(&mut ctx);
        assert!(has_runtime_layout(&ctx, bytes, BYTES));
        assert!(has_runtime_layout(&ctx, data, BYTES_DATA));
        let i8_ty = ctx.intern_type(TypeDataBuilder::new("core", "i8").build());
        let plain_i8_array = core::array(&mut ctx, i8_ty).as_type_ref();
        assert_ne!(plain_i8_array, data);
    }

    #[test]
    fn verified_identity_lowers_to_shared_reads_of_the_bytes_layout() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::__bytes_get_or_panic"}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @read} : core.i32
    func.return %byte
  }
}"#,
        );

        lower(&mut ctx, module).expect("bytes intrinsic lowering");

        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("func.call"), "{printed}");
        assert!(!printed.contains("@read"), "{printed}");
        assert!(!printed.contains("tribute.compiler_intrinsic"), "{printed}");
        assert!(!printed.contains("wasm."), "{printed}");
        assert_eq!(printed.matches("adt.struct_get").count(), 2, "{printed}");
        assert!(printed.contains("adt.array_get"), "{printed}");
        assert!(printed.contains("arith.extui"), "{printed}");
        assert!(printed.contains("layout = \"bytes\""), "{printed}");
    }
}
