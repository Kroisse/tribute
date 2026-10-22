//! Identifiers of compiler-owned runtime storage layouts.
//!
//! A compiler-owned layout carries the reserved type attribute
//! [`LAYOUT_ATTR`] with one of the string values below. Only the layout's canonical
//! constructor attaches it, so passes after the representation/ABI boundary
//! identify these layouts by the attribute alone, never by struct name, field
//! shape, or an erased reference type.

use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
pub use trunk_ir::types::LAYOUT_ATTR;

/// The closure base: the code reference alone. Every closure value has this
/// storage type, and a capture-free closure is allocated with it.
pub const CLOSURE: &str = "closure";
/// A closure with captures: the closure base's fields followed by the captures,
/// with [`CLOSURE_BASE_ATTR`] naming that base.
pub const CLOSURE_CAPTURES: &str = "closure_captures";
/// The type attribute naming the closure base of a [`CLOSURE_CAPTURES`] layout,
/// whose fields are a prefix of the capture layout's fields.
pub const CLOSURE_BASE_ATTR: &str = "closure_base";
/// Wasm `Bytes` storage: a backing array, a start offset, and a length.
pub const BYTES: &str = "bytes";
/// The byte array a Wasm `Bytes` points into.
pub const BYTES_DATA: &str = "bytes_data";
/// A boxed `Float`: one `f64` field, stored where a uniform reference is
/// expected.
pub const BOXED_F64: &str = "boxed_f64";
/// The Wasm supertype of every user struct and variant: its runtime type
/// descriptor field alone. Builtin layouts, arrays, and boxed scalars are not
/// its subtypes.
pub const DESCRIBED: &str = "described";

/// The closure base whose code field has the representation `code`.
pub fn closure_base_layout(ctx: &mut IrContext, code: TypeRef) -> TypeRef {
    let mut attrs = trunk_ir::types::AttributeMap::new();
    let layout = ctx.string_attr(CLOSURE);
    attrs.view_mut(ctx).insert(LAYOUT_ATTR, layout);
    crate::dialect::adt::Struct::from_fields(ctx, "_closure", [("code", code)], attrs).as_type_ref()
}

/// The closure base of the source-logical and target-neutral IR, whose code
/// field is a function table index.
pub fn closure_base(ctx: &mut IrContext) -> TypeRef {
    let code = trunk_ir::types::TypeDataBuilder::new("core.i32").intern(ctx);
    closure_base_layout(ctx, code)
}

/// Whether `ty` carries the runtime layout identifier `layout`.
pub fn has_runtime_layout(ctx: &IrContext, ty: TypeRef, layout: &str) -> bool {
    ctx.type_attrs(ty).get_str(LAYOUT_ATTR) == Some(layout)
}
