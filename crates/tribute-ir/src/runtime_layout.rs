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

/// Closure storage: a function reference and an environment.
pub const CLOSURE: &str = "closure";
/// An evidence marker: one handler's ability id, prompt, and dispatch closures.
pub const EVIDENCE_MARKER: &str = "evidence_marker";
/// The evidence array: markers sorted by ability id.
pub const EVIDENCE: &str = "evidence";
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

/// Whether `ty` carries the runtime layout identifier `layout`.
pub fn has_runtime_layout(ctx: &IrContext, ty: TypeRef, layout: &str) -> bool {
    ctx.get_type(ty).attrs.get_str(ctx, LAYOUT_ATTR) == Some(layout)
}
