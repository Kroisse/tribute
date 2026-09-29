//! Identifiers of compiler-owned runtime storage layouts.
//!
//! A compiler-owned layout carries the reserved type attribute
//! [`LAYOUT_ATTR`] with one of the symbols below. Only the layout's canonical
//! constructor attaches it, so passes after the representation/ABI boundary
//! identify these layouts by the attribute alone, never by struct name, field
//! shape, or an erased reference type.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
pub use trunk_ir::types::LAYOUT_ATTR;

/// Closure storage: a function reference and an environment.
pub const CLOSURE: &str = "closure";
/// An evidence marker: one handler's ability id, prompt, and dispatch closures.
pub const EVIDENCE_MARKER: &str = "evidence_marker";
/// The evidence array: markers sorted by ability id.
pub const EVIDENCE: &str = "evidence";

/// Whether `ty` carries the runtime layout identifier `layout`.
pub fn has_runtime_layout(ctx: &IrContext, ty: TypeRef, layout: &str) -> bool {
    ctx.get_type(ty).attrs.get_symbol(LAYOUT_ATTR) == Some(Symbol::from_dynamic(layout))
}
