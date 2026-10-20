//! The native layout of a managed array: `[len: i64] [elements...]` after the
//! RC header, with each element at its native representation size.

use tribute_ir::dialect::adt::layout::type_size_align;
use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use trunk_ir::TypeRef;
use trunk_ir::context::IrContext;
use trunk_ir::rewrite::TypeConverter;

/// Payload offset of the element count.
pub(super) const LEN_OFFSET: i32 = 0;
/// Payload offset of the first element.
pub(super) const ELEMENTS_OFFSET: i32 = 8;
/// The allocation size of an empty array: the RC header and the count.
pub(super) const FIXED_ALLOC_SIZE: u64 = RC_HEADER_SIZE + ELEMENTS_OFFSET as u64;

/// The size of one element of semantic type `element`: a pointer when the
/// array releases its elements, and the element's native representation
/// otherwise.
pub(super) fn element_stride(
    ctx: &IrContext,
    element: TypeRef,
    released: bool,
    type_converter: &TypeConverter,
) -> u32 {
    if released {
        return 8;
    }
    let native = type_converter.convert_type_or_identity(ctx, element);
    type_size_align(ctx, native).0
}
