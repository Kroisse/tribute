//! Lower intrinsic calls to WASM operations.
//!
//! This pass transforms high-level intrinsic calls to low-level WASM instructions:
//! - `__print_line` -> WASI `fd_write` call
//! - `__bytes_len`, `__bytes_get_or_panic`, etc. -> WasmGC struct/array operations
//!
//! Two-phase approach for WASI intrinsics:
//! 1. Analysis: Collect all intrinsic calls and allocate runtime data segments
//! 2. Transform: Replace intrinsic calls with WASM instruction sequences

use std::collections::HashMap;

use tribute_ir::ModulePathExt;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::ir::Symbol;

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
    location: trunk_ir::arena::types::Location,
    bytes_value: ValueRef,
) -> (BytesFields, Vec<OpRef>) {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let i8_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
    let array_ty = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
            .param(i8_ty)
            .build(),
    );
    let array_ref_ty = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ref"))
            .param(array_ty)
            .build(),
    );

    let get_data = arena_wasm::struct_get(
        ctx,
        location,
        bytes_value,
        array_ref_ty,
        BYTES_STRUCT_IDX,
        BYTES_DATA_FIELD,
    );

    let get_offset = arena_wasm::struct_get(
        ctx,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_OFFSET_FIELD,
    );

    let get_len = arena_wasm::struct_get(
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

/// Result of intrinsic analysis - tracks WASI needs and data segment allocations.
pub struct IntrinsicAnalysis {
    /// Whether fd_write import is needed.
    pub needs_fd_write: bool,
    /// Iovec allocations: (ptr, len) -> offset in data segment.
    pub iovec_allocations: Vec<(u32, u32, u32)>,
    /// Offset of nwritten buffer (if any intrinsics need it).
    pub nwritten_offset: Option<u32>,
    /// Total size of runtime data segments.
    pub total_size: u32,
}

impl IntrinsicAnalysis {
    /// Look up iovec offset for given (ptr, len) pair.
    pub fn iovec_offset(&self, ptr: u32, len: u32) -> Option<u32> {
        self.iovec_allocations
            .iter()
            .find(|(p, l, _)| *p == ptr && *l == len)
            .map(|(_, _, offset)| *offset)
    }
}

/// Walk all operations in a region recursively.
fn walk_ops_in_region(
    ctx: &IrContext,
    region: RegionRef,
    callback: &mut impl FnMut(&IrContext, OpRef),
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            callback(ctx, op);
            for &nested in ctx.op(op).regions.iter() {
                walk_ops_in_region(ctx, nested, callback);
            }
        }
    }
}

/// Get literal pointer and length from a value's defining operation.
fn get_literal_info(ctx: &IrContext, value: ValueRef) -> Option<(u32, u32)> {
    let def = ctx.value_def(value);
    let trunk_ir::arena::ValueDef::OpResult(op, _) = def else {
        return None;
    };
    let data = ctx.op(op);
    if data.dialect != arena_wasm::DIALECT_NAME() {
        return None;
    }
    if data.name != Symbol::new("i32_const") {
        return None;
    }
    let ArenaAttribute::IntBits(ptr) = data.attributes.get(&Symbol::new("value"))? else {
        return None;
    };
    let ArenaAttribute::IntBits(len) = data.attributes.get(&Symbol::new("literal_len"))? else {
        return None;
    };
    let ptr_u32 = u32::try_from(*ptr).ok()?;
    let len_u32 = u32::try_from(*len).ok()?;
    Some((ptr_u32, len_u32))
}

/// Analyze a module to collect intrinsic calls and allocate runtime data segments.
pub fn analyze_intrinsics(
    ctx: &IrContext,
    module: ArenaModule,
    base_offset: u32,
) -> IntrinsicAnalysis {
    let mut needs_fd_write = false;
    let mut iovec_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut iovec_allocations: Vec<(u32, u32, u32)> = Vec::new();
    let mut next_offset = base_offset;

    // Align to 4-byte boundary
    fn align_to(value: u32, align: u32) -> u32 {
        if align == 0 {
            return value;
        }
        value.div_ceil(align) * align
    }

    // Visit operations to find __print_line calls with literal args
    fn visit_op(
        ctx: &IrContext,
        op: OpRef,
        needs_fd_write: &mut bool,
        iovec_map: &mut HashMap<(u32, u32), u32>,
        iovec_allocations: &mut Vec<(u32, u32, u32)>,
        next_offset: &mut u32,
    ) {
        // Check for wasm.call to __print_line
        if let Ok(call) = arena_wasm::Call::from_op(ctx, op) {
            let callee = call.callee(ctx);
            if callee.last_segment() == Symbol::new("__print_line") {
                let operands = ctx.op_operands(op);
                if let Some(&arg) = operands.first()
                    && let Some((ptr, len)) = get_literal_info(ctx, arg)
                {
                    *needs_fd_write = true;

                    // Allocate iovec if not already done
                    iovec_map.entry((ptr, len)).or_insert_with(|| {
                        let offset = align_to(*next_offset, 4);
                        iovec_allocations.push((ptr, len, offset));
                        *next_offset = offset + 8; // iovec is 8 bytes (ptr + len)
                        offset
                    });
                }
            }
        }
    }

    // Walk all operations in module body
    if let Some(body) = module.body(ctx) {
        walk_ops_in_region(ctx, body, &mut |ctx, op| {
            visit_op(
                ctx,
                op,
                &mut needs_fd_write,
                &mut iovec_map,
                &mut iovec_allocations,
                &mut next_offset,
            );
        });
    }

    // Allocate nwritten buffer if needed
    let nwritten_offset = if needs_fd_write {
        let offset = align_to(next_offset, 4);
        next_offset = offset + 4;
        Some(offset)
    } else {
        None
    };

    IntrinsicAnalysis {
        needs_fd_write,
        iovec_allocations,
        nwritten_offset,
        total_size: next_offset - base_offset,
    }
}

/// Lower intrinsic calls using pre-computed analysis.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, analysis: &IntrinsicAnalysis) {
    let mut applicator = PatternApplicator::new(ArenaTypeConverter::new());

    // Add __print_line pattern if needed
    if analysis.needs_fd_write {
        let iovec_allocations = analysis.iovec_allocations.clone();
        let nwritten_offset = analysis.nwritten_offset;
        applicator =
            applicator.add_pattern(PrintLinePattern::new(iovec_allocations, nwritten_offset));
    }

    // Always add Bytes intrinsic patterns
    applicator = applicator
        .add_pattern(BytesLenPattern)
        .add_pattern(BytesGetOrPanicPattern)
        .add_pattern(BytesSliceOrPanicPattern)
        .add_pattern(BytesConcatPattern);

    applicator.apply_partial(ctx, module);
}

/// Pattern for `wasm.call(__print_line)` -> `fd_write` sequence
struct PrintLinePattern {
    iovec_allocations: Vec<(u32, u32, u32)>,
    nwritten_offset: Option<u32>,
}

impl PrintLinePattern {
    fn new(iovec_allocations: Vec<(u32, u32, u32)>, nwritten_offset: Option<u32>) -> Self {
        Self {
            iovec_allocations,
            nwritten_offset,
        }
    }

    fn lookup_iovec(&self, ptr: u32, len: u32) -> Option<u32> {
        self.iovec_allocations
            .iter()
            .find(|(p, l, _)| *p == ptr && *l == len)
            .map(|(_, _, offset)| *offset)
    }
}

impl ArenaRewritePattern for PrintLinePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        // Check if this is wasm.call to __print_line
        let Ok(call_op) = arena_wasm::Call::from_op(ctx, op) else {
            return false;
        };

        if call_op.callee(ctx).last_segment() != Symbol::new("__print_line") {
            return false;
        }

        // Get the string literal argument
        let operands = ctx.op_operands(op).to_vec();
        let Some(arg) = operands.first().copied() else {
            return false;
        };
        let Some((ptr, len)) = get_literal_info(ctx, arg) else {
            return false;
        };

        // Look up allocated offsets
        let Some(iovec_offset) = self.lookup_iovec(ptr, len) else {
            return false;
        };
        let Some(nwritten_offset) = self.nwritten_offset else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        // Generate fd_write call sequence:
        // fd_const = wasm.i32_const(1)  // stdout
        // iovec_const = wasm.i32_const(iovec_offset)
        // iovec_len_const = wasm.i32_const(1)  // one iovec entry
        // nwritten_const = wasm.i32_const(nwritten_offset)
        // result = wasm.call(fd_write, fd_const, iovec_const, iovec_len_const, nwritten_const)
        // wasm.drop(result)

        let fd_const = arena_wasm::i32_const(ctx, location, i32_ty, 1); // stdout
        let iovec_const = arena_wasm::i32_const(ctx, location, i32_ty, iovec_offset as i32);
        let iovec_len_const = arena_wasm::i32_const(ctx, location, i32_ty, 1); // one iovec entry
        let nwritten_const = arena_wasm::i32_const(ctx, location, i32_ty, nwritten_offset as i32);

        let call = arena_wasm::call(
            ctx,
            location,
            vec![
                fd_const.result(ctx),
                iovec_const.result(ctx),
                iovec_len_const.result(ctx),
                nwritten_const.result(ctx),
            ],
            vec![i32_ty],
            Symbol::new("fd_write"),
        );

        let drop_op = arena_wasm::drop(ctx, location, call.results(ctx)[0]);

        // Emit all operations
        let result_types = ctx.op_result_types(op).to_vec();
        let nil_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());
        if result_types.is_empty() || (result_types.len() == 1 && result_types[0] == nil_ty) {
            // Void: emit operations and drop the fd_write result
            rewriter.insert_op(fd_const.op_ref());
            rewriter.insert_op(iovec_const.op_ref());
            rewriter.insert_op(iovec_len_const.op_ref());
            rewriter.insert_op(nwritten_const.op_ref());
            rewriter.insert_op(call.op_ref());
            rewriter.replace_op(drop_op.op_ref());
        } else {
            // Non-void: emit operations, call result becomes the replacement value
            rewriter.insert_op(fd_const.op_ref());
            rewriter.insert_op(iovec_const.op_ref());
            rewriter.insert_op(iovec_len_const.op_ref());
            rewriter.insert_op(nwritten_const.op_ref());
            rewriter.replace_op(call.op_ref());
        }
        true
    }

    fn name(&self) -> &'static str {
        "PrintLinePattern"
    }
}

// =============================================================================
// Bytes intrinsic patterns
// =============================================================================

/// Check if operation is a wasm.call to a `__bytes_*` intrinsic.
fn is_bytes_intrinsic_call(ctx: &IrContext, op: OpRef, intrinsic_name: &'static str) -> bool {
    let Ok(call) = arena_wasm::Call::from_op(ctx, op) else {
        return false;
    };
    let callee = call.callee(ctx);
    // Check if callee is "__bytes_xxx"
    callee.last_segment() == Symbol::new(intrinsic_name)
}

/// Pattern for `__bytes_len(bytes)` -> `struct.get $bytes 2`
///
/// Returns i32 directly since Nat is mapped to i32.
struct BytesLenPattern;

impl ArenaRewritePattern for BytesLenPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, "__bytes_len") {
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
        let get_len = arena_wasm::struct_get(
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

impl ArenaRewritePattern for BytesGetOrPanicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, "__bytes_get_or_panic") {
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
        let array_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
                .param(i8_ty)
                .build(),
        );
        let array_ref_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ref"))
                .param(array_ty)
                .build(),
        );
        let get_data = arena_wasm::struct_get(
            ctx,
            location,
            bytes_ref,
            array_ref_ty,
            BYTES_STRUCT_IDX,
            BYTES_DATA_FIELD,
        );

        // Get offset (field 1)
        let get_offset = arena_wasm::struct_get(
            ctx,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_OFFSET_FIELD,
        );

        // Add offset to index: actual_index = offset + index
        let add_offset = arena_wasm::i32_add(ctx, location, get_offset.result(ctx), index, i32_ty);

        // array.get_u (unsigned extend to i32, for byte values 0-255)
        let array_get = arena_wasm::array_get_u(
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

/// Pattern for `Bytes::slice_or_panic(bytes, start, end)` -> new struct with adjusted offset/len
///
/// Start and end are i32 (Nat), returns Bytes.
struct BytesSliceOrPanicPattern;

impl ArenaRewritePattern for BytesSliceOrPanicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, "__bytes_slice_or_panic") {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 3 {
            return false;
        }
        let bytes_ref = operands[0];
        let start = operands[1]; // i32 (Nat)
        let end = operands[2]; // i32 (Nat)

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
        let bytes_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bytes")).build());

        // Get data array ref (field 0) - shared, zero-copy
        let array_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
                .param(i8_ty)
                .build(),
        );
        let array_ref_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ref"))
                .param(array_ty)
                .build(),
        );
        let get_data = arena_wasm::struct_get(
            ctx,
            location,
            bytes_ref,
            array_ref_ty,
            BYTES_STRUCT_IDX,
            BYTES_DATA_FIELD,
        );

        // Get current offset (field 1)
        let get_offset = arena_wasm::struct_get(
            ctx,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_OFFSET_FIELD,
        );

        // new_offset = offset + start
        let new_offset = arena_wasm::i32_add(ctx, location, get_offset.result(ctx), start, i32_ty);

        // new_len = end - start
        let new_len = arena_wasm::i32_sub(ctx, location, end, start, i32_ty);

        // struct.new to create new Bytes (shares the underlying array)
        let struct_new = arena_wasm::struct_new(
            ctx,
            location,
            vec![
                get_data.result(ctx),
                new_offset.result(ctx),
                new_len.result(ctx),
            ],
            bytes_ty,
            BYTES_STRUCT_IDX,
        );

        rewriter.insert_op(get_data.op_ref());
        rewriter.insert_op(get_offset.op_ref());
        rewriter.insert_op(new_offset.op_ref());
        rewriter.insert_op(new_len.op_ref());
        rewriter.replace_op(struct_new.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesSliceOrPanicPattern"
    }
}

/// Pattern for `Bytes::concat(left, right)` -> allocate new array and copy both
struct BytesConcatPattern;

impl ArenaRewritePattern for BytesConcatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, "__bytes_concat") {
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
        let bytes_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bytes")).build());
        let array_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
                .param(i8_ty)
                .build(),
        );
        let array_ref_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ref"))
                .param(array_ty)
                .build(),
        );

        // Extract fields from left and right Bytes structs
        let (left_fields, left_ops) = extract_bytes_fields(ctx, location, left);
        let (right_fields, right_ops) = extract_bytes_fields(ctx, location, right);

        // Calculate total_len = left.len + right.len
        let total_len =
            arena_wasm::i32_add(ctx, location, left_fields.len, right_fields.len, i32_ty);

        // Allocate new array: array_new_default(total_len)
        let new_array = arena_wasm::array_new_default(
            ctx,
            location,
            total_len.result(ctx),
            array_ref_ty,
            BYTES_ARRAY_IDX,
        );

        // Copy left bytes: array_copy(new_arr, 0, left.data, left.offset, left.len)
        let zero = arena_wasm::i32_const(ctx, location, i32_ty, 0);

        let copy_left = arena_wasm::array_copy(
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
        let copy_right = arena_wasm::array_copy(
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
        let struct_new = arena_wasm::struct_new(
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
