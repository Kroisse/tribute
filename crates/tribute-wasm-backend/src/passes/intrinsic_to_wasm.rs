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
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol};

use crate::type_converter::wasm_type_converter;

// Constants for Bytes struct layout (must match emit.rs)
const BYTES_ARRAY_IDX: u32 = 1;
const BYTES_STRUCT_IDX: u32 = 2;
const BYTES_DATA_FIELD: u32 = 0; // ref (array i8)
const BYTES_OFFSET_FIELD: u32 = 1; // i32
const BYTES_LEN_FIELD: u32 = 2; // i32

/// Extracted Bytes struct fields: (data, offset, len) values.
struct BytesFields<'db> {
    data: trunk_ir::Value<'db>,
    offset: trunk_ir::Value<'db>,
    len: trunk_ir::Value<'db>,
}

/// Extract (data, offset, len) fields from a Bytes struct value.
///
/// Returns the extracted field values and the operations that produced them.
fn extract_bytes_fields<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    bytes_value: trunk_ir::Value<'db>,
) -> (BytesFields<'db>, Vec<Operation<'db>>) {
    let i32_ty = core::I32::new(db).as_type();
    let i8_ty = core::I8::new(db).as_type();
    let array_ref_ty = core::Ref::new(db, core::Array::new(db, i8_ty).as_type(), false).as_type();

    let get_data = wasm::struct_get(
        db,
        location,
        bytes_value,
        array_ref_ty,
        BYTES_STRUCT_IDX,
        BYTES_DATA_FIELD,
    );

    let get_offset = wasm::struct_get(
        db,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_OFFSET_FIELD,
    );

    let get_len = wasm::struct_get(
        db,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_LEN_FIELD,
    );

    let fields = BytesFields {
        data: get_data.result(db),
        offset: get_offset.result(db),
        len: get_len.result(db),
    };

    let ops = vec![
        get_data.operation(),
        get_offset.operation(),
        get_len.operation(),
    ];

    (fields, ops)
}

/// Result of intrinsic analysis - tracks WASI needs and data segment allocations.
#[salsa::tracked]
pub struct IntrinsicAnalysis<'db> {
    /// Whether fd_write import is needed.
    pub needs_fd_write: bool,
    /// Iovec allocations: (ptr, len) -> offset in data segment.
    /// Using Vec for salsa compatibility.
    #[returns(ref)]
    pub iovec_allocations: Vec<(u32, u32, u32)>,
    /// Offset of nwritten buffer (if any intrinsics need it).
    pub nwritten_offset: Option<u32>,
    /// Total size of runtime data segments.
    pub total_size: u32,
}

impl<'db> IntrinsicAnalysis<'db> {
    /// Look up iovec offset for given (ptr, len) pair.
    pub fn iovec_offset(&self, db: &'db dyn salsa::Database, ptr: u32, len: u32) -> Option<u32> {
        self.iovec_allocations(db)
            .iter()
            .find(|(p, l, _)| *p == ptr && *l == len)
            .map(|(_, _, offset)| *offset)
    }
}

/// Analyze a module to collect intrinsic calls and allocate runtime data segments.
/// Note: This is not a salsa::tracked function because base_offset is a runtime value.
pub fn analyze_intrinsics<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    base_offset: u32,
) -> IntrinsicAnalysis<'db> {
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
    fn visit_op<'db>(
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        needs_fd_write: &mut bool,
        iovec_map: &mut HashMap<(u32, u32), u32>,
        iovec_allocations: &mut Vec<(u32, u32, u32)>,
        next_offset: &mut u32,
    ) {
        // Check for wasm.call to __print_line
        if let Ok(call) = wasm::Call::from_operation(db, *op)
            && call.callee(db).last_segment() == Symbol::new("__print_line")
            && let Some(arg) = op.operands(db).first()
            && let Some((ptr, len)) = get_literal_info(db, *arg)
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

        // Recurse into regions
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                for nested_op in block.operations(db).iter() {
                    visit_op(
                        db,
                        nested_op,
                        needs_fd_write,
                        iovec_map,
                        iovec_allocations,
                        next_offset,
                    );
                }
            }
        }
    }

    // Walk all operations in module body
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            visit_op(
                db,
                op,
                &mut needs_fd_write,
                &mut iovec_map,
                &mut iovec_allocations,
                &mut next_offset,
            );
        }
    }

    // Allocate nwritten buffer if needed
    let nwritten_offset = if needs_fd_write {
        let offset = align_to(next_offset, 4);
        next_offset = offset + 4;
        Some(offset)
    } else {
        None
    };

    IntrinsicAnalysis::new(
        db,
        needs_fd_write,
        iovec_allocations,
        nwritten_offset,
        next_offset - base_offset,
    )
}

/// Get literal pointer and length from a value's defining operation.
fn get_literal_info(db: &dyn salsa::Database, value: trunk_ir::Value<'_>) -> Option<(u32, u32)> {
    let def = value.def(db);
    let trunk_ir::ValueDef::OpResult(op) = def else {
        return None;
    };
    if op.dialect(db) != wasm::DIALECT_NAME() {
        return None;
    }
    if op.name(db) != Symbol::new("i32_const") {
        return None;
    }
    let attrs = op.attributes(db);
    let Attribute::IntBits(ptr) = attrs.get(&Symbol::new("value"))? else {
        return None;
    };
    let Attribute::IntBits(len) = attrs.get(&Symbol::new("literal_len"))? else {
        return None;
    };
    let ptr_u32 = u32::try_from(*ptr).ok()?;
    let len_u32 = u32::try_from(*len).ok()?;
    Some((ptr_u32, len_u32))
}

/// Lower intrinsic calls using pre-computed analysis.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    analysis: IntrinsicAnalysis<'db>,
) -> Module<'db> {
    let mut applicator = PatternApplicator::new(wasm_type_converter());

    // Add __print_line pattern if needed
    if analysis.needs_fd_write(db) {
        let iovec_allocations = analysis.iovec_allocations(db).clone();
        let nwritten_offset = analysis.nwritten_offset(db);
        applicator =
            applicator.add_pattern(PrintLinePattern::new(iovec_allocations, nwritten_offset));
    }

    // Always add Bytes intrinsic patterns
    applicator = applicator
        .add_pattern(BytesLenPattern)
        .add_pattern(BytesGetOrPanicPattern)
        .add_pattern(BytesSliceOrPanicPattern)
        .add_pattern(BytesConcatPattern);

    // No specific conversion target - intrinsic lowering is a dialect transformation
    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
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

impl<'db> RewritePattern<'db> for PrintLinePattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        // Check if this is wasm.call to __print_line
        let Ok(call_op) = wasm::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        if call_op.callee(db).last_segment() != Symbol::new("__print_line") {
            return RewriteResult::Unchanged;
        }

        // Get the string literal argument
        let operands = op.operands(db);
        let Some(arg) = operands.first().copied() else {
            return RewriteResult::Unchanged;
        };
        let Some((ptr, len)) = get_literal_info(db, arg) else {
            return RewriteResult::Unchanged;
        };

        // Look up allocated offsets
        let Some(iovec_offset) = self.lookup_iovec(ptr, len) else {
            return RewriteResult::Unchanged;
        };
        let Some(nwritten_offset) = self.nwritten_offset else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Generate fd_write call sequence:
        // fd_const = wasm.i32_const(1)  // stdout
        // iovec_const = wasm.i32_const(iovec_offset)
        // iovec_len_const = wasm.i32_const(1)  // one iovec entry
        // nwritten_const = wasm.i32_const(nwritten_offset)
        // result = wasm.call(fd_write, fd_const, iovec_const, iovec_len_const, nwritten_const)
        // wasm.drop(result)

        let fd_const = wasm::i32_const(db, location, i32_ty, 1); // stdout
        let iovec_const = wasm::i32_const(db, location, i32_ty, iovec_offset as i32);
        let iovec_len_const = wasm::i32_const(db, location, i32_ty, 1); // one iovec entry
        let nwritten_const = wasm::i32_const(db, location, i32_ty, nwritten_offset as i32);

        let call = wasm::call(
            db,
            location,
            vec![
                fd_const.result(db),
                iovec_const.result(db),
                iovec_len_const.result(db),
                nwritten_const.result(db),
            ],
            vec![i32_ty],
            Symbol::new("fd_write"),
        );

        let drop_op = wasm::drop(db, location, call.result(db, 0));

        // Use Expand to emit all operations
        let results = op.results(db);
        if results.is_empty()
            || (results.len() == 1
                && results[0].dialect(db) == core::DIALECT_NAME()
                && results[0].name(db) == Symbol::new("nil"))
        {
            // Void: emit operations and drop the fd_write result
            RewriteResult::Expand(vec![
                fd_const.operation(),
                iovec_const.operation(),
                iovec_len_const.operation(),
                nwritten_const.operation(),
                call.operation(),
                drop_op.operation(),
            ])
        } else {
            // Non-void: emit operations, call result becomes the replacement value
            RewriteResult::Expand(vec![
                fd_const.operation(),
                iovec_const.operation(),
                iovec_len_const.operation(),
                nwritten_const.operation(),
                call.operation(),
            ])
        }
    }
}

// =============================================================================
// Bytes intrinsic patterns
// =============================================================================

/// Check if operation is a wasm.call to a `__bytes_*` intrinsic.
fn is_bytes_intrinsic_call<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    intrinsic_name: &'static str,
) -> bool {
    let Ok(call) = wasm::Call::from_operation(db, *op) else {
        return false;
    };
    let callee = call.callee(db);
    // Check if callee is "__bytes_xxx"
    callee.last_segment() == Symbol::new(intrinsic_name)
}

/// Pattern for `__bytes_len(bytes)` -> `struct.get $bytes 2` + `i64.extend_i32_u`
struct BytesLenPattern;

impl<'db> RewritePattern<'db> for BytesLenPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        if !is_bytes_intrinsic_call(db, op, "__bytes_len") {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);
        let Some(bytes_ref) = operands.first().copied() else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        // struct.get to get len field (field 2)
        let get_len = wasm::struct_get(
            db,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_LEN_FIELD,
        );

        // Extend i32 to i64 (Int type in Tribute is i64)
        let extend = wasm::i64_extend_i32_u(db, location, get_len.result(db), i64_ty);

        RewriteResult::Expand(vec![get_len.operation(), extend.operation()])
    }
}

/// Pattern for `Bytes::get_or_panic(bytes, index)` -> array access with offset
struct BytesGetOrPanicPattern;

impl<'db> RewritePattern<'db> for BytesGetOrPanicPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        if !is_bytes_intrinsic_call(db, op, "__bytes_get_or_panic") {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);
        if operands.len() < 2 {
            return RewriteResult::Unchanged;
        }
        let bytes_ref = operands[0];
        let index = operands[1]; // i64

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let i8_ty = core::I8::new(db).as_type();

        // Get data array ref (field 0)
        let array_ref_ty =
            core::Ref::new(db, core::Array::new(db, i8_ty).as_type(), false).as_type();
        let get_data = wasm::struct_get(
            db,
            location,
            bytes_ref,
            array_ref_ty,
            BYTES_STRUCT_IDX,
            BYTES_DATA_FIELD,
        );

        // Get offset (field 1)
        let get_offset = wasm::struct_get(
            db,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_OFFSET_FIELD,
        );

        // Wrap index to i32
        let index_i32 = wasm::i32_wrap_i64(db, location, index, i32_ty);

        // Add offset to index: actual_index = offset + index
        let add_offset = wasm::i32_add(
            db,
            location,
            get_offset.result(db),
            index_i32.result(db),
            i32_ty,
        );

        // array.get_u (unsigned extend to i32, for byte values 0-255)
        let array_get = wasm::array_get_u(
            db,
            location,
            get_data.result(db),
            add_offset.result(db),
            i32_ty,
            BYTES_ARRAY_IDX,
        );

        // Extend i32 to i64 (unsigned, Int type in Tribute is i64)
        let extend = wasm::i64_extend_i32_u(db, location, array_get.result(db), i64_ty);

        RewriteResult::Expand(vec![
            get_data.operation(),
            get_offset.operation(),
            index_i32.operation(),
            add_offset.operation(),
            array_get.operation(),
            extend.operation(),
        ])
    }
}

/// Pattern for `Bytes::slice_or_panic(bytes, start, end)` -> new struct with adjusted offset/len
struct BytesSliceOrPanicPattern;

impl<'db> RewritePattern<'db> for BytesSliceOrPanicPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        if !is_bytes_intrinsic_call(db, op, "__bytes_slice_or_panic") {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);
        if operands.len() < 3 {
            return RewriteResult::Unchanged;
        }
        let bytes_ref = operands[0];
        let start = operands[1]; // i64
        let end = operands[2]; // i64

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i8_ty = core::I8::new(db).as_type();
        let bytes_ty = core::Bytes::new(db).as_type();

        // Get data array ref (field 0) - shared, zero-copy
        let array_ref_ty =
            core::Ref::new(db, core::Array::new(db, i8_ty).as_type(), false).as_type();
        let get_data = wasm::struct_get(
            db,
            location,
            bytes_ref,
            array_ref_ty,
            BYTES_STRUCT_IDX,
            BYTES_DATA_FIELD,
        );

        // Get current offset (field 1)
        let get_offset = wasm::struct_get(
            db,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_OFFSET_FIELD,
        );

        // Wrap start and end to i32
        let start_i32 = wasm::i32_wrap_i64(db, location, start, i32_ty);

        let end_i32 = wasm::i32_wrap_i64(db, location, end, i32_ty);

        // new_offset = offset + start
        let new_offset = wasm::i32_add(
            db,
            location,
            get_offset.result(db),
            start_i32.result(db),
            i32_ty,
        );

        // new_len = end - start
        let new_len = wasm::i32_sub(
            db,
            location,
            end_i32.result(db),
            start_i32.result(db),
            i32_ty,
        );

        // struct.new to create new Bytes (shares the underlying array)
        let struct_new = wasm::struct_new(
            db,
            location,
            vec![
                get_data.result(db),
                new_offset.result(db),
                new_len.result(db),
            ],
            bytes_ty,
            BYTES_STRUCT_IDX,
        );

        RewriteResult::Expand(vec![
            get_data.operation(),
            get_offset.operation(),
            start_i32.operation(),
            end_i32.operation(),
            new_offset.operation(),
            new_len.operation(),
            struct_new.operation(),
        ])
    }
}

/// Pattern for `Bytes::concat(left, right)` -> allocate new array and copy both
struct BytesConcatPattern;

impl<'db> RewritePattern<'db> for BytesConcatPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        if !is_bytes_intrinsic_call(db, op, "__bytes_concat") {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);
        if operands.len() < 2 {
            return RewriteResult::Unchanged;
        }
        let left = operands[0];
        let right = operands[1];

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i8_ty = core::I8::new(db).as_type();
        let bytes_ty = core::Bytes::new(db).as_type();
        let array_ref_ty =
            core::Ref::new(db, core::Array::new(db, i8_ty).as_type(), false).as_type();

        // Extract fields from left and right Bytes structs
        let (left_fields, left_ops) = extract_bytes_fields(db, location, left);
        let (right_fields, right_ops) = extract_bytes_fields(db, location, right);

        // Calculate total_len = left.len + right.len
        let total_len = wasm::i32_add(db, location, left_fields.len, right_fields.len, i32_ty);

        // Allocate new array: array_new_default(total_len)
        let new_array = wasm::array_new_default(
            db,
            location,
            total_len.result(db),
            array_ref_ty,
            BYTES_ARRAY_IDX,
        );

        // Copy left bytes: array_copy(new_arr, 0, left.data, left.offset, left.len)
        let zero = wasm::i32_const(db, location, i32_ty, 0);

        let copy_left = wasm::array_copy(
            db,
            location,
            new_array.result(db),
            zero.result(db),
            left_fields.data,
            left_fields.offset,
            left_fields.len,
            BYTES_ARRAY_IDX,
            BYTES_ARRAY_IDX,
        );

        // Copy right bytes: array_copy(new_arr, left.len, right.data, right.offset, right.len)
        let copy_right = wasm::array_copy(
            db,
            location,
            new_array.result(db),
            left_fields.len,
            right_fields.data,
            right_fields.offset,
            right_fields.len,
            BYTES_ARRAY_IDX,
            BYTES_ARRAY_IDX,
        );

        // Create new Bytes struct: struct_new(new_arr, 0, total_len)
        let struct_new = wasm::struct_new(
            db,
            location,
            vec![new_array.result(db), zero.result(db), total_len.result(db)],
            bytes_ty,
            BYTES_STRUCT_IDX,
        );

        // Combine all operations in order
        let mut ops = Vec::with_capacity(left_ops.len() + right_ops.len() + 6);
        ops.extend(left_ops);
        ops.extend(right_ops);
        ops.push(total_len.operation());
        ops.push(new_array.operation());
        ops.push(zero.operation());
        ops.push(copy_left.operation());
        ops.push(copy_right.operation());
        ops.push(struct_new.operation());

        RewriteResult::Expand(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockArg, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_print_line_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        // Create string constant (offset 0 in data section)
        let string_const = wasm::i32_const(db, location, i32_ty, 0);

        // Create __print_line call
        let print_line = wasm::call(
            db,
            location,
            vec![string_const.result(db)],
            vec![nil_ty],
            Symbol::new("__print_line"),
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![string_const.as_operation(), print_line.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn analyze_and_check(db: &dyn salsa::Database, module: Module<'_>) -> (bool, usize, bool) {
        let analysis = analyze_intrinsics(db, module, 0);
        (
            analysis.needs_fd_write(db),
            analysis.iovec_allocations(db).len(),
            analysis.nwritten_offset(db).is_some(),
        )
    }

    #[salsa_test]
    #[ignore = "TODO: Update after DataRegistry integration - literal_len attribute removed"]
    fn test_intrinsic_analysis(db: &salsa::DatabaseImpl) {
        let module = make_print_line_module(db);
        let (needs_fd_write, iovec_count, has_nwritten) = analyze_and_check(db, module);

        assert!(needs_fd_write);
        assert_eq!(iovec_count, 1);
        assert!(has_nwritten);
    }

    #[salsa::tracked]
    fn lower_and_check(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let analysis = analyze_intrinsics(db, module, 0);
        let lowered = lower(db, module, analysis);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    /// Extract callee names from all wasm.call operations in the module.
    #[salsa::tracked]
    fn extract_callees(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let analysis = analyze_intrinsics(db, module, 0);
        let lowered = lower(db, module, analysis);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter()
            .filter_map(|op| {
                let Ok(call) = wasm::Call::from_operation(db, *op) else {
                    return None;
                };
                Some(call.callee(db).last_segment().to_string())
            })
            .collect()
    }

    #[salsa_test]
    #[ignore = "TODO: Update after DataRegistry integration - literal_len attribute removed"]
    fn test_print_line_to_fd_write(db: &salsa::DatabaseImpl) {
        let module = make_print_line_module(db);
        let op_names = lower_and_check(db, module);
        let callees = extract_callees(db, module);

        // Should have wasm.call operations
        assert!(op_names.iter().any(|n| n == "wasm.call"));
        // The call should be to fd_write, not __print_line
        assert!(callees.contains(&"fd_write".to_string()));
        assert!(!callees.contains(&"__print_line".to_string()));
    }

    // === Bytes intrinsic tests ===

    /// Create an intrinsic name for __bytes_* functions
    fn bytes_intrinsic_name(method: &'static str) -> Symbol {
        Symbol::from_dynamic(&format!("__bytes_{}", method))
    }

    #[salsa::tracked]
    fn make_bytes_len_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let bytes_ty = core::Bytes::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        // Create a fake bytes value (block arg)
        let block_id = BlockId::fresh();
        let bytes_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);

        // Create Bytes::len call
        let len_call = wasm::call(
            db,
            location,
            vec![bytes_val],
            vec![i64_ty],
            bytes_intrinsic_name("len"),
        )
        .as_operation();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![BlockArg::of_type(db, bytes_ty)],
            idvec![len_call],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_bytes_len_to_struct_get(db: &salsa::DatabaseImpl) {
        let module = make_bytes_len_module(db);
        let op_names = lower_and_check(db, module);

        // Should have struct_get and extend operations, not a wasm.call
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(op_names.iter().any(|n| n == "wasm.i64_extend_i32_u"));
        // No Bytes::len call should remain
        let callees = extract_callees(db, module);
        assert!(!callees.iter().any(|n| n == "__bytes_len"));
    }

    #[salsa::tracked]
    fn make_bytes_get_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let bytes_ty = core::Bytes::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        let block_id = BlockId::fresh();
        let bytes_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);
        let index_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 1);

        // Create Bytes::get_or_panic call
        let get_call = wasm::call(
            db,
            location,
            vec![bytes_val, index_val],
            vec![i64_ty],
            bytes_intrinsic_name("get_or_panic"),
        )
        .as_operation();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![
                BlockArg::of_type(db, bytes_ty),
                BlockArg::of_type(db, i64_ty)
            ],
            idvec![get_call],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_bytes_get_to_array_get(db: &salsa::DatabaseImpl) {
        let module = make_bytes_get_module(db);
        let op_names = lower_and_check(db, module);

        // Should have struct_get (for data and offset), i32_add, and array_get_u
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(op_names.iter().any(|n| n == "wasm.i32_add"));
        assert!(op_names.iter().any(|n| n == "wasm.array_get_u"));
        // No Bytes::get_or_panic call should remain
        let callees = extract_callees(db, module);
        assert!(!callees.iter().any(|n| n == "__bytes_get_or_panic"));
    }

    #[salsa::tracked]
    fn make_bytes_slice_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let bytes_ty = core::Bytes::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        let block_id = BlockId::fresh();
        let bytes_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);
        let start_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 1);
        let end_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 2);

        // Create Bytes::slice_or_panic call
        let slice_call = wasm::call(
            db,
            location,
            vec![bytes_val, start_val, end_val],
            vec![bytes_ty],
            bytes_intrinsic_name("slice_or_panic"),
        )
        .as_operation();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![
                BlockArg::of_type(db, bytes_ty),
                BlockArg::of_type(db, i64_ty),
                BlockArg::of_type(db, i64_ty)
            ],
            idvec![slice_call],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_bytes_slice_to_struct_new(db: &salsa::DatabaseImpl) {
        let module = make_bytes_slice_module(db);
        let op_names = lower_and_check(db, module);

        // Should have struct_get, i32_add, i32_sub, and struct_new
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(op_names.iter().any(|n| n == "wasm.i32_add"));
        assert!(op_names.iter().any(|n| n == "wasm.i32_sub"));
        assert!(op_names.iter().any(|n| n == "wasm.struct_new"));
        // No Bytes::slice_or_panic call should remain
        let callees = extract_callees(db, module);
        assert!(!callees.iter().any(|n| n == "__bytes_slice_or_panic"));
    }

    #[salsa::tracked]
    fn make_bytes_concat_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let bytes_ty = core::Bytes::new(db).as_type();

        let block_id = BlockId::fresh();
        let left_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);
        let right_val = trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 1);

        // Create Bytes::concat call
        let concat_call = wasm::call(
            db,
            location,
            vec![left_val, right_val],
            vec![bytes_ty],
            bytes_intrinsic_name("concat"),
        )
        .as_operation();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![
                BlockArg::of_type(db, bytes_ty),
                BlockArg::of_type(db, bytes_ty)
            ],
            idvec![concat_call],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_bytes_concat_to_array_copy(db: &salsa::DatabaseImpl) {
        let module = make_bytes_concat_module(db);
        let op_names = lower_and_check(db, module);

        // Should have struct_get, i32_add, array_new_default, array_copy, and struct_new
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(op_names.iter().any(|n| n == "wasm.i32_add"));
        assert!(op_names.iter().any(|n| n == "wasm.array_new_default"));
        assert!(op_names.iter().any(|n| n == "wasm.array_copy"));
        assert!(op_names.iter().any(|n| n == "wasm.struct_new"));
        // No Bytes::concat call should remain
        let callees = extract_callees(db, module);
        assert!(!callees.iter().any(|n| n == "__bytes_concat"));
    }
}
