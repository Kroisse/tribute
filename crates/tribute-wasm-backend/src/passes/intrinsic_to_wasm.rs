//! Lower intrinsic calls to WASM operations.
//!
//! This pass transforms high-level intrinsic calls to low-level WASM instructions:
//! - `print_line` -> WASI `fd_write` call
//! - `Bytes::len`, `Bytes::get_or_panic`, etc. -> WasmGC struct/array operations
//!
//! Two-phase approach for WASI intrinsics:
//! 1. Analysis: Collect all intrinsic calls and allocate runtime data segments
//! 2. Transform: Replace intrinsic calls with WASM instruction sequences

use std::collections::HashMap;

use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, QualifiedName, Symbol, idvec};

// Constants for Bytes struct layout (must match emit.rs)
const BYTES_ARRAY_IDX: u32 = 1;
const BYTES_STRUCT_IDX: u32 = 2;
const BYTES_DATA_FIELD: u32 = 0; // ref (array i8)
const BYTES_OFFSET_FIELD: u32 = 1; // i32
const BYTES_LEN_FIELD: u32 = 2; // i32

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

    // Visit operations to find print_line calls with literal args
    fn visit_op<'db>(
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        needs_fd_write: &mut bool,
        iovec_map: &mut HashMap<(u32, u32), u32>,
        iovec_allocations: &mut Vec<(u32, u32, u32)>,
        next_offset: &mut u32,
    ) {
        // Check for wasm.call to print_line
        if let Ok(call) = wasm::Call::from_operation(db, *op)
            && let Attribute::QualifiedName(callee) = call.callee(db)
            && callee.name() == Symbol::new("print_line")
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
    let mut applicator = PatternApplicator::new();

    // Add print_line pattern if needed
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
        .add_pattern(BytesSliceOrPanicPattern);

    applicator.apply(db, module).module
}

/// Pattern for `wasm.call(print_line)` -> `fd_write` sequence
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

impl RewritePattern for PrintLinePattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
    ) -> RewriteResult<'a> {
        // Check if this is wasm.call to print_line
        let Ok(call_op) = wasm::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let Attribute::QualifiedName(callee) = call_op.callee(db) else {
            return RewriteResult::Unchanged;
        };
        if callee.name() != Symbol::new("print_line") {
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

        let fd_const = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1)); // stdout
        let iovec_const = wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(u64::from(iovec_offset)),
        );
        let iovec_len_const = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1)); // one iovec entry
        let nwritten_const = wasm::i32_const(
            db,
            location,
            i32_ty,
            Attribute::IntBits(u64::from(nwritten_offset)),
        );

        let fd_write_callee = QualifiedName::simple(Symbol::new("fd_write"));
        let call = Operation::of_name(db, location, "wasm.call")
            .operands(trunk_ir::idvec![
                fd_const.result(db),
                iovec_const.result(db),
                iovec_len_const.result(db),
                nwritten_const.result(db),
            ])
            .results(trunk_ir::idvec![i32_ty])
            .attr("callee", Attribute::QualifiedName(fd_write_callee))
            .build();

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
                call,
                drop_op.operation(),
            ])
        } else {
            // Non-void: emit operations, call result becomes the replacement value
            RewriteResult::Expand(vec![
                fd_const.operation(),
                iovec_const.operation(),
                iovec_len_const.operation(),
                nwritten_const.operation(),
                call,
            ])
        }
    }
}

// =============================================================================
// Bytes intrinsic patterns
// =============================================================================

/// Check if operation is a wasm.call to the given Bytes:: method.
fn is_bytes_method_call<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    method: &'static str,
) -> bool {
    let Ok(call) = wasm::Call::from_operation(db, *op) else {
        return false;
    };
    let Attribute::QualifiedName(callee) = call.callee(db) else {
        return false;
    };
    callee.name() == Symbol::new(method)
        && callee
            .as_parent()
            .first()
            .map(|s| *s == "Bytes")
            .unwrap_or(false)
}

/// Pattern for `Bytes::len(bytes)` -> `struct.get $bytes 2` + `i64.extend_i32_u`
struct BytesLenPattern;

impl RewritePattern for BytesLenPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
    ) -> RewriteResult<'a> {
        if !is_bytes_method_call(db, op, "len") {
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
        let get_len = Operation::of_name(db, location, "wasm.struct_get")
            .operands(idvec![bytes_ref])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_STRUCT_IDX)))
            .attr("field_idx", Attribute::IntBits(u64::from(BYTES_LEN_FIELD)))
            .build();

        // Extend i32 to i64 (Int type in Tribute is i64)
        let extend = Operation::of_name(db, location, "wasm.i64_extend_i32_u")
            .operands(idvec![get_len.result(db, 0)])
            .results(idvec![i64_ty])
            .build();

        RewriteResult::Expand(vec![get_len, extend])
    }
}

/// Pattern for `Bytes::get_or_panic(bytes, index)` -> array access with offset
struct BytesGetOrPanicPattern;

impl RewritePattern for BytesGetOrPanicPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
    ) -> RewriteResult<'a> {
        if !is_bytes_method_call(db, op, "get_or_panic") {
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
        let get_data = Operation::of_name(db, location, "wasm.struct_get")
            .operands(idvec![bytes_ref])
            .results(idvec![array_ref_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_STRUCT_IDX)))
            .attr("field_idx", Attribute::IntBits(u64::from(BYTES_DATA_FIELD)))
            .build();

        // Get offset (field 1)
        let get_offset = Operation::of_name(db, location, "wasm.struct_get")
            .operands(idvec![bytes_ref])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_STRUCT_IDX)))
            .attr(
                "field_idx",
                Attribute::IntBits(u64::from(BYTES_OFFSET_FIELD)),
            )
            .build();

        // Wrap index to i32
        let index_i32 = Operation::of_name(db, location, "wasm.i32_wrap_i64")
            .operands(idvec![index])
            .results(idvec![i32_ty])
            .build();

        // Add offset to index: actual_index = offset + index
        let add_offset = Operation::of_name(db, location, "wasm.i32_add")
            .operands(idvec![get_offset.result(db, 0), index_i32.result(db, 0)])
            .results(idvec![i32_ty])
            .build();

        // array.get (signed extend to i32)
        let array_get = Operation::of_name(db, location, "wasm.array_get_s")
            .operands(idvec![get_data.result(db, 0), add_offset.result(db, 0)])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_ARRAY_IDX)))
            .build();

        // Extend i32 to i64 (Int type in Tribute is i64)
        let extend = Operation::of_name(db, location, "wasm.i64_extend_i32_s")
            .operands(idvec![array_get.result(db, 0)])
            .results(idvec![i64_ty])
            .build();

        RewriteResult::Expand(vec![
            get_data, get_offset, index_i32, add_offset, array_get, extend,
        ])
    }
}

/// Pattern for `Bytes::slice_or_panic(bytes, start, end)` -> new struct with adjusted offset/len
struct BytesSliceOrPanicPattern;

impl RewritePattern for BytesSliceOrPanicPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
    ) -> RewriteResult<'a> {
        if !is_bytes_method_call(db, op, "slice_or_panic") {
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
        let get_data = Operation::of_name(db, location, "wasm.struct_get")
            .operands(idvec![bytes_ref])
            .results(idvec![array_ref_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_STRUCT_IDX)))
            .attr("field_idx", Attribute::IntBits(u64::from(BYTES_DATA_FIELD)))
            .build();

        // Get current offset (field 1)
        let get_offset = Operation::of_name(db, location, "wasm.struct_get")
            .operands(idvec![bytes_ref])
            .results(idvec![i32_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_STRUCT_IDX)))
            .attr(
                "field_idx",
                Attribute::IntBits(u64::from(BYTES_OFFSET_FIELD)),
            )
            .build();

        // Wrap start and end to i32
        let start_i32 = Operation::of_name(db, location, "wasm.i32_wrap_i64")
            .operands(idvec![start])
            .results(idvec![i32_ty])
            .build();

        let end_i32 = Operation::of_name(db, location, "wasm.i32_wrap_i64")
            .operands(idvec![end])
            .results(idvec![i32_ty])
            .build();

        // new_offset = offset + start
        let new_offset = Operation::of_name(db, location, "wasm.i32_add")
            .operands(idvec![get_offset.result(db, 0), start_i32.result(db, 0)])
            .results(idvec![i32_ty])
            .build();

        // new_len = end - start
        let new_len = Operation::of_name(db, location, "wasm.i32_sub")
            .operands(idvec![end_i32.result(db, 0), start_i32.result(db, 0)])
            .results(idvec![i32_ty])
            .build();

        // struct.new to create new Bytes (shares the underlying array)
        let struct_new = Operation::of_name(db, location, "wasm.struct_new")
            .operands(idvec![
                get_data.result(db, 0),
                new_offset.result(db, 0),
                new_len.result(db, 0)
            ])
            .results(idvec![bytes_ty])
            .attr("type_idx", Attribute::IntBits(u64::from(BYTES_STRUCT_IDX)))
            .build();

        RewriteResult::Expand(vec![
            get_data, get_offset, start_i32, end_i32, new_offset, new_len, struct_new,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_print_line_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        // Create string constant with literal_len attribute
        let string_const = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(0)) // offset
            .attr("literal_len", Attribute::IntBits(5)) // "hello"
            .results(idvec![i32_ty])
            .build();

        // Create print_line call
        let print_line = Operation::of_name(db, location, "wasm.call")
            .operands(idvec![string_const.result(db, 0)])
            .results(idvec![nil_ty])
            .attr(
                "callee",
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("print_line"))),
            )
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![string_const, print_line],
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
                let Attribute::QualifiedName(callee) = call.callee(db) else {
                    return None;
                };
                Some(callee.name().to_string())
            })
            .collect()
    }

    #[salsa_test]
    fn test_print_line_to_fd_write(db: &salsa::DatabaseImpl) {
        let module = make_print_line_module(db);
        let op_names = lower_and_check(db, module);
        let callees = extract_callees(db, module);

        // Should have wasm.call operations
        assert!(op_names.iter().any(|n| n == "wasm.call"));
        // The call should be to fd_write, not print_line
        assert!(callees.contains(&"fd_write".to_string()));
        assert!(!callees.contains(&"print_line".to_string()));
    }

    // === Bytes intrinsic tests ===

    /// Create a qualified name for Bytes::method
    fn bytes_method_name(method: &'static str) -> QualifiedName {
        QualifiedName::new(vec![Symbol::new("Bytes")], Symbol::new(method))
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
        let len_call = Operation::of_name(db, location, "wasm.call")
            .operands(idvec![bytes_val])
            .results(idvec![i64_ty])
            .attr("callee", Attribute::QualifiedName(bytes_method_name("len")))
            .build();

        let block = Block::new(db, block_id, location, idvec![bytes_ty], idvec![len_call]);
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
        assert!(!callees.iter().any(|n| n == "len"));
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
        let get_call = Operation::of_name(db, location, "wasm.call")
            .operands(idvec![bytes_val, index_val])
            .results(idvec![i64_ty])
            .attr(
                "callee",
                Attribute::QualifiedName(bytes_method_name("get_or_panic")),
            )
            .build();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![bytes_ty, i64_ty],
            idvec![get_call],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_bytes_get_to_array_get(db: &salsa::DatabaseImpl) {
        let module = make_bytes_get_module(db);
        let op_names = lower_and_check(db, module);

        // Should have struct_get (for data and offset), i32_add, and array_get_s
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(op_names.iter().any(|n| n == "wasm.i32_add"));
        assert!(op_names.iter().any(|n| n == "wasm.array_get_s"));
        // No Bytes::get_or_panic call should remain
        let callees = extract_callees(db, module);
        assert!(!callees.iter().any(|n| n == "get_or_panic"));
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
        let slice_call = Operation::of_name(db, location, "wasm.call")
            .operands(idvec![bytes_val, start_val, end_val])
            .results(idvec![bytes_ty])
            .attr(
                "callee",
                Attribute::QualifiedName(bytes_method_name("slice_or_panic")),
            )
            .build();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![bytes_ty, i64_ty, i64_ty],
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
        assert!(!callees.iter().any(|n| n == "slice_or_panic"));
    }
}
