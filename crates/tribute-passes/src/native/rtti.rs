//! RTTI (Runtime Type Information) pass for the native backend.
//!
//! This pass scans the module for struct types used in `adt.struct_new` operations,
//! assigns each a unique `rtti_idx`, and generates per-type release functions that
//! recursively release pointer fields before deallocating the struct itself.
//!
//! ## RTTI Index Layout
//!
//! | Index | Type | Release |
//! |-------|------|---------|
//! | 0 | Nil | shallow |
//! | 1 | Bool | shallow |
//! | 2 | Nat | shallow |
//! | 3 | Int | shallow |
//! | 4 | Float | shallow |
//! | 5 | Rune | shallow |
//! | 6 | Bytes | shallow |
//! | 7 | Array | generic (future) |
//! | 8-31 | reserved | — |
//! | 32+ | user structs | per-type deep release |
//!
//! ## Pipeline Position
//!
//! Runs before `adt_to_clif` (Phase 1.9) so that `adt_to_clif` can use the
//! `RttiMap` to store correct `rtti_idx` values in allocation headers.

use std::collections::HashMap;

use tribute_ir::dialect::tribute_rt::{self, RC_HEADER_SIZE};
use trunk_ir::adt_layout::compute_struct_layout;
use trunk_ir::dialect::{adt, clif, core};
use trunk_ir::rewrite::TypeConverter;
use trunk_ir::{
    Block, BlockId, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type, Value,
};

/// First index for user-defined struct types.
pub const RTTI_USER_START: u32 = 32;

/// Reserved RTTI indices for built-in types.
pub const RTTI_NIL: u32 = 0;
pub const RTTI_BOOL: u32 = 1;
pub const RTTI_NAT: u32 = 2;
pub const RTTI_INT: u32 = 3;
pub const RTTI_FLOAT: u32 = 4;

/// Name prefix for per-type release functions.
pub const RELEASE_FN_PREFIX: &str = "__tribute_release_";

/// Name of the runtime deallocation function.
const DEALLOC_FN: &str = "__tribute_dealloc";

/// Mapping from struct types to their RTTI indices.
#[derive(Debug, Clone, Default)]
pub struct RttiMap<'db> {
    /// Struct type → rtti_idx.
    pub type_to_idx: HashMap<Type<'db>, u32>,
    /// Next available index for user structs.
    next_idx: u32,
}

impl<'db> RttiMap<'db> {
    pub fn new() -> Self {
        Self {
            type_to_idx: HashMap::new(),
            next_idx: RTTI_USER_START,
        }
    }

    /// Get or assign an rtti_idx for a struct type.
    pub fn get_or_insert(&mut self, ty: Type<'db>) -> u32 {
        if let Some(&idx) = self.type_to_idx.get(&ty) {
            return idx;
        }
        let idx = self.next_idx;
        self.next_idx += 1;
        self.type_to_idx.insert(ty, idx);
        idx
    }

    /// Look up the rtti_idx for a type, returning `None` if not registered.
    pub fn get(&self, ty: &Type<'db>) -> Option<u32> {
        self.type_to_idx.get(ty).copied()
    }
}

/// Run the RTTI pass: scan for struct types, assign indices, generate release functions.
///
/// Returns the updated module (with release functions appended) and the `RttiMap`.
pub fn generate_rtti<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    type_converter: &TypeConverter,
) -> (core::Module<'db>, RttiMap<'db>) {
    let mut rtti_map = RttiMap::new();

    // Phase 1: Scan module for all adt.struct_new operations to collect struct types.
    collect_struct_types(db, &module, &mut rtti_map);

    if rtti_map.type_to_idx.is_empty() {
        return (module, rtti_map);
    }

    // Phase 2: Generate per-type release functions.
    let release_fns = generate_release_functions(db, &rtti_map, type_converter);

    if release_fns.is_empty() {
        return (module, rtti_map);
    }

    // Phase 3: Append release functions to the module body.
    let module = append_functions_to_module(db, module, release_fns);

    (module, rtti_map)
}

/// Scan module body for `adt.struct_new` operations and register their types.
fn collect_struct_types<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
    rtti_map: &mut RttiMap<'db>,
) {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            collect_struct_types_in_op(db, op, rtti_map);
        }
    }
}

/// Recursively scan an operation (and its nested regions) for `adt.struct_new`.
fn collect_struct_types_in_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    rtti_map: &mut RttiMap<'db>,
) {
    // Check if this op is adt.struct_new
    if let Ok(struct_new) = adt::StructNew::from_operation(db, *op) {
        let struct_ty = struct_new.r#type(db);
        rtti_map.get_or_insert(struct_ty);
    }

    // Recurse into regions
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                collect_struct_types_in_op(db, nested_op, rtti_map);
            }
        }
    }
}

/// Generate per-type release functions for all struct types in the RTTI map.
///
/// Each release function:
/// 1. Loads each pointer field from the struct payload
/// 2. Calls `tribute_rt.release` on each pointer field (to be lowered later by rc_lowering)
/// 3. Deallocates the struct itself (payload_ptr - 8 = raw_ptr)
fn generate_release_functions<'db>(
    db: &'db dyn salsa::Database,
    rtti_map: &RttiMap<'db>,
    type_converter: &TypeConverter,
) -> Vec<Operation<'db>> {
    let mut funcs = Vec::new();

    // Sort by rtti_idx for deterministic output
    let mut entries: Vec<_> = rtti_map.type_to_idx.iter().collect();
    entries.sort_by_key(|(_, idx)| *idx);

    for (struct_ty, rtti_idx) in entries {
        let func_op =
            generate_release_function_for_struct(db, *struct_ty, *rtti_idx, type_converter);
        funcs.push(func_op);
    }

    funcs
}

/// Generate a single release function for a struct type.
///
/// ```text
/// clif.func @__tribute_release_<idx> {type = core.func(core.nil, core.ptr)} {
/// ^entry(%payload_ptr: core.ptr):
///   // For each pointer field:
///   %field = clif.load %payload_ptr {offset = <field_offset>} : core.ptr
///   tribute_rt.release %field {alloc_size = 0}
///
///   // Dealloc self:
///   %hdr_sz = clif.iconst {value = 8} : core.i64
///   %raw_ptr = clif.isub %payload_ptr, %hdr_sz : core.ptr
///   %size = clif.iconst {value = <total_alloc_size>} : core.i64
///   clif.call %raw_ptr, %size {callee = @__tribute_dealloc} : core.nil
///   clif.return
/// }
/// ```
fn generate_release_function_for_struct<'db>(
    db: &'db dyn salsa::Database,
    struct_ty: Type<'db>,
    rtti_idx: u32,
    type_converter: &TypeConverter,
) -> Operation<'db> {
    let fields = adt::get_struct_fields(db, struct_ty).unwrap_or_else(|| {
        unreachable!(
            "struct type registered in RttiMap must have fields: {}.{}",
            struct_ty.dialect(db),
            struct_ty.name(db),
        )
    });
    let layout = compute_struct_layout(db, struct_ty, type_converter).unwrap_or_else(|| {
        unreachable!(
            "struct type registered in RttiMap must have a valid layout: {}.{}",
            struct_ty.dialect(db),
            struct_ty.name(db),
        )
    });

    let ptr_ty = core::Ptr::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();

    // Create a dummy location for generated code
    let path = trunk_ir::PathId::new(db, "<rtti>".to_owned());
    let location = Location::new(path, trunk_ir::Span::new(0, 0));

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);

    // Function type: (core.ptr) -> core.nil
    let func_ty = core::Func::new(db, IdVec::from(vec![ptr_ty]), nil_ty).as_type();

    // Build body
    let entry_block_id = BlockId::fresh();
    let payload_arg = trunk_ir::BlockArg::of_type(db, ptr_ty);
    let payload_ptr = Value::new(db, trunk_ir::ValueDef::BlockArg(entry_block_id), 0);

    let mut ops: Vec<Operation<'db>> = Vec::new();

    // For each pointer field, load and release
    for (i, (_field_name, field_ty)) in fields.iter().enumerate() {
        let native_ty = type_converter
            .convert_type(db, *field_ty)
            .unwrap_or(*field_ty);

        // Only release pointer fields
        if core::Ptr::from_type(db, native_ty).is_none() {
            continue;
        }

        let offset = layout.field_offsets[i] as i32;

        // Load field value
        let load_op = clif::load(db, location, payload_ptr, ptr_ty, offset);
        let field_val = load_op.result(db);
        ops.push(load_op.as_operation());

        // Release field (will be lowered by rc_lowering later)
        let release_op = tribute_rt::release(db, location, field_val, 0);
        ops.push(release_op.as_operation());
    }

    // Dealloc self: raw_ptr = payload_ptr - RC_HEADER_SIZE
    let hdr_sz = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_sz.as_operation());
    let raw_ptr = clif::isub(db, location, payload_ptr, hdr_sz.result(db), ptr_ty);
    ops.push(raw_ptr.as_operation());

    // Total allocation size = payload + header
    let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
    let size_op = clif::iconst(db, location, i64_ty, alloc_size as i64);
    ops.push(size_op.as_operation());

    // Call __tribute_dealloc(raw_ptr, size)
    let dealloc_call = clif::call(
        db,
        location,
        [raw_ptr.result(db), size_op.result(db)],
        nil_ty,
        Symbol::new(DEALLOC_FN),
    );
    ops.push(dealloc_call.as_operation());

    // Return
    let ret_op = clif::r#return(db, location, []);
    ops.push(ret_op.as_operation());

    // Build the block and region
    let entry_block = Block::new(
        db,
        entry_block_id,
        location,
        IdVec::from(vec![payload_arg]),
        ops.into_iter().collect(),
    );
    let body = Region::new(db, location, IdVec::from(vec![entry_block]));

    // Build the function operation
    let func_op = clif::func(
        db,
        location,
        Symbol::from_dynamic(&func_name),
        func_ty,
        body,
    );

    func_op.as_operation()
}

/// Append function operations to a module's body.
fn append_functions_to_module<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    new_ops: Vec<Operation<'db>>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    if blocks.is_empty() || new_ops.is_empty() {
        return module;
    }

    // Append new ops to the first (and typically only) block
    let first_block = &blocks[0];
    let mut combined_ops: Vec<Operation<'db>> =
        first_block.operations(db).iter().copied().collect();
    combined_ops.extend(new_ops);

    let new_block = Block::new(
        db,
        first_block.id(db),
        first_block.location(db),
        first_block.args(db).clone(),
        combined_ops.into_iter().collect(),
    );

    // Keep other blocks unchanged
    let mut new_blocks: Vec<Block<'db>> = vec![new_block];
    new_blocks.extend(blocks.iter().skip(1).copied());

    let new_body = Region::new(db, body.location(db), new_blocks.into_iter().collect());
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_op;

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    #[salsa::input]
    struct TextInput {
        #[returns(ref)]
        text: String,
    }

    #[salsa::tracked]
    fn do_rtti_pass(db: &dyn salsa::Database, input: TextInput) -> String {
        let module = parse_test_module(db, input.text(db));
        let (new_module, _rtti_map) = generate_rtti(db, module, &test_converter());
        print_op(db, new_module.as_operation())
    }

    fn run_rtti_pass(db: &salsa::DatabaseImpl, ir: &str) -> String {
        let input = TextInput::new(db, ir.to_string());
        do_rtti_pass(db, input)
    }

    #[salsa_test]
    fn test_rtti_map_assignment(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let point_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );
        let node_ty = adt::struct_type(
            db,
            Symbol::new("Node"),
            vec![
                (Symbol::new("value"), i32_ty),
                (Symbol::new("next"), ptr_ty),
            ],
        );

        let mut rtti_map = RttiMap::new();
        let idx1 = rtti_map.get_or_insert(point_ty);
        let idx2 = rtti_map.get_or_insert(node_ty);

        assert_eq!(idx1, RTTI_USER_START);
        assert_eq!(idx2, RTTI_USER_START + 1);

        // Same type gets same index
        assert_eq!(rtti_map.get_or_insert(point_ty), idx1);
    }

    // === generate_rtti: no structs ===

    #[salsa_test]
    fn test_no_structs_noop(db: &salsa::DatabaseImpl) {
        let ir = run_rtti_pass(
            db,
            r#"
            core.module @test {
                %val = clif.iconst {value = 42} : core.i32
            }
            "#,
        );
        assert!(
            !ir.contains("__tribute_release_"),
            "should not contain release functions: {ir}"
        );
    }

    // === generate_rtti: struct with no pointer fields ===

    #[salsa_test]
    fn test_struct_no_ptr_fields(db: &salsa::DatabaseImpl) {
        let ir = run_rtti_pass(
            db,
            r#"
            core.module @test {
                %x = clif.iconst {value = 10} : core.i32
                %y = clif.iconst {value = 20} : core.i32
                %s = adt.struct_new %x, %y {type = adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @Point}} : adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @Point}
            }
            "#,
        );
        assert!(
            ir.contains("__tribute_release_32"),
            "should generate release function: {ir}"
        );
        assert!(
            !ir.contains("tribute_rt.release"),
            "should not have release for non-ptr fields: {ir}"
        );
        assert!(
            ir.contains("__tribute_dealloc"),
            "should have dealloc: {ir}"
        );
        insta::assert_snapshot!(ir);
    }

    // === generate_rtti: struct with pointer fields ===

    #[salsa_test]
    fn test_struct_with_ptr_fields(db: &salsa::DatabaseImpl) {
        let ir = run_rtti_pass(
            db,
            r#"
            core.module @test {
                %v = clif.iconst {value = 42} : core.i32
                %p = clif.iconst {value = 0} : core.ptr
                %s = adt.struct_new %v, %p {type = adt.struct() {fields = [[@value, core.i32], [@next, core.ptr]], name = @Node}} : adt.struct() {fields = [[@value, core.i32], [@next, core.ptr]], name = @Node}
            }
            "#,
        );
        assert!(
            ir.contains("__tribute_release_32"),
            "should generate release function: {ir}"
        );
        assert!(
            ir.contains("tribute_rt.release"),
            "should release ptr field: {ir}"
        );
        assert!(
            ir.contains("__tribute_dealloc"),
            "should have dealloc: {ir}"
        );
        insta::assert_snapshot!(ir);
    }

    // === generate_rtti: multiple struct types ===

    #[salsa_test]
    fn test_multiple_struct_types(db: &salsa::DatabaseImpl) {
        let ir = run_rtti_pass(
            db,
            r#"
            core.module @test {
                %x = clif.iconst {value = 1} : core.i32
                %y = clif.iconst {value = 2} : core.i32
                %s1 = adt.struct_new %x, %y {type = adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @Point}} : adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @Point}
                %p = clif.iconst {value = 0} : core.ptr
                %s2 = adt.struct_new %x, %p {type = adt.struct() {fields = [[@value, core.i32], [@next, core.ptr]], name = @Node}} : adt.struct() {fields = [[@value, core.i32], [@next, core.ptr]], name = @Node}
            }
            "#,
        );
        assert!(
            ir.contains("__tribute_release_32"),
            "should generate release for first struct: {ir}"
        );
        assert!(
            ir.contains("__tribute_release_33"),
            "should generate release for second struct: {ir}"
        );
    }

    #[salsa_test]
    fn test_rc_insertion_skip_prefix(_db: &salsa::DatabaseImpl) {
        let name = format!("{}{}", RELEASE_FN_PREFIX, 32);
        assert!(name.starts_with("__tribute_release_"));
    }
}
