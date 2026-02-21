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
use trunk_ir::adt_layout::{StructLayout, compute_enum_layout, compute_struct_layout};
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

    // Phase 1: Scan module for all adt.struct_new and adt.variant_new operations.
    collect_struct_types(db, &module, &mut rtti_map);
    collect_enum_types(db, &module, &mut rtti_map);

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
    // Check if this op is adt.struct_new with a proper struct type
    if let Ok(struct_new) = adt::StructNew::from_operation(db, *op) {
        let struct_ty = struct_new.r#type(db);
        // Only register types that are actual adt.struct types (with field info).
        // Both named structs and tuples use adt.struct types since #408.
        if adt::is_struct_type(db, struct_ty) {
            rtti_map.get_or_insert(struct_ty);
        }
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

/// Scan module body for `adt.variant_new` operations and register their base enum types.
fn collect_enum_types<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
    rtti_map: &mut RttiMap<'db>,
) {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            collect_enum_types_in_op(db, op, rtti_map);
        }
    }
}

/// Recursively scan an operation (and its nested regions) for `adt.variant_new`.
fn collect_enum_types_in_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    rtti_map: &mut RttiMap<'db>,
) {
    if let Ok(variant_new) = adt::VariantNew::from_operation(db, *op) {
        let enum_ty = variant_new.r#type(db);
        if adt::is_enum_type(db, enum_ty) {
            rtti_map.get_or_insert(enum_ty);
        }
    }

    // Recurse into regions
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                collect_enum_types_in_op(db, nested_op, rtti_map);
            }
        }
    }
}

/// Generate per-type release functions for all registered types in the RTTI map.
///
/// Each release function:
/// 1. Loads and releases pointer fields from the payload
/// 2. Deallocates the object itself (payload_ptr - 8 = raw_ptr)
fn generate_release_functions<'db>(
    db: &'db dyn salsa::Database,
    rtti_map: &RttiMap<'db>,
    type_converter: &TypeConverter,
) -> Vec<Operation<'db>> {
    let mut funcs = Vec::new();

    // Sort by rtti_idx for deterministic output
    let mut entries: Vec<_> = rtti_map.type_to_idx.iter().collect();
    entries.sort_by_key(|(_, idx)| *idx);

    for (ty, rtti_idx) in entries {
        let func_op = if adt::is_enum_type(db, *ty) {
            generate_release_function_for_enum(db, *ty, *rtti_idx, type_converter)
        } else {
            generate_release_function_for_struct(db, *ty, *rtti_idx, type_converter)
        };
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
    let i8_ty = core::I8::new(db).as_type();

    // Create a dummy location for generated code
    let path = trunk_ir::PathId::new(db, "<rtti>".to_owned());
    let location = Location::new(path, trunk_ir::Span::new(0, 0));

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);

    // Function type: (core.ptr) -> core.nil
    let func_ty = core::Func::new(db, IdVec::from(vec![ptr_ty]), nil_ty).as_type();

    // Collect pointer field offsets, skipping function pointer fields.
    //
    // Fields named "func_ptr" hold static code pointers (in __TEXT segment)
    // and must not be RC-released. This convention is established by the
    // native closure struct in func_to_clif.
    let func_ptr_sym = Symbol::new("func_ptr");
    let ptr_field_offsets: Vec<i32> = fields
        .iter()
        .enumerate()
        .filter_map(|(i, (name, field_ty))| {
            if *name == func_ptr_sym {
                return None;
            }
            let native_ty = type_converter
                .convert_type(db, *field_ty)
                .unwrap_or(*field_ty);
            if core::Ptr::from_type(db, native_ty).is_some() {
                Some(layout.field_offsets[i] as i32)
            } else {
                None
            }
        })
        .collect();

    let entry_block_id = BlockId::fresh();
    let payload_arg = trunk_ir::BlockArg::of_type(db, ptr_ty);
    let payload_ptr = Value::new(db, trunk_ir::ValueDef::BlockArg(entry_block_id), 0);

    // Build body with null-guarded field releases.
    //
    // For each pointer field, we emit a null check before releasing. This
    // prevents SIGBUS when a field contains a null pointer (e.g., nil-typed
    // closure environments) or a static/code pointer.
    //
    // Structure (built backwards, then reversed):
    //   ^entry:      load f0, null-check → ^after_f0 / ^release_f0
    //   ^release_f0: release f0, jump ^after_f0
    //   ^after_f0:   load f1, null-check → ^after_f1 / ^release_f1
    //   ^release_f1: release f1, jump ^after_f1
    //   ...
    //   ^dealloc:    dealloc self, return

    // Step 1: Build the dealloc block (always the last block)
    let dealloc_block_id = BlockId::fresh();
    let mut dealloc_ops: Vec<Operation<'db>> = Vec::new();
    gen_dealloc_and_return(
        db,
        location,
        payload_ptr,
        &layout,
        ptr_ty,
        i64_ty,
        &mut dealloc_ops,
    );
    let dealloc_block = Block::new(
        db,
        dealloc_block_id,
        location,
        IdVec::new(),
        dealloc_ops.into_iter().collect(),
    );

    // Step 2: Build field check/release blocks backwards from the last field
    // to the first. Each iteration produces a "check" block and a "release" block.
    // `next_block` tracks the continuation destination.
    let mut blocks_reversed: Vec<Block<'db>> = vec![dealloc_block];
    let mut next_block_for_field: Block<'db> = *blocks_reversed.last().unwrap();

    for &offset in ptr_field_offsets.iter().rev() {
        let release_block_id = BlockId::fresh();
        let check_block_id = BlockId::fresh();

        // Release block: reload field, release, jump to next
        let mut release_ops: Vec<Operation<'db>> = Vec::new();
        let reload = clif::load(db, location, payload_ptr, ptr_ty, offset);
        release_ops.push(reload.as_operation());
        let release = tribute_rt::release(db, location, reload.result(db), 0);
        release_ops.push(release.as_operation());
        // Jump to the next field check (or dealloc) after releasing.
        // rc_lowering will split this block further for the refcount check,
        // placing this jump in the resulting "continue" block.
        let jump = clif::jump(db, location, [], next_block_for_field);
        release_ops.push(jump.as_operation());

        let release_block = Block::new(
            db,
            release_block_id,
            location,
            IdVec::new(),
            release_ops.into_iter().collect(),
        );

        // Check block: load field, compare to null, branch
        let mut check_ops: Vec<Operation<'db>> = Vec::new();
        let load = clif::load(db, location, payload_ptr, ptr_ty, offset);
        check_ops.push(load.as_operation());
        let null_const = clif::iconst(db, location, ptr_ty, 0);
        check_ops.push(null_const.as_operation());
        let is_null = clif::icmp(
            db,
            location,
            load.result(db),
            null_const.result(db),
            i8_ty,
            Symbol::new("eq"),
        );
        check_ops.push(is_null.as_operation());
        let brif = clif::brif(
            db,
            location,
            is_null.result(db),
            next_block_for_field, // null → skip to next field / dealloc
            release_block,        // non-null → release
        );
        check_ops.push(brif.as_operation());

        let check_block = Block::new(
            db,
            check_block_id,
            location,
            IdVec::new(),
            check_ops.into_iter().collect(),
        );

        blocks_reversed.push(release_block);
        blocks_reversed.push(check_block);
        next_block_for_field = check_block;
    }

    // Step 3: Build the entry block
    let body = if ptr_field_offsets.is_empty() {
        // No ptr fields: entry block is just dealloc
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            IdVec::from(vec![payload_arg]),
            blocks_reversed.pop().unwrap().operations(db).clone(),
        );
        Region::new(db, location, IdVec::from(vec![entry_block]))
    } else {
        // Entry block jumps to the first check block
        let first_check = blocks_reversed.pop().unwrap();
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            IdVec::from(vec![payload_arg]),
            first_check.operations(db).clone(),
        );

        blocks_reversed.reverse();
        let mut all_blocks = vec![entry_block];
        all_blocks.extend(blocks_reversed);
        Region::new(db, location, all_blocks.into_iter().collect())
    };

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

/// Emit dealloc + return ops into the given `ops` vector.
///
/// Generates:
/// ```text
/// %hdr_sz  = clif.iconst(8) : core.i64
/// %raw_ptr = clif.isub(%payload, %hdr_sz) : core.ptr
/// %size    = clif.iconst(<alloc_size>) : core.i64
/// clif.call @__tribute_dealloc(%raw_ptr, %size) : core.nil
/// clif.return
/// ```
fn gen_dealloc_and_return<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    payload_ptr: Value<'db>,
    layout: &StructLayout,
    ptr_ty: Type<'db>,
    i64_ty: Type<'db>,
    ops: &mut Vec<Operation<'db>>,
) {
    let nil_ty = core::Nil::new(db).as_type();
    let hdr_sz = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_sz.as_operation());
    let raw_ptr = clif::isub(db, location, payload_ptr, hdr_sz.result(db), ptr_ty);
    ops.push(raw_ptr.as_operation());

    let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
    let size_op = clif::iconst(db, location, i64_ty, alloc_size as i64);
    ops.push(size_op.as_operation());

    let dealloc_call = clif::call(
        db,
        location,
        [raw_ptr.result(db), size_op.result(db)],
        nil_ty,
        Symbol::new(DEALLOC_FN),
    );
    ops.push(dealloc_call.as_operation());

    let ret_op = clif::r#return(db, location, []);
    ops.push(ret_op.as_operation());
}

/// Generate a release function for an enum type.
///
/// The function loads the tag, then for each variant with pointer fields,
/// branches to a release block. Finally deallocates the object.
///
/// Uses `clif.brif` and basic blocks directly (not `scf.if`) because
/// this function is generated after `lower_scf_to_cf` has already run.
///
/// All non-entry blocks have NO block parameters. They reference the entry
/// block's `payload_ptr` value directly via SSA dominance, since `clif.brif`
/// does not pass block arguments.
///
/// ```text
/// clif.func @__tribute_release_<idx> {type = core.func(core.nil, core.ptr)} {
/// ^entry(%payload_ptr: core.ptr):
///   %tag = clif.load %payload_ptr {offset = 0} : core.i32
///   %expected_0 = clif.iconst <tag_0>
///   %is_v0 = clif.icmp eq %tag, %expected_0
///   clif.brif %is_v0, ^release_v0, ^check_v1
/// ^check_v1:
///   %expected_1 = clif.iconst <tag_1>
///   %is_v1 = clif.icmp eq %tag, %expected_1
///   clif.brif %is_v1, ^release_v1, ^dealloc
/// ^release_v0:
///   // load & release pointer fields for variant 0
///   clif.jump ^dealloc
/// ^release_v1:
///   // load & release pointer fields for variant 1
///   clif.jump ^dealloc
/// ^dealloc:
///   clif.call @__tribute_dealloc(raw_ptr, size)
///   clif.return
/// }
/// ```
fn generate_release_function_for_enum<'db>(
    db: &'db dyn salsa::Database,
    enum_ty: Type<'db>,
    rtti_idx: u32,
    type_converter: &TypeConverter,
) -> Operation<'db> {
    let layout = compute_enum_layout(db, enum_ty, type_converter).unwrap_or_else(|| {
        unreachable!(
            "enum type registered in RttiMap must have a valid layout: {}.{}",
            enum_ty.dialect(db),
            enum_ty.name(db),
        )
    });

    let variants = adt::get_enum_variants(db, enum_ty).unwrap_or_default();

    let ptr_ty = core::Ptr::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();
    let i1_ty = core::I1::new(db).as_type();

    let path = trunk_ir::PathId::new(db, "<rtti>".to_owned());
    let location = Location::new(path, trunk_ir::Span::new(0, 0));

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);
    let func_ty = core::Func::new(db, IdVec::from(vec![ptr_ty]), nil_ty).as_type();

    let entry_block_id = BlockId::fresh();
    let payload_arg = trunk_ir::BlockArg::of_type(db, ptr_ty);
    let payload_ptr = Value::new(db, trunk_ir::ValueDef::BlockArg(entry_block_id), 0);

    // Collect variants that have pointer fields and need release blocks
    struct VariantRelease {
        tag_value: u32,
        ptr_field_offsets: Vec<i32>,
    }
    let mut variants_with_ptrs: Vec<VariantRelease> = Vec::new();

    for (variant_idx, (_variant_name, field_types)) in variants.iter().enumerate() {
        let variant_layout = &layout.variant_layouts[variant_idx];

        let ptr_field_offsets: Vec<i32> = field_types
            .iter()
            .enumerate()
            .filter_map(|(field_idx, field_ty)| {
                let native_ty = type_converter
                    .convert_type(db, *field_ty)
                    .unwrap_or(*field_ty);
                if core::Ptr::from_type(db, native_ty).is_some() {
                    Some((layout.fields_offset + variant_layout.field_offsets[field_idx]) as i32)
                } else {
                    None
                }
            })
            .collect();

        if !ptr_field_offsets.is_empty() {
            variants_with_ptrs.push(VariantRelease {
                tag_value: variant_layout.tag_value,
                ptr_field_offsets,
            });
        }
    }

    // Build dealloc block (final block, all paths converge here).
    // No block params — references entry's payload_ptr via SSA dominance.
    let dealloc_block_id = BlockId::fresh();
    let dealloc_block = {
        let mut ops: Vec<Operation<'db>> = Vec::new();

        let hdr_sz = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
        ops.push(hdr_sz.as_operation());
        let raw_ptr = clif::isub(db, location, payload_ptr, hdr_sz.result(db), ptr_ty);
        ops.push(raw_ptr.as_operation());

        let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = clif::iconst(db, location, i64_ty, alloc_size as i64);
        ops.push(size_op.as_operation());

        let dealloc_call = clif::call(
            db,
            location,
            [raw_ptr.result(db), size_op.result(db)],
            nil_ty,
            Symbol::new(DEALLOC_FN),
        );
        ops.push(dealloc_call.as_operation());

        let ret_op = clif::r#return(db, location, []);
        ops.push(ret_op.as_operation());

        Block::new(
            db,
            dealloc_block_id,
            location,
            IdVec::new(),
            ops.into_iter().collect(),
        )
    };

    if variants_with_ptrs.is_empty() {
        // No pointer fields in any variant: entry jumps straight to dealloc
        let jump = clif::jump(db, location, [], dealloc_block);

        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            IdVec::from(vec![payload_arg]),
            IdVec::from(vec![jump.as_operation()]),
        );

        let body = Region::new(db, location, IdVec::from(vec![entry_block, dealloc_block]));
        let func_op = clif::func(
            db,
            location,
            Symbol::from_dynamic(&func_name),
            func_ty,
            body,
        );
        return func_op.as_operation();
    }

    // Build release blocks for each variant.
    // No block params — references entry's payload_ptr via SSA dominance.
    let mut release_blocks: Vec<Block<'db>> = Vec::new();

    for vr in &variants_with_ptrs {
        let release_block_id = BlockId::fresh();
        let mut release_ops: Vec<Operation<'db>> = Vec::new();

        for &offset in &vr.ptr_field_offsets {
            let load_op = clif::load(db, location, payload_ptr, ptr_ty, offset);
            let field_val = load_op.result(db);
            release_ops.push(load_op.as_operation());

            let release_op = tribute_rt::release(db, location, field_val, 0);
            release_ops.push(release_op.as_operation());
        }

        let jump_to_dealloc = clif::jump(db, location, [], dealloc_block);
        release_ops.push(jump_to_dealloc.as_operation());

        let release_block = Block::new(
            db,
            release_block_id,
            location,
            IdVec::new(),
            release_ops.into_iter().collect(),
        );
        release_blocks.push(release_block);
    }

    // Build the entry block: load tag, then chain check-and-branch for each variant.
    // For a single variant with ptrs, the entry block does:
    //   load tag → icmp → brif(release_block, dealloc)
    // For multiple, we chain: entry checks first, then check_blocks for the rest.
    //
    // Entry block handles the first check. Additional check blocks handle the rest.
    let mut entry_ops: Vec<Operation<'db>> = Vec::new();

    // Load tag from payload_ptr + 0
    let tag_load = clif::load(db, location, payload_ptr, i32_ty, 0);
    let tag_val = tag_load.result(db);
    entry_ops.push(tag_load.as_operation());

    // Build additional check blocks (for variants_with_ptrs[1..]) in reverse order
    // so we can reference the "next" block when building each one.
    let mut check_blocks: Vec<Block<'db>> = Vec::new();
    let num_variants = variants_with_ptrs.len();

    // Build check blocks for index 1..n in reverse
    // The last check's else goes to dealloc, previous checks' else goes to the next check.
    let mut next_else_block: Block<'db> = dealloc_block;

    for i in (1..num_variants).rev() {
        let vr = &variants_with_ptrs[i];
        let check_block_id = BlockId::fresh();
        let mut check_ops: Vec<Operation<'db>> = Vec::new();

        let expected = clif::iconst(db, location, i32_ty, vr.tag_value as i64);
        check_ops.push(expected.as_operation());

        let cmp_op = clif::icmp(
            db,
            location,
            tag_val,
            expected.result(db),
            i1_ty,
            Symbol::new("eq"),
        );
        check_ops.push(cmp_op.as_operation());

        let brif_op = clif::brif(
            db,
            location,
            cmp_op.result(db),
            release_blocks[i],
            next_else_block,
        );
        check_ops.push(brif_op.as_operation());

        let check_block = Block::new(
            db,
            check_block_id,
            location,
            IdVec::new(),
            check_ops.into_iter().collect(),
        );

        next_else_block = check_block;
        check_blocks.push(check_block);
    }

    // Reverse to get forward order (check[1], check[2], ...)
    check_blocks.reverse();

    // Entry block: check for first variant, else goes to first check block (or dealloc)
    let first_vr = &variants_with_ptrs[0];
    let expected = clif::iconst(db, location, i32_ty, first_vr.tag_value as i64);
    entry_ops.push(expected.as_operation());

    let cmp_op = clif::icmp(
        db,
        location,
        tag_val,
        expected.result(db),
        i1_ty,
        Symbol::new("eq"),
    );
    entry_ops.push(cmp_op.as_operation());

    let brif_op = clif::brif(
        db,
        location,
        cmp_op.result(db),
        release_blocks[0],
        next_else_block,
    );
    entry_ops.push(brif_op.as_operation());

    let entry_block = Block::new(
        db,
        entry_block_id,
        location,
        IdVec::from(vec![payload_arg]),
        entry_ops.into_iter().collect(),
    );

    // Assemble all blocks: entry, check blocks, release blocks, dealloc
    let mut all_blocks: Vec<Block<'db>> = Vec::new();
    all_blocks.push(entry_block);
    all_blocks.extend(check_blocks);
    all_blocks.extend(release_blocks);
    all_blocks.push(dealloc_block);

    let body = Region::new(db, location, all_blocks.into_iter().collect());
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

    // === generate_rtti: closure struct with func_ptr field skipped ===

    #[salsa_test]
    fn test_closure_struct_skips_func_ptr(db: &salsa::DatabaseImpl) {
        let ir = run_rtti_pass(
            db,
            r#"
            core.module @test {
                %fp = clif.iconst {value = 0} : core.ptr
                %env = clif.iconst {value = 0} : core.ptr
                %s = adt.struct_new %fp, %env {type = adt.struct() {fields = [[@func_ptr, core.ptr], [@env, core.ptr]], name = @_closure}} : adt.struct() {fields = [[@func_ptr, core.ptr], [@env, core.ptr]], name = @_closure}
            }
            "#,
        );
        assert!(
            ir.contains("__tribute_release_32"),
            "should generate release function: {ir}"
        );
        // The release function should only release the env field (offset 8),
        // not the func_ptr field (offset 0) which is a static code pointer.
        let release_section = ir.split("__tribute_release_32").nth(1).unwrap();
        let release_count = release_section.matches("tribute_rt.release").count();
        assert_eq!(
            release_count, 1,
            "should have exactly 1 release (env only), got {}: {ir}",
            release_count
        );
    }

    #[salsa_test]
    fn test_rc_insertion_skip_prefix(_db: &salsa::DatabaseImpl) {
        let name = format!("{}{}", RELEASE_FN_PREFIX, 32);
        assert!(name.starts_with("__tribute_release_"));
    }
}
