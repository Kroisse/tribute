//! GC type collection from wasm dialect operations (arena IR version).
//!
//! This module traverses wasm operations to collect WebAssembly GC type
//! definitions (structs and arrays) and build the type index mappings.

use rustc_hash::FxHashMap as HashMap;

use tracing::debug;

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::types::Attribute;
use wasm_encoder::{FieldType, StorageType, ValType};

use crate::gc_types::{self, EVIDENCE_IDX, FIRST_USER_TYPE_IDX, GcTypeDef, MARKER_IDX};
use crate::passes::wasm_gc_to_wasm::GC_TYPES_ATTR;
use crate::{CompilationError, CompilationResult};

use super::helpers;

/// Result type for GC type collection.
pub(crate) type GcTypesResult = (Vec<GcTypeDef>, HashMap<TypeRef, u32>);

/// GC type kind enum
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GcKind {
    Struct,
    Array,
    Unknown,
}

/// Builder for GC types (struct or array)
struct GcTypeBuilder {
    kind: GcKind,
    fields: Vec<Option<TypeRef>>,
    array_elem: Option<TypeRef>,
    field_count: Option<usize>,
    /// Whether `fields` come from a structural struct type rather than from
    /// the operations that access the type.
    declared: bool,
}

impl GcTypeBuilder {
    fn new() -> Self {
        Self {
            kind: GcKind::Unknown,
            fields: Vec::new(),
            array_elem: None,
            field_count: None,
            declared: false,
        }
    }

    /// A struct whose fields are the parameters of its structural type.
    fn declared(fields: &[TypeRef]) -> Self {
        Self {
            kind: GcKind::Struct,
            fields: fields.iter().copied().map(Some).collect(),
            array_elem: None,
            field_count: Some(fields.len()),
            declared: true,
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Returns true if this is a built-in type (0-5) that shouldn't use a builder
fn is_builtin_type(idx: u32) -> bool {
    idx < FIRST_USER_TYPE_IDX
}

/// Get or create a builder for a user-defined type.
/// Returns None for built-in types (0-5) which are predefined.
fn try_get_builder(builders: &mut Vec<GcTypeBuilder>, idx: u32) -> Option<&mut GcTypeBuilder> {
    // Skip built-in types (0-5) as they are predefined
    if is_builtin_type(idx) {
        return None;
    }
    // Subtract FIRST_USER_TYPE_IDX because indices 0-5 are reserved for built-in types
    // User type indices start at FIRST_USER_TYPE_IDX
    let adjusted_idx = (idx - FIRST_USER_TYPE_IDX) as usize;
    if builders.len() <= adjusted_idx {
        builders.resize_with(adjusted_idx + 1, GcTypeBuilder::new);
    }
    Some(&mut builders[adjusted_idx])
}

/// Register a type in the type_idx_by_type map
fn register_type(
    ctx: &IrContext,
    type_idx_by_type: &mut HashMap<TypeRef, u32>,
    idx: u32,
    ty: TypeRef,
) {
    // An operation index identifies its layout, not every value of an abstract
    // reference type elsewhere in the module. Builtin ABI mappings are seeded
    // separately and must not be inferred from individual operations.
    if wasm_dialect::is_abstract_heap_type(ctx, ty) {
        return;
    }
    type_idx_by_type.entry(ty).or_insert(idx);
}

/// Check retained Marker declarations against the predefined evidence layout.
/// The runtime layout identifier selects the layout; it does not validate fields.
fn validate_marker_layout(ctx: &IrContext, ty: TypeRef) -> CompilationResult<()> {
    let invalid = || CompilationError::type_error("Marker declaration differs from builtin layout");
    let marker = ctx.get_type(ty);
    let GcTypeDef::Struct(expected) = &gc_types::builtin_types()[MARKER_IDX as usize] else {
        unreachable!("Marker is a builtin struct")
    };
    if marker.params.len() != expected.len() {
        return Err(invalid());
    }
    for (((ty, attrs), expected), role) in marker.params_with_attrs(ctx).zip(expected).zip([
        "ability_id",
        "prompt_tag",
        "tr_dispatch_fn",
        "shadowed",
        "outer",
    ]) {
        let field_type = ctx.get_type(ty);
        if attrs.view(ctx).get_str("name") != Some(role)
            || !field_type.params.is_empty()
            || !field_type.attrs.is_empty()
            || !(trunk_ir::dialect::core::I32::matches(ctx, ty)
                || trunk_ir::dialect::core::Ptr::matches(ctx, ty)
                || trunk_ir::dialect::wasm::Anyref::matches(ctx, ty))
        {
            return Err(invalid());
        }
        // Marker pointer fields are target-owned GC references, as specified
        // by the builtin layout, even in the retained pre-target declaration:
        // the dispatch closure is the closure base and the others are anyref.
        let actual = if trunk_ir::dialect::core::Ptr::matches(ctx, ty) {
            let StorageType::Val(reference @ ValType::Ref(_)) = expected.element_type else {
                return Err(invalid());
            };
            reference
        } else {
            helpers::type_to_valtype(ctx, ty, &HashMap::default())?
        };
        if StorageType::Val(actual) != expected.element_type {
            return Err(invalid());
        }
    }
    Ok(())
}

/// Register a validated retained declaration for an indexed evidence operation.
fn register_builtin_evidence_type(
    ctx: &IrContext,
    map: &mut HashMap<TypeRef, u32>,
    index: u32,
    ty: TypeRef,
) -> CompilationResult<()> {
    // Builtins bypass user builders, but retained full declarations still need
    // local/signature mappings. Abstract refs must not acquire an indexed type.
    if matches!(index, MARKER_IDX | EVIDENCE_IDX)
        && crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty) == Some(index)
    {
        let marker = if index == MARKER_IDX {
            ty
        } else {
            ctx.get_type(ty).params[0]
        };
        validate_marker_layout(ctx, marker)?;
        register_type(ctx, map, index, ty);
    }
    Ok(())
}

/// Normalize a type for GC struct field comparison.
///
/// Normalizes `core.i1` storage to its canonical form.
fn normalize_type_for_gc(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    // Wasm has no i1 storage type. Match type_to_valtype before comparing
    // constructor, getter, and setter observations of the same field.
    if trunk_ir::dialect::core::I1::matches(ctx, ty) {
        return trunk_ir::types::TypeDataBuilder::new("core.i32").intern(ctx);
    }

    ty
}

/// [`normalize_type_for_gc`], with a structural struct as the struct
/// supertype.
fn normalize_type_for_comparison(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    let ty = normalize_type_for_gc(ctx, ty);
    if wasm_gc::Struct::matches(ctx, ty) {
        return intern_wasm_structref(ctx);
    }
    ty
}

/// Check if two types are semantically equivalent for GC struct fields.
fn types_equivalent_for_gc(ctx: &mut IrContext, ty1: TypeRef, ty2: TypeRef) -> bool {
    // First try direct comparison
    if ty1 == ty2 {
        return true;
    }
    // Normalize both types. A structural struct is compared as the struct
    // supertype.
    let ty1_norm = normalize_type_for_comparison(ctx, ty1);
    let ty2_norm = normalize_type_for_comparison(ctx, ty2);
    if ty1_norm == ty2_norm {
        return true;
    }
    // Types of one builtin layout share its physical struct. A logical
    // closure reference and the materialized closure struct declaration are
    // such a pair.
    let layout_1 = helpers::builtin_layout_type_idx(ctx, ty1_norm);
    if layout_1.is_some() && layout_1 == helpers::builtin_layout_type_idx(ctx, ty2_norm) {
        return true;
    }
    // anyref is a supertype of all concrete GC reference types (builtin
    // layouts, wasm.structref, wasm.arrayref, etc.). When one code path records a field
    // as anyref and another records the concrete type, they are compatible —
    // the field should remain anyref (the wider type).
    let is_anyref_1 = trunk_ir::dialect::wasm::Anyref::matches(ctx, ty1_norm);
    let is_anyref_2 = trunk_ir::dialect::wasm::Anyref::matches(ctx, ty2_norm);
    if is_anyref_1 || is_anyref_2 {
        let other = if is_anyref_1 { ty2_norm } else { ty1_norm };
        // Accept if the other type is a builtin layout or a wasm heap type
        // that is a subtype of anyref.
        // Note: funcref and externref are NOT subtypes of anyref.
        if helpers::builtin_layout_type_idx(ctx, other).is_some() {
            return true;
        }
        if trunk_ir::dialect::wasm::Structref::matches(ctx, other)
            || trunk_ir::dialect::wasm::Arrayref::matches(ctx, other)
            || trunk_ir::dialect::wasm::Eqref::matches(ctx, other)
            || trunk_ir::dialect::wasm::I31ref::matches(ctx, other)
            || trunk_ir::dialect::wasm_gc::Array::matches(ctx, other)
        {
            return true;
        }
    }
    // arrayref is the supertype of every array type, so a concrete array
    // stored in or read from an arrayref field is compatible with it.
    let is_arrayref_1 = trunk_ir::dialect::wasm::Arrayref::matches(ctx, ty1_norm);
    let is_arrayref_2 = trunk_ir::dialect::wasm::Arrayref::matches(ctx, ty2_norm);
    if is_arrayref_1 || is_arrayref_2 {
        let other = if is_arrayref_1 { ty2_norm } else { ty1_norm };
        if trunk_ir::dialect::wasm_gc::Array::matches(ctx, other) {
            return true;
        }
    }
    false
}

/// Record a struct field type
fn record_struct_field(
    ctx: &mut IrContext,
    type_idx: u32,
    builder: &mut GcTypeBuilder,
    field_idx: u32,
    ty: TypeRef,
) -> CompilationResult<()> {
    // Normalize type before storing/comparing
    let ty = normalize_type_for_gc(ctx, ty);

    if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
        let count = builder.field_count.expect("count checked by matches");
        return Err(CompilationError::type_error(format!(
            "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
        )));
    }
    let idx = field_idx as usize;
    if builder.fields.len() <= idx {
        builder.fields.resize_with(idx + 1, || None);
    }
    if let Some(existing) = builder.fields[idx] {
        // Check if types are semantically equivalent
        if !types_equivalent_for_gc(ctx, existing, ty) {
            let existing_data = ctx.get_type(existing);
            let new_data = ctx.get_type(ty);
            return Err(CompilationError::type_error(format!(
                "struct type index {type_idx} field {field_idx} type mismatch: existing={:?} ({}), new={:?} ({})",
                existing,
                existing_data.name.display(ctx),
                ty,
                new_data.name.display(ctx),
            )));
        }
        // A structural type declares its fields; accesses do not refine them.
        if builder.declared {
            return Ok(());
        }
        // `anyref` is the widest compatible reference type. Otherwise, keep
        // the physical struct supertype rather than an equivalent concrete
        // struct type, independent of visitation order.
        let existing_is_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, existing);
        let new_is_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, ty);
        let existing_is_structref = trunk_ir::dialect::wasm::Structref::matches(ctx, existing);
        let new_is_structref = trunk_ir::dialect::wasm::Structref::matches(ctx, ty);
        if new_is_anyref || (!existing_is_anyref && new_is_structref && !existing_is_structref) {
            builder.fields[idx] = Some(ty);
        }
    } else {
        debug!(
            "GC: record_struct_field type_idx={} setting field {} to {:?}",
            type_idx, field_idx, ty
        );
        builder.fields[idx] = Some(ty);
    }
    Ok(())
}

/// The element type `ty` declares, if it is a `wasm_gc.array`.
fn declared_array_element(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    trunk_ir::dialect::wasm_gc::Array::from_type_ref(ctx, ty).map(|array| array.element(ctx))
}

/// Record array element type
fn record_array_elem(
    type_idx: u32,
    builder: &mut GcTypeBuilder,
    ty: TypeRef,
) -> CompilationResult<()> {
    if let Some(existing) = builder.array_elem {
        if existing != ty {
            return Err(CompilationError::type_error(format!(
                "array type index {type_idx} element type mismatch",
            )));
        }
    } else {
        builder.array_elem = Some(ty);
    }
    Ok(())
}

/// Convert a TypeRef to a wasm FieldType for GC type building (arena version).
fn type_to_field_type(
    ctx: &IrContext,
    ty: TypeRef,
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> CompilationResult<FieldType> {
    let val_type = helpers::type_to_valtype(ctx, ty, type_idx_by_type)?;
    Ok(FieldType {
        element_type: StorageType::Val(val_type),
        mutable: true,
    })
}

/// Create the physical WasmGC struct supertype `structref`.
fn intern_wasm_structref(ctx: &mut IrContext) -> TypeRef {
    wasm_dialect::Structref::params().intern(ctx).as_type_ref()
}

/// The supertype index of each GC type in index order, from the `supertype`
/// attribute of the structural structs that received user indices. Builtin
/// types have no supertype. A supertype is a builtin layout or another
/// indexed GC type; the subtype relation is never inferred from fields.
pub(crate) fn collect_gc_supertypes(
    ctx: &IrContext,
    module: Module,
    type_count: usize,
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> CompilationResult<Vec<Option<u32>>> {
    let mut supertypes = vec![None; type_count];
    let Some(Attribute::List(types)) = ctx.op_attrs(module.op()).get(GC_TYPES_ATTR) else {
        return Ok(supertypes);
    };
    for (offset, attr) in types.iter().enumerate() {
        let Attribute::Type(ty) = attr else {
            continue;
        };
        let Some(supertype) =
            wasm_gc::Struct::from_type_ref(ctx, *ty).and_then(|gc_struct| gc_struct.supertype(ctx))
        else {
            continue;
        };
        let index = helpers::builtin_layout_type_idx(ctx, supertype)
            .or_else(|| type_idx_by_type.get(&supertype).copied())
            .ok_or_else(|| {
                CompilationError::invalid_module("GC struct supertype has no type index")
            })?;
        let slot = supertypes
            .get_mut(FIRST_USER_TYPE_IDX as usize + offset)
            .ok_or_else(|| CompilationError::invalid_module("GC type table out of range"))?;
        *slot = Some(index);
    }
    Ok(supertypes)
}

// ============================================================================
// Main collection function
// ============================================================================

/// Collect GC types from wasm dialect operations in a module.
///
/// Traverses all operations to identify struct and array types, recording their
/// field/element types. Returns type definitions and type index mappings.
pub(crate) fn collect_gc_types(
    ctx: &mut IrContext,
    module: Module,
) -> CompilationResult<GcTypesResult> {
    let mut builders: Vec<GcTypeBuilder> = Vec::new();
    let mut type_idx_by_type: HashMap<TypeRef, u32> = HashMap::default();
    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;

    // The GC types that received user indices, in index order. A structural
    // struct type declares its fields, so its builder starts complete.
    let declared_types: Vec<TypeRef> = match ctx.op_attrs(module.op()).get(GC_TYPES_ATTR) {
        Some(Attribute::List(types)) => types
            .iter()
            .map(|attr| match attr {
                Attribute::Type(ty) => Ok(*ty),
                _ => Err(CompilationError::invalid_module(
                    "GC type table entries must be types",
                )),
            })
            .collect::<CompilationResult<_>>()?,
        Some(_) => {
            return Err(CompilationError::invalid_module(
                "GC type table must be a list",
            ));
        }
        None => Vec::new(),
    };
    for (offset, &ty) in declared_types.iter().enumerate() {
        let idx = FIRST_USER_TYPE_IDX + offset as u32;
        register_type(ctx, &mut type_idx_by_type, idx, ty);
        if let Some(fields) = wasm_gc::Struct::from_type_ref(ctx, ty) {
            if builders.len() <= offset {
                builders.resize_with(offset + 1, GcTypeBuilder::new);
            }
            builders[offset] = GcTypeBuilder::declared(fields.fields(ctx));
        }
    }

    // Builtin closure, marker, and evidence layouts are selected by their
    // runtime layout identifier (`helpers::builtin_layout_type_idx`), never
    // by an erased `arrayref` or a struct name.
    // Collect all ops to visit in order (we need to borrow ctx immutably)
    let ops_to_visit = {
        let mut ops = Vec::new();
        fn collect_ops_from_region(ctx: &IrContext, region: RegionRef, ops: &mut Vec<OpRef>) {
            for &block in ctx.region(region).blocks.iter() {
                for &op in ctx.block(block).ops.iter() {
                    // Recurse into nested core.module operations
                    if trunk_ir::dialect::core::Module::matches(ctx, op) {
                        for nested_region in ctx.op_regions(op) {
                            collect_ops_from_region(ctx, nested_region, ops);
                        }
                        continue;
                    }

                    // Visit wasm.func body (recursively including nested regions)
                    if wasm_dialect::Func::matches(ctx, op) {
                        if let Some(func_region) = ctx.op_region(op, 0) {
                            collect_all_ops_recursive(ctx, func_region, ops);
                        }
                    } else {
                        ops.push(op);
                        // Also visit nested regions for control flow ops
                        for nested_region in ctx.op_regions(op) {
                            collect_all_ops_recursive(ctx, nested_region, ops);
                        }
                    }
                }
            }
        }
        fn collect_all_ops_recursive(ctx: &IrContext, region: RegionRef, ops: &mut Vec<OpRef>) {
            for &block in ctx.region(region).blocks.iter() {
                for &op in ctx.block(block).ops.iter() {
                    ops.push(op);
                    for nested_region in ctx.op_regions(op) {
                        collect_all_ops_recursive(ctx, nested_region, ops);
                    }
                }
            }
        }
        collect_ops_from_region(ctx, body, &mut ops);
        ops
    };

    // Process collected ops
    for op in ops_to_visit {
        if ctx.op(op).name.dialect(ctx) != wasm_dialect::DIALECT_NAME {
            continue;
        }

        if wasm_dialect::StructNew::matches(ctx, op) {
            let struct_new =
                wasm_dialect::StructNew::from_op(ctx, op).expect("matched wasm.struct_new");
            let operands = ctx.op_operands(op).to_vec();
            let field_count = operands.len();
            let result_types = ctx.op_result_types(op).to_vec();
            let result_type = result_types.first().copied();
            let type_idx = struct_new.type_idx(ctx);

            if type_idx == MARKER_IDX
                && let Some(ty) = result_type
            {
                register_builtin_evidence_type(ctx, &mut type_idx_by_type, type_idx, ty)?;
            }

            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;

                if matches!(builder.field_count, Some(existing_count) if existing_count != field_count)
                {
                    let existing_count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field count mismatch ({existing_count} vs {field_count})",
                    )));
                }

                builder.field_count = Some(field_count);
                if builder.fields.len() < field_count {
                    builder.fields.resize_with(field_count, || None);
                }

                if let Some(result_ty) = result_type {
                    register_type(ctx, &mut type_idx_by_type, type_idx, result_ty);
                }
                for (field_idx, &value) in operands.iter().enumerate() {
                    let ty = helpers::value_type(ctx, value);
                    let ty_data = ctx.get_type(ty);
                    let field_idx_u32 = u32::try_from(field_idx).map_err(|_| {
                        CompilationError::invalid_module("struct field index out of u32 range")
                    })?;
                    debug!(
                        "GC: struct_new type_idx={} recording field {} with type {}",
                        type_idx,
                        field_idx,
                        ty_data.name.display(ctx)
                    );
                    record_struct_field(ctx, type_idx, builder, field_idx_u32, ty)?;
                }
            }
        } else if wasm_dialect::StructGet::matches(ctx, op) {
            let struct_get =
                wasm_dialect::StructGet::from_op(ctx, op).expect("matched wasm.struct_get");
            let type_idx = struct_get.type_idx(ctx);
            let field_idx = struct_get.field_idx(ctx);
            let operands = ctx.op_operands(op).to_vec();
            if type_idx == MARKER_IDX
                && let Some(&value) = operands.first()
            {
                register_builtin_evidence_type(
                    ctx,
                    &mut type_idx_by_type,
                    type_idx,
                    helpers::value_type(ctx, value),
                )?;
            }
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;

                if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
                    let count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
                    )));
                }
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(ctx, &mut type_idx_by_type, type_idx, ty);
                }
                // Record field type from result type
                // Note: type variables should be resolved to concrete types before emit
                let result_types = ctx.op_result_types(op).to_vec();
                if let Some(&result_ty) = result_types.first() {
                    let result_data = ctx.get_type(result_ty);
                    debug!(
                        "GC: struct_get type_idx={} recording field {} with result_ty {}",
                        type_idx,
                        field_idx,
                        result_data.name.display(ctx)
                    );
                    record_struct_field(ctx, type_idx, builder, field_idx, result_ty)?;
                }
            }
        } else if wasm_dialect::StructSet::matches(ctx, op) {
            let struct_set =
                wasm_dialect::StructSet::from_op(ctx, op).expect("matched wasm.struct_set");
            let operands = ctx.op_operands(op).to_vec();
            let type_idx = struct_set.type_idx(ctx);
            let field_idx = struct_set.field_idx(ctx);
            if type_idx == MARKER_IDX
                && let Some(&value) = operands.first()
            {
                register_builtin_evidence_type(
                    ctx,
                    &mut type_idx_by_type,
                    type_idx,
                    helpers::value_type(ctx, value),
                )?;
            }
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;
                if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
                    let count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
                    )));
                }
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(ctx, &mut type_idx_by_type, type_idx, ty);
                }
                if let Some(&second_operand) = operands.get(1) {
                    let ty = helpers::value_type(ctx, second_operand);
                    record_struct_field(ctx, type_idx, builder, field_idx, ty)?;
                }
            }
        } else if wasm_dialect::ArrayNew::matches(ctx, op)
            || wasm_dialect::ArrayNewDefault::matches(ctx, op)
            || wasm_dialect::ArrayNewFixed::matches(ctx, op)
        {
            let result_types = ctx.op_result_types(op).to_vec();
            let type_idx = wasm_dialect::ArrayNew::from_op(ctx, op)
                .map(|op| op.type_idx(ctx))
                .or_else(|_| {
                    wasm_dialect::ArrayNewDefault::from_op(ctx, op).map(|op| op.type_idx(ctx))
                })
                .or_else(|_| {
                    wasm_dialect::ArrayNewFixed::from_op(ctx, op).map(|op| op.type_idx(ctx))
                })
                .expect("matched indexed array.new operation");
            if type_idx == EVIDENCE_IDX
                && let Some(&ty) = result_types.first()
            {
                register_builtin_evidence_type(ctx, &mut type_idx_by_type, type_idx, ty)?;
            }
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(&result_ty) = result_types.first() {
                    register_type(ctx, &mut type_idx_by_type, type_idx, result_ty);
                }
                // `array.new` takes its element after the size; `array.new_fixed`
                // takes only elements.
                let element_position = if wasm_dialect::ArrayNewFixed::matches(ctx, op) {
                    0
                } else {
                    1
                };
                // A declared result type names the element; an element operand
                // may be a subtype of it. Otherwise the operand's type is the
                // element.
                let element_ty = result_types
                    .first()
                    .and_then(|&ty| declared_array_element(ctx, ty))
                    .or_else(|| {
                        ctx.op_operands(op)
                            .get(element_position)
                            .map(|&element| helpers::value_type(ctx, element))
                    });
                if let Some(ty) = element_ty {
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if wasm_dialect::ArrayGet::matches(ctx, op)
            || wasm_dialect::ArrayGetS::matches(ctx, op)
            || wasm_dialect::ArrayGetU::matches(ctx, op)
        {
            let operands = ctx.op_operands(op).to_vec();
            let type_idx = wasm_dialect::ArrayGet::from_op(ctx, op)
                .map(|op| op.type_idx(ctx))
                .or_else(|_| wasm_dialect::ArrayGetS::from_op(ctx, op).map(|op| op.type_idx(ctx)))
                .or_else(|_| wasm_dialect::ArrayGetU::from_op(ctx, op).map(|op| op.type_idx(ctx)))
                .expect("matched indexed array.get operation");
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(ctx, &mut type_idx_by_type, type_idx, ty);
                }
                // Record element type from result type
                // Note: type variables should be resolved to concrete types before emit
                let result_types = ctx.op_result_types(op).to_vec();
                if let Some(&result_ty) = result_types.first() {
                    record_array_elem(type_idx, builder, result_ty)?;
                }
            }
        } else if wasm_dialect::ArraySet::matches(ctx, op) {
            let array_set =
                wasm_dialect::ArraySet::from_op(ctx, op).expect("matched wasm.array_set");
            let operands = ctx.op_operands(op).to_vec();
            let type_idx = array_set.type_idx(ctx);
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(ctx, &mut type_idx_by_type, type_idx, ty);
                }
                // The stored value may be a subtype of the declared element.
                let element_ty = operands
                    .first()
                    .and_then(|&array| declared_array_element(ctx, helpers::value_type(ctx, array)))
                    .or_else(|| {
                        operands
                            .get(2)
                            .map(|&value| helpers::value_type(ctx, value))
                    });
                if let Some(ty) = element_ty {
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if wasm_dialect::ArrayCopy::matches(ctx, op) {
            let array_copy =
                wasm_dialect::ArrayCopy::from_op(ctx, op).expect("matched wasm.array_copy");
            if let Some(builder) = try_get_builder(&mut builders, array_copy.dst_type_idx(ctx)) {
                builder.kind = GcKind::Array;
            }
            if let Some(builder) = try_get_builder(&mut builders, array_copy.src_type_idx(ctx)) {
                builder.kind = GcKind::Array;
            }
        } else if wasm_dialect::RefNull::matches(ctx, op)
            || wasm_dialect::RefCast::matches(ctx, op)
            || wasm_dialect::RefTest::matches(ctx, op)
        {
            let result_types = ctx.op_result_types(op).to_vec();
            let type_idx = wasm_dialect::RefNull::from_op(ctx, op)
                .map(|op| op.type_idx(ctx))
                .or_else(|_| wasm_dialect::RefCast::from_op(ctx, op).map(|op| op.type_idx(ctx)))
                .or_else(|_| wasm_dialect::RefTest::from_op(ctx, op).map(|op| op.type_idx(ctx)))
                .expect("matched wasm reference operation");
            let Some(type_idx) = type_idx else {
                continue;
            };
            if let Some(&result_ty) = result_types.first() {
                register_type(ctx, &mut type_idx_by_type, type_idx, result_ty);
            }
            if let Some(builder) = try_get_builder(&mut builders, type_idx)
                && builder.kind == GcKind::Unknown
            {
                builder.kind = GcKind::Struct;
            }
        }
    }

    // Build user-defined types from builders
    let mut user_types = Vec::new();
    for builder in builders {
        match builder.kind {
            GcKind::Array => {
                let elem = match builder.array_elem {
                    Some(ty) => type_to_field_type(ctx, ty, &type_idx_by_type)?,
                    None => FieldType {
                        element_type: StorageType::Val(ValType::I32),
                        mutable: false,
                    },
                };
                user_types.push(GcTypeDef::Array(elem));
            }
            GcKind::Struct | GcKind::Unknown => {
                let fields = builder
                    .fields
                    .into_iter()
                    .map(|ty| match ty {
                        Some(ty) => type_to_field_type(ctx, ty, &type_idx_by_type),
                        None => Ok(FieldType {
                            element_type: StorageType::Val(ValType::I32),
                            mutable: false,
                        }),
                    })
                    .collect::<CompilationResult<Vec<_>>>()?;
                user_types.push(GcTypeDef::Struct(fields));
            }
        }
    }

    // Combine builtin types with user-defined types
    let mut result = gc_types::builtin_types();
    result.extend(user_types);

    Ok((result, type_idx_by_type))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc_types::CLOSURE_STRUCT_IDX;
    use trunk_ir::types::{Attribute, TypeDataBuilder};

    #[test]
    fn unrelated_indexed_operations_preserve_abstract_function_parameters() {
        for reference in ["wasm.structref", "wasm.anyref"] {
            for projection in ["get", "null", "cast", "new"] {
                for observer_first in [false, true] {
                    let a = FIRST_USER_TYPE_IDX;
                    let b = a + 1;
                    let operation = match projection {
                        "get" => format!(
                            "%v = wasm.struct_get %arg {{type_idx = {a}, field_idx = 0}} : core.i32"
                        ),
                        "null" => format!("%v = wasm.ref_null {{type_idx = {a}}} : {reference}"),
                        "cast" => format!(
                            "%v = wasm.ref_cast %arg {{target_type = {reference}, type_idx = {a}}} : {reference}"
                        ),
                        "new" => format!(
                            "%x = wasm.i32_const {{value = 0}} : core.i32\n%v = wasm.struct_new %x {{type_idx = {a}}} : {reference}"
                        ),
                        _ => unreachable!(),
                    };
                    let observer = format!(
                        "wasm.func @observe(%arg: {reference}) {{ {operation}\nwasm.return }}"
                    );
                    let accept = format!("wasm.func @accept(%arg: {reference}) {{ wasm.return }}");
                    let functions = if observer_first {
                        format!("{observer}\n{accept}")
                    } else {
                        format!("{accept}\n{observer}")
                    };
                    let mut ctx = IrContext::new();
                    let module = trunk_ir::parser::parse_test_module(
                        &mut ctx,
                        &format!(
                            r#"core.module @test {{
                        !A = wasm_gc.struct<core.i32>
                        !B = wasm_gc.struct<core.i64>
                        wasm.func @make_a() {{
                            %x = wasm.i32_const {{value = 0}} : core.i32
                            %a = wasm.struct_new %x {{type_idx = {a}}} : !A
                            wasm.return
                        }}
                        {functions}
                        wasm.func @main() {{
                            %x = wasm.i64_const {{value = 42}} : core.i64
                            %b = wasm.struct_new %x {{type_idx = {b}}} : !B
                            wasm.call %b {{callee = @accept}}
                            wasm.return
                        }}
                    }}"#
                        ),
                    );
                    let (_, map) = collect_gc_types(&mut ctx, module).unwrap();
                    let a_ty = ctx.type_alias_by_text("A").unwrap();
                    let b_ty = ctx.type_alias_by_text("B").unwrap();
                    assert_eq!(map.get(&a_ty), Some(&a));
                    assert_eq!(map.get(&b_ty), Some(&b));
                    let abstract_ty = TypeDataBuilder::new(reference).intern(&mut ctx);
                    assert!(!map.contains_key(&abstract_ty), "{reference}: {projection}");
                    let binary = crate::emit_module_to_wasm(&mut ctx, module).unwrap();
                    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
                        .validate_all(&binary.bytes)
                        .unwrap_or_else(|error| panic!("{reference}: {projection}: {error}"));
                }
            }
        }
    }

    #[test]
    fn array_of_a_supertype_takes_its_element_from_the_result_type() {
        let mut ctx = IrContext::new();
        let first = FIRST_USER_TYPE_IDX;
        let array = FIRST_USER_TYPE_IDX + 1;
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
                    !A = wasm_gc.struct<core.i32>
                    wasm.func @fill() {{
                        %x = wasm.i32_const {{value = 0}} : core.i32
                        %a = wasm.struct_new %x {{type_idx = {first}}} : !A
                        %n = wasm.i32_const {{value = 2}} : core.i32
                        %array = wasm.array_new %n, %a {{type_idx = {array}}} : wasm_gc.array<wasm.structref>
                        %i = wasm.i32_const {{value = 1}} : core.i32
                        wasm.array_set %array, %i, %a {{type_idx = {array}}}
                        wasm.return
                    }}
                }}"#
            ),
        );
        let (types, _) = collect_gc_types(&mut ctx, module).unwrap();
        assert!(
            matches!(
                &types[array as usize],
                GcTypeDef::Array(element) if matches!(
                    element.element_type,
                    StorageType::Val(ValType::Ref(wasm_encoder::RefType {
                        heap_type: wasm_encoder::HeapType::Abstract {
                            ty: wasm_encoder::AbstractHeapType::Struct,
                            ..
                        },
                        ..
                    }))
                )
            ),
            "{:?}",
            types[array as usize]
        );
        let binary = crate::emit_module_to_wasm(&mut ctx, module).unwrap();
        wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
            .validate_all(&binary.bytes)
            .unwrap();
    }

    #[test]
    fn an_array_without_a_declared_type_takes_its_element_from_the_value() {
        let mut ctx = IrContext::new();
        let array = FIRST_USER_TYPE_IDX;
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
                    wasm.func @fill() {{
                        %n = wasm.i32_const {{value = 2}} : core.i32
                        %x = wasm.i64_const {{value = 7}} : core.i64
                        %array = wasm.array_new %n, %x {{type_idx = {array}}} : wasm.arrayref
                        %i = wasm.i32_const {{value = 1}} : core.i32
                        wasm.array_set %array, %i, %x {{type_idx = {array}}}
                        wasm.return
                    }}
                }}"#
            ),
        );
        let (types, _) = collect_gc_types(&mut ctx, module).unwrap();
        assert!(
            matches!(
                &types[array as usize],
                GcTypeDef::Array(element) if element.element_type == StorageType::Val(ValType::I64)
            ),
            "{:?}",
            types[array as usize]
        );
    }

    #[test]
    fn empty_fixed_array_takes_its_element_from_the_result_type() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
                    wasm.func @empty() {{
                        %array = wasm.array_new_fixed {{type_idx = {FIRST_USER_TYPE_IDX}}} : wasm_gc.array<core.i64>
                        wasm.return
                    }}
                }}"#
            ),
        );
        let (types, _) = collect_gc_types(&mut ctx, module).unwrap();
        let GcTypeDef::Array(element) = &types[FIRST_USER_TYPE_IDX as usize] else {
            panic!("the fixed array's type is an array");
        };
        assert_eq!(element.element_type, StorageType::Val(ValType::I64));
    }

    #[test]
    fn incompatible_named_marker_and_evidence_layouts_are_rejected_before_emission() {
        for (field_type, dispatch_type) in [
            ("core.i64", "core.ptr"),
            ("wasm.anyref", "core.ptr"),
            // A pointer only stands for a reference field.
            ("core.ptr", "core.ptr"),
            // The dispatch field holds the closure base, not an erased anyref.
            ("core.i32", "wasm.anyref"),
        ] {
            for evidence in [false, true] {
                let mut ctx = IrContext::new();
                let producer = if evidence {
                    format!(
                        "%size = wasm.i32_const {{value = 0}} : core.i32\n%value = wasm.array_new_default %size {{type_idx = {EVIDENCE_IDX}}} : wasm_gc.array<!Marker, {{layout = \"evidence\"}}>"
                    )
                } else {
                    format!(
                        "%value = wasm.struct_get %marker {{type_idx = {MARKER_IDX}, field_idx = 0}} : core.i32"
                    )
                };
                let module = trunk_ir::parser::parse_test_module(&mut ctx, &format!(
                    "core.module @test {{
                        !Marker = test.layout<{field_type} {{name = \"ability_id\"}}, core.i32 {{name = \"prompt_tag\"}}, {dispatch_type} {{name = \"tr_dispatch_fn\"}}, core.ptr {{name = \"shadowed\"}}, core.ptr {{name = \"outer\"}}, {{name = \"_Marker\", layout = \"evidence_marker\"}}>
                        wasm.func @test(%marker: !Marker) -> core.i32 {{
                            {producer}
                            wasm.unreachable
                        }}
                    }}"
                ));
                let error = crate::emit_module_to_wasm(&mut ctx, module)
                    .err()
                    .expect("incompatible Marker layout must fail before emission");
                assert!(
                    error.to_string().contains("Marker declaration differs"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn indexed_marker_access_does_not_specialize_abstract_or_unrelated_references() {
        for reference in [
            "wasm.anyref",
            "wasm.structref",
            "wasm.arrayref",
            r#"test.layout<{name = "Other"}>"#,
        ] {
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{
                    wasm.func @test(%marker: {reference}) -> core.i32 {{
                        %value = wasm.struct_get %marker {{type_idx = {MARKER_IDX}, field_idx = 0}} : core.i32
                        wasm.return %value
                    }}
                }}"
                ),
            );
            let function = wasm_dialect::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
            let entry = ctx.region(function.body(&ctx)).blocks[0];
            let ty = ctx.value_ty(ctx.block_args(entry)[0]);
            let (_, map) = collect_gc_types(&mut ctx, module).unwrap();
            // No abstract or unrelated reference acquires a builtin index: not
            // the Marker index from struct_get, and not the Evidence index from
            // an erased arrayref.
            assert_eq!(map.get(&ty), None);
        }
    }

    #[test]
    fn marker_declaration_at_a_different_builtin_index_is_not_registered() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
            !Marker = test.layout<core.i32 {{name = "ability_id"}}, core.i32 {{name = "prompt_tag"}}, core.ptr {{name = "tr_dispatch_fn"}}, core.ptr {{name = "shadowed"}}, core.ptr {{name = "outer"}}, {{name = "_Marker"}}>
            wasm.func @test(%marker: !Marker) -> core.i32 {{
                %value = wasm.struct_get %marker {{type_idx = {CLOSURE_STRUCT_IDX}, field_idx = 0}} : core.i32
                wasm.return %value
            }}
        }}"#
            ),
        );
        let ty = ctx.type_alias_by_text("Marker").unwrap();
        let (_, map) = collect_gc_types(&mut ctx, module).unwrap();
        assert!(!map.contains_key(&ty));
    }

    #[test]
    fn boolean_struct_fields_use_i32_independently_of_collection_order() {
        for stored in ["core.i1", "core.i32"] {
            for projected in ["core.i1", "core.i32", "core.i64", "core.f32", "wasm.anyref"] {
                for projection_first in [false, true] {
                    let index = FIRST_USER_TYPE_IDX;
                    let projection = format!(
                        "wasm.func @observe(%cell: !Cell) {{
%value = wasm.struct_get %cell {{type_idx = {index}, field_idx = 0}} : {projected}
wasm.struct_set %cell, %value {{type_idx = {index}, field_idx = 0}}
wasm.return
}}"
                    );
                    let constructor = format!(
                        "wasm.func @make() {{
%zero = wasm.i32_const {{value = 0}} : {stored}
%cell = wasm.struct_new %zero {{type_idx = {index}}} : !Cell
wasm.return
}}"
                    );
                    let functions = if projection_first {
                        format!("{projection}\n{constructor}")
                    } else {
                        format!("{constructor}\n{projection}")
                    };
                    let mut ctx = IrContext::new();
                    let module = trunk_ir::parser::parse_test_module(
                        &mut ctx,
                        &format!(
                            "core.module @test {{
!Cell = test.ref<{{name = \"Cell\"}}>
{functions}
}}"
                        ),
                    );
                    if matches!(projected, "core.i1" | "core.i32") {
                        let (types, _) = collect_gc_types(&mut ctx, module).unwrap();
                        let GcTypeDef::Struct(fields) = &types[index as usize] else {
                            panic!("struct field expected")
                        };
                        assert_eq!(fields.len(), 1);
                        assert_eq!(fields[0].element_type, StorageType::Val(ValType::I32));
                        let binary = crate::emit_module_to_wasm(&mut ctx, module).unwrap();
                        wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
                            .validate_all(&binary.bytes)
                            .unwrap();
                    } else {
                        let error = collect_gc_types(&mut ctx, module).unwrap_err();
                        assert!(error.to_string().contains("type mismatch"), "{error}");
                    }
                }
            }
        }
    }

    #[test]
    fn record_struct_field_widens_concrete_to_anyref() {
        let mut ctx = IrContext::new();
        let concrete = intern_wasm_structref(&mut ctx);
        let anyref = wasm_dialect::Anyref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let mut builder = GcTypeBuilder::new();

        record_struct_field(&mut ctx, FIRST_USER_TYPE_IDX, &mut builder, 0, concrete)
            .expect("concrete field records");
        record_struct_field(&mut ctx, FIRST_USER_TYPE_IDX, &mut builder, 0, anyref)
            .expect("anyref field widens compatible concrete field");

        assert_eq!(builder.fields, vec![Some(anyref)]);
    }

    #[test]
    fn structural_struct_fields_compare_as_the_struct_supertype() {
        let mut ctx = IrContext::new();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let structural = wasm_gc::Struct::params([i32_ty])
            .intern(&mut ctx)
            .as_type_ref();
        let structref = intern_wasm_structref(&mut ctx);

        assert!(types_equivalent_for_gc(&mut ctx, structural, structref));
        assert_eq!(normalize_type_for_gc(&mut ctx, structural), structural);
    }

    #[test]
    fn concrete_arrays_compare_as_arrayref_and_anyref() {
        let mut ctx = IrContext::new();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let array = wasm_gc::Array::params(i32_ty)
            .intern(&mut ctx)
            .as_type_ref();
        let arrayref = trunk_ir::dialect::wasm::Arrayref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let anyref = trunk_ir::dialect::wasm::Anyref::params()
            .intern(&mut ctx)
            .as_type_ref();

        assert!(types_equivalent_for_gc(&mut ctx, array, arrayref));
        assert!(types_equivalent_for_gc(&mut ctx, anyref, array));
    }

    #[test]
    fn types_of_one_builtin_layout_are_gc_equivalent_without_a_new_type() {
        let mut ctx = IrContext::new();
        let mut closure_layout = |name: &'static str| {
            let layout = ctx.string_attr(crate::gc_types::CLOSURE_LAYOUT);
            TypeDataBuilder::new(name)
                .attr(trunk_ir::types::LAYOUT_ATTR, layout)
                .intern(&mut ctx)
        };
        let reference = closure_layout("test.reference");
        let declaration = closure_layout("test.declaration");
        let bytes_layout = ctx.string_attr(crate::gc_types::BYTES_LAYOUT);
        let bytes = TypeDataBuilder::new("test.bytes")
            .attr(trunk_ir::types::LAYOUT_ATTR, bytes_layout)
            .intern(&mut ctx);
        let types_before = ctx.types().iter().count();

        assert!(types_equivalent_for_gc(&mut ctx, reference, declaration));
        assert!(!types_equivalent_for_gc(&mut ctx, reference, bytes));
        assert_eq!(normalize_type_for_gc(&mut ctx, reference), reference);
        assert_eq!(ctx.types().iter().count(), types_before);
    }

    #[test]
    fn same_named_types_with_different_attributes_are_not_gc_equivalent() {
        let mut ctx = IrContext::new();
        let name_attr = ctx.string_attr("String");
        let canonical = TypeDataBuilder::new("test.named")
            .attr("name", name_attr)
            .attr("layout", Attribute::Bool(true))
            .intern(&mut ctx);
        let name_attr = ctx.string_attr("String");
        let unrelated = TypeDataBuilder::new("test.named")
            .attr("name", name_attr)
            .attr("layout", Attribute::Bool(false))
            .intern(&mut ctx);

        assert!(!types_equivalent_for_gc(&mut ctx, canonical, unrelated));
    }

    #[test]
    fn structural_structs_take_their_fields_from_their_type() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Inner = wasm_gc.struct<core.i32, core.f64>
  !Outer = wasm_gc.struct<core.i32, !Inner, wasm.structref>
  wasm.func @main() -> core.nil {
    %outer = wasm_gc.ref_null {target_type = !Outer} : !Outer
    wasm.return
  }
}"#,
        );
        crate::passes::wasm_gc_to_wasm::lower(&mut ctx, module);

        let (types, _) = collect_gc_types(&mut ctx, module).unwrap();

        // No operation reads a field, and the inner struct is named only by
        // the outer one's field; both declare every field.
        let field = |ty| FieldType {
            element_type: StorageType::Val(ty),
            mutable: true,
        };
        let inner = FIRST_USER_TYPE_IDX + 1;
        let inner_ref = ValType::Ref(wasm_encoder::RefType {
            nullable: true,
            heap_type: wasm_encoder::HeapType::Concrete(inner),
        });
        let structref = ValType::Ref(wasm_encoder::RefType {
            nullable: true,
            heap_type: wasm_encoder::HeapType::Abstract {
                shared: false,
                ty: wasm_encoder::AbstractHeapType::Struct,
            },
        });
        let user = &types[FIRST_USER_TYPE_IDX as usize..];
        assert!(matches!(
            user,
            [GcTypeDef::Struct(outer), GcTypeDef::Struct(inner)]
                if *outer == [field(ValType::I32), field(inner_ref), field(structref)]
                    && *inner == [field(ValType::I32), field(ValType::F64)]
        ));
    }

    #[test]
    fn accesses_must_match_declared_field_types() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = wasm_gc.struct<core.i32, core.f64>
  wasm.func @main() -> core.nil {
    %s = wasm_gc.ref_null {target_type = !S} : !S
    %field = wasm_gc.struct_get %s {type = !S, field_idx = 1} : core.i32
    wasm.return
  }
}"#,
        );
        crate::passes::wasm_gc_to_wasm::lower(&mut ctx, module);

        let error = collect_gc_types(&mut ctx, module).unwrap_err();

        assert!(
            error.to_string().contains("field 1 type mismatch"),
            "{error}"
        );
    }
}
