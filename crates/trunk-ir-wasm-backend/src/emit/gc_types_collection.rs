//! GC type collection from wasm dialect operations (arena IR version).
//!
//! This module traverses wasm operations to collect WebAssembly GC type
//! definitions (structs and arrays) and build the type index mappings.

use std::collections::HashMap;

use tracing::debug;

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::types::{Attribute, TypeData};
use wasm_encoder::{FieldType, StorageType, ValType};

use crate::gc_types::{
    self, CLOSURE_STRUCT_IDX, CONTINUATION_IDX, EVIDENCE_IDX, FIRST_USER_TYPE_IDX, GcTypeDef,
    MARKER_IDX, RESUME_WRAPPER_IDX, STEP_IDX,
};
use crate::{CompilationError, CompilationResult};

use super::helpers;

// Symbol definitions - reuse from parent module
trunk_ir::symbols! {
    ATTR_TYPE_IDX => "type_idx",
    ATTR_TYPE => "type",
    ATTR_TARGET_TYPE => "target_type",
    ATTR_HEAP_TYPE => "heap_type",
}

/// Checked conversion from IntBits (u64) to u32.
fn intbits_to_u32(value: u64) -> CompilationResult<u32> {
    u32::try_from(value).map_err(|_| {
        CompilationError::invalid_attribute(format!("IntBits value {} out of u32 range", value))
    })
}

/// Checked conversion from IntBits (u64) to usize.
fn intbits_to_usize(value: u64) -> CompilationResult<usize> {
    usize::try_from(value).map_err(|_| {
        CompilationError::invalid_attribute(format!("IntBits value {} out of usize range", value))
    })
}

/// Result type for GC type collection.
/// Returns: (type_defs, type_idx_by_type, placeholder_struct_type_idx)
pub(crate) type GcTypesResult = (
    Vec<GcTypeDef>,
    HashMap<TypeRef, u32>,
    HashMap<(TypeRef, usize), u32>,
);

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
}

impl GcTypeBuilder {
    fn new() -> Self {
        Self {
            kind: GcKind::Unknown,
            fields: Vec::new(),
            array_elem: None,
            field_count: None,
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Returns true if this is a built-in type (0-8) that shouldn't use a builder
fn is_builtin_type(idx: u32) -> bool {
    idx < FIRST_USER_TYPE_IDX
}

/// Check if a type is an abstract WASM heap type that should NOT get a concrete GC type index.
///
/// Abstract types (anyref, i31ref, structref, funcref, etc.) are WASM built-in types.
/// They must NOT be allocated concrete GC struct/array indices because they don't correspond
/// to user-defined type definitions. If registered, they create spurious empty `(struct)`
/// definitions that cause `ref.cast` to emit invalid concrete type references instead of
/// abstract heap type references.
///
/// Note: `wasm.arrayref` may be pre-registered as EVIDENCE_IDX — that's handled by the
/// existing lookup path (which finds it before reaching the allocation code).
fn is_abstract_wasm_heap_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("wasm") {
        return false;
    }
    let name = data.name;
    name == Symbol::new("anyref")
        || name == Symbol::new("i31ref")
        || name == Symbol::new("structref")
        || name == Symbol::new("funcref")
        || name == Symbol::new("externref")
        || name == Symbol::new("eqref")
}

/// Get the builtin struct type index for closure, continuation, or resume wrapper types.
///
/// Returns `Some(idx)` if the type is one of the three builtin struct types,
/// `None` for all other types.
fn get_builtin_struct_idx(ctx: &IrContext, ty: TypeRef) -> Option<u32> {
    if helpers::is_closure_struct_type(ctx, ty) {
        Some(CLOSURE_STRUCT_IDX)
    } else if helpers::is_continuation_struct_type(ctx, ty) {
        Some(CONTINUATION_IDX)
    } else if helpers::is_resume_wrapper_struct_type(ctx, ty) {
        Some(RESUME_WRAPPER_IDX)
    } else {
        None
    }
}

/// Get or create a builder for a user-defined type.
/// Returns None for built-in types (0-8) which are predefined.
fn try_get_builder(builders: &mut Vec<GcTypeBuilder>, idx: u32) -> Option<&mut GcTypeBuilder> {
    // Skip built-in types (0-8) as they are predefined
    if is_builtin_type(idx) {
        return None;
    }
    // Subtract FIRST_USER_TYPE_IDX because indices 0-8 are reserved for built-in types
    // User type indices start at FIRST_USER_TYPE_IDX
    let adjusted_idx = (idx - FIRST_USER_TYPE_IDX) as usize;
    if builders.len() <= adjusted_idx {
        builders.resize_with(adjusted_idx + 1, GcTypeBuilder::new);
    }
    Some(&mut builders[adjusted_idx])
}

/// Register a type in the type_idx_by_type map
fn register_type(type_idx_by_type: &mut HashMap<TypeRef, u32>, idx: u32, ty: TypeRef) {
    type_idx_by_type.entry(ty).or_insert(idx);
}

/// Get the canonical type name for comparison.
///
/// For variant instance types, returns the base enum's name.
/// For adt.enum/struct types, returns the name attribute.
fn get_canonical_type_name(ctx: &IrContext, ty: TypeRef) -> Option<Symbol> {
    let data = ctx.types.get(ty);

    // Check if this is a variant instance type (adt type with "base_enum" attr)
    if data.dialect == Symbol::new("adt")
        && let Some(Attribute::Type(base_enum)) = data.attrs.get(&Symbol::new("base_enum"))
    {
        let base_data = ctx.types.get(*base_enum);
        if let Some(Attribute::Symbol(name_sym)) = base_data.attrs.get(&Symbol::new("name")) {
            return Some(*name_sym);
        }
    }

    // Check if this is an adt.enum type
    if data.dialect == Symbol::new("adt")
        && data.name == Symbol::new("enum")
        && let Some(Attribute::Symbol(name_sym)) = data.attrs.get(&Symbol::new("name"))
    {
        return Some(*name_sym);
    }

    // Check if this is an adt.struct type
    if data.dialect == Symbol::new("adt")
        && data.name == Symbol::new("struct")
        && let Some(Attribute::Symbol(name_sym)) = data.attrs.get(&Symbol::new("name"))
    {
        return Some(*name_sym);
    }

    None
}

/// Normalize a type for GC struct field comparison.
///
/// Normalizes tribute_rt types and variant instances to their canonical form.
fn normalize_type_for_gc(ctx: &IrContext, ty: TypeRef) -> TypeRef {
    let data = ctx.types.get(ty);

    // Normalize variant instance types to their base enum
    if data.dialect == Symbol::new("adt")
        && let Some(Attribute::Type(base_enum)) = data.attrs.get(&Symbol::new("base_enum"))
    {
        return *base_enum;
    }

    // Note: tribute_rt types (int, nat, bool, float, any, intref) should be
    // converted to core/wasm types by normalize_primitive_types pass before emit.
    ty
}

/// Check if two types are semantically equivalent for GC struct fields.
fn types_equivalent_for_gc(ctx: &IrContext, ty1: TypeRef, ty2: TypeRef) -> bool {
    // First try direct comparison
    if ty1 == ty2 {
        return true;
    }
    // Normalize both types
    let ty1_norm = normalize_type_for_gc(ctx, ty1);
    let ty2_norm = normalize_type_for_gc(ctx, ty2);
    if ty1_norm == ty2_norm {
        return true;
    }
    // Compare canonical names for user-defined types
    // (e.g., adt.struct with name=Foo vs adt.enum with base_enum Foo)
    if let (Some(name1), Some(name2)) = (
        get_canonical_type_name(ctx, ty1),
        get_canonical_type_name(ctx, ty2),
    ) && name1 == name2
    {
        return true;
    }
    // anyref is a supertype of all concrete GC reference types (adt.struct,
    // wasm.structref, wasm.arrayref, etc.). When one code path records a field
    // as anyref and another records the concrete type, they are compatible —
    // the field should remain anyref (the wider type).
    let is_anyref_1 = helpers::is_type(ctx, ty1_norm, "wasm", "anyref");
    let is_anyref_2 = helpers::is_type(ctx, ty2_norm, "wasm", "anyref");
    if is_anyref_1 || is_anyref_2 {
        let other = if is_anyref_1 { ty2_norm } else { ty1_norm };
        // Accept if the other type is an ADT reference (adt.struct, adt.enum)
        // or a wasm heap type that is a subtype of anyref.
        // Note: funcref and externref are NOT subtypes of anyref.
        let other_data = ctx.types.get(other);
        let adt_dialect = Symbol::new("adt");
        if other_data.dialect == adt_dialect {
            return true;
        }
        if helpers::is_type(ctx, other, "wasm", "structref")
            || helpers::is_type(ctx, other, "wasm", "arrayref")
            || helpers::is_type(ctx, other, "wasm", "eqref")
            || helpers::is_type(ctx, other, "wasm", "i31ref")
        {
            return true;
        }
    }
    false
}

/// Record a struct field type
fn record_struct_field(
    ctx: &IrContext,
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
            return Err(CompilationError::type_error(format!(
                "struct type index {type_idx} field {field_idx} type mismatch: existing={:?}, new={:?}",
                existing, ty
            )));
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

use super::helpers::{attr_field_idx, attr_u32};

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

/// Create an adt.struct TypeRef with a given name attribute.
fn intern_named_adt_struct(ctx: &mut IrContext, name: &'static str) -> TypeRef {
    let mut attrs = std::collections::BTreeMap::new();
    attrs.insert(Symbol::new("name"), Attribute::Symbol(Symbol::new(name)));
    ctx.types.intern(TypeData {
        dialect: Symbol::new("adt"),
        name: Symbol::new("struct"),
        params: Default::default(),
        attrs,
    })
}

/// Create a wasm.arrayref TypeRef.
fn intern_wasm_arrayref(ctx: &mut IrContext) -> TypeRef {
    ctx.types.intern(TypeData {
        dialect: Symbol::new("wasm"),
        name: Symbol::new("arrayref"),
        params: Default::default(),
        attrs: Default::default(),
    })
}

/// Check if a TypeRef is wasm.structref.
fn is_structref(ctx: &IrContext, ty: TypeRef) -> bool {
    helpers::is_type(ctx, ty, "wasm", "structref")
}

/// Check if a TypeRef is an adt.struct (non-builtin).
fn is_adt_struct(ctx: &IrContext, ty: TypeRef) -> bool {
    helpers::is_type(ctx, ty, "adt", "struct")
}

/// Get the number of fields for an adt.struct TypeRef.
/// In arena IR, field count is derived from params length.
fn get_adt_struct_field_count(ctx: &IrContext, ty: TypeRef) -> Option<usize> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return None;
    }
    Some(data.params.len())
}

// ============================================================================
// Main collection function
// ============================================================================

/// Collect GC types from wasm dialect operations in a module.
///
/// Traverses all operations to identify struct and array types, recording their
/// field/element types. Returns type definitions, type index mappings, and
/// placeholder struct mappings.
pub(crate) fn collect_gc_types(
    ctx: &mut IrContext,
    module: Module,
) -> CompilationResult<GcTypesResult> {
    use std::collections::HashSet;

    let wasm_dialect = Symbol::new("wasm");
    let mut builders: Vec<GcTypeBuilder> = Vec::new();
    let mut type_idx_by_type: HashMap<TypeRef, u32> = HashMap::new();
    // For placeholder types like wasm.structref, use (type, field_count) as key
    // to handle multiple structs with same placeholder type but different field counts
    let mut placeholder_struct_type_idx: HashMap<(TypeRef, usize), u32> = HashMap::new();

    // Phase 1: Collect explicit type_idx values from all operations
    // This ensures placeholder allocation doesn't conflict with explicit indices
    // (covers struct_new, ref_cast, and any other op with an explicit type_idx attribute)
    let mut reserved_indices: HashSet<u32> = HashSet::new();
    fn collect_reserved_indices(
        ctx: &IrContext,
        region: RegionRef,
        reserved: &mut HashSet<u32>,
    ) -> CompilationResult<()> {
        for &block in ctx.region(region).blocks.iter() {
            for &op in ctx.block(block).ops.iter() {
                if let Some(Attribute::IntBits(idx)) = ctx.op(op).attributes.get(&ATTR_TYPE_IDX()) {
                    reserved.insert(intbits_to_u32(*idx)?);
                }
                for &nested_region in ctx.op(op).regions.iter() {
                    collect_reserved_indices(ctx, nested_region, reserved)?;
                }
            }
        }
        Ok(())
    }
    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;
    collect_reserved_indices(ctx, body, &mut reserved_indices)?;
    debug!(
        "GC: collected {} reserved type indices from explicit struct_new: {:?}",
        reserved_indices.len(),
        reserved_indices
    );

    // Register builtin types in type_idx_by_type:
    // 0: BoxedF64, 1: BytesArray, 2: BytesStruct, 3: Step, 4: ClosureStruct, 5: Marker,
    // 6: Evidence, 7: Continuation, 8: ResumeWrapper
    // Step marker type needs to be registered so wasm.if can use it
    let step_ty = intern_named_adt_struct(ctx, "_Step");
    type_idx_by_type.insert(step_ty, STEP_IDX);
    // Abstract wasm.arrayref maps to EVIDENCE_IDX because type_converter lowers
    // all core::Array types to wasm::Arrayref (which erases element type info).
    // Currently the only array in the system is the evidence array, so this is safe.
    // TODO: if user-defined arrays are added, this will need per-element-type indices.
    let arrayref_ty = intern_wasm_arrayref(ctx);
    type_idx_by_type.insert(arrayref_ty, EVIDENCE_IDX);
    // Marker ADT type needs to be registered so wasm.struct_new for Marker reuses MARKER_IDX
    // instead of creating a separate user type index.
    let marker_ty = intern_named_adt_struct(ctx, "_Marker");
    type_idx_by_type.insert(marker_ty, MARKER_IDX);
    // Evidence ADT type (core.array(Marker)) — use a core.array type with marker param
    let evidence_ty = {
        let mut attrs = std::collections::BTreeMap::new();
        // No additional attrs needed, params carry the element type
        let _ = &mut attrs;
        ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("array"),
            params: trunk_ir::smallvec::smallvec![marker_ty],
            attrs,
        })
    };
    type_idx_by_type.insert(evidence_ty, EVIDENCE_IDX);
    // Continuation ADT type (_Continuation) maps to CONTINUATION_IDX
    let continuation_ty = intern_named_adt_struct(ctx, "_Continuation");
    type_idx_by_type.insert(continuation_ty, CONTINUATION_IDX);
    // ResumeWrapper ADT type (_ResumeWrapper) maps to RESUME_WRAPPER_IDX
    let resume_wrapper_ty = intern_named_adt_struct(ctx, "_ResumeWrapper");
    type_idx_by_type.insert(resume_wrapper_ty, RESUME_WRAPPER_IDX);

    // Start at FIRST_USER_TYPE_IDX since indices 0-8 are reserved for built-in types
    let mut next_type_idx: u32 = FIRST_USER_TYPE_IDX;

    // Helper to get next available type_idx, skipping reserved indices
    let next_available_idx = |next_type_idx: &mut u32, reserved: &HashSet<u32>| -> u32 {
        while reserved.contains(next_type_idx) {
            *next_type_idx += 1;
        }
        let idx = *next_type_idx;
        *next_type_idx += 1;
        idx
    };

    // Collect all ops to visit in order (we need to borrow ctx immutably)
    let ops_to_visit = {
        let mut ops = Vec::new();
        fn collect_ops_from_region(ctx: &IrContext, region: RegionRef, ops: &mut Vec<OpRef>) {
            let wasm_dialect = Symbol::new("wasm");
            let core_dialect = Symbol::new("core");
            let module_name = Symbol::new("module");
            let func_name = Symbol::new("func");

            for &block in ctx.region(region).blocks.iter() {
                for &op in ctx.block(block).ops.iter() {
                    let op_data = ctx.op(op);
                    let dialect = op_data.dialect;
                    let name = op_data.name;

                    // Recurse into nested core.module operations
                    if dialect == core_dialect && name == module_name {
                        for &nested_region in op_data.regions.iter() {
                            collect_ops_from_region(ctx, nested_region, ops);
                        }
                        continue;
                    }

                    // Visit wasm.func body (recursively including nested regions)
                    if dialect == wasm_dialect && name == func_name {
                        if let Some(&func_region) = op_data.regions.first() {
                            collect_all_ops_recursive(ctx, func_region, ops);
                        }
                    } else {
                        ops.push(op);
                        // Also visit nested regions for control flow ops
                        for &nested_region in op_data.regions.iter() {
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
                    for &nested_region in ctx.op(op).regions.iter() {
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
        let op_data = ctx.op(op);
        if op_data.dialect != wasm_dialect {
            continue;
        }

        if wasm_dialect::StructNew::matches(ctx, op) {
            let attrs = &ctx.op(op).attributes;
            let operands = ctx.op_operands(op).to_vec();
            let field_count = operands.len();
            let result_types = ctx.op_result_types(op).to_vec();
            let result_type = result_types.first().copied();

            // Check if this uses a placeholder type (wasm.structref) that allows
            // multiple structs with same type but different field counts.
            // Check both the `type` attribute and the result type for placeholder.
            let placeholder_type_from_attr = attrs.get(&ATTR_TYPE()).and_then(|attr| {
                if let Attribute::Type(ty) = attr {
                    if is_structref(ctx, *ty) {
                        Some(*ty)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            let placeholder_type_from_result = result_type.and_then(|ty| {
                if is_structref(ctx, ty) {
                    Some(ty)
                } else {
                    None
                }
            });

            let placeholder_type = placeholder_type_from_attr.or(placeholder_type_from_result);

            // First check for explicit type_idx attribute (set by wasm_gc_type_assign pass)
            let type_idx = if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
                let idx = u32::try_from(*idx).map_err(|_| {
                    CompilationError::invalid_attribute(format!(
                        "type_idx value {} out of u32 range",
                        idx
                    ))
                })?;
                // Advance next_type_idx to avoid collision with explicit indices
                next_type_idx = next_type_idx.max(idx.saturating_add(1));
                debug!(
                    "GC: struct_new using explicit type_idx={} from attribute (field_count={})",
                    idx, field_count
                );
                idx
            } else if let Some(ty) = placeholder_type {
                // For placeholder types without explicit type_idx, use (type, field_count) as key
                let key = (ty, field_count);
                if let Some(&idx) = placeholder_struct_type_idx.get(&key) {
                    debug!(
                        "GC: struct_new reusing existing type_idx={} for placeholder (field_count={})",
                        idx, field_count
                    );
                    idx
                } else {
                    let idx = next_available_idx(&mut next_type_idx, &reserved_indices);
                    placeholder_struct_type_idx.insert(key, idx);
                    // Note: Placeholder types are NOT inserted into type_idx_by_type
                    // They are only stored in placeholder_struct_type_idx to avoid
                    // confusion and ensure proper lookup via (type, field_count) key
                    debug!(
                        "GC: struct_new allocated type_idx={} for placeholder (field_count={})",
                        idx, field_count
                    );
                    idx
                }
            } else {
                // For regular types, use standard type_idx lookup with result type as fallback
                let inferred_type = result_type;
                let Some(idx) = get_type_idx(
                    ctx,
                    attrs,
                    &mut type_idx_by_type,
                    &mut next_type_idx,
                    inferred_type,
                    &reserved_indices,
                )?
                else {
                    continue;
                };
                idx
            };

            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;

                // For placeholder types, we allow different field counts via different type_idx
                // For explicit type_idx, check for mismatch (error case)
                if placeholder_type.is_none()
                    && matches!(builder.field_count, Some(existing_count) if existing_count != field_count)
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
                    register_type(&mut type_idx_by_type, type_idx, result_ty);
                }
                for (field_idx, &value) in operands.iter().enumerate() {
                    let ty = helpers::value_type(ctx, value);
                    let ty_data = ctx.types.get(ty);
                    let field_idx_u32 = u32::try_from(field_idx).map_err(|_| {
                        CompilationError::invalid_module("struct field index out of u32 range")
                    })?;
                    debug!(
                        "GC: struct_new type_idx={} recording field {} with type {}.{}",
                        type_idx, field_idx, ty_data.dialect, ty_data.name
                    );
                    record_struct_field(ctx, type_idx, builder, field_idx_u32, ty)?;
                }
            }
        } else if wasm_dialect::StructGet::matches(ctx, op) {
            let attrs = &ctx.op(op).attributes;

            // Honor explicit type_idx attribute first (set by wasm_gc_type_assign pass),
            // matching the behavior of struct_new and ref_cast.
            let explicit_type_idx = get_type_idx(
                ctx,
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                None,
                &reserved_indices,
            )?;

            // Check if this uses a placeholder type (wasm.structref) that allows
            // multiple structs with same type but different field counts
            let is_placeholder_type = attrs
                .get(&ATTR_TYPE())
                .map(|attr| {
                    if let Attribute::Type(ty) = attr {
                        is_structref(ctx, *ty)
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            // Check if this is a builtin struct type
            let builtin_type_idx = attrs.get(&ATTR_TYPE()).and_then(|attr| {
                if let Attribute::Type(ty) = attr {
                    get_builtin_struct_idx(ctx, *ty)
                } else {
                    None
                }
            });

            // Check if this uses an adt.struct type (which should also use placeholder lookup)
            // Returns (adt_struct_type, field_count) if it's an adt.struct type
            let adt_struct_info = attrs.get(&ATTR_TYPE()).and_then(|attr| {
                if let Attribute::Type(ty) = attr {
                    if is_adt_struct(ctx, *ty) && get_builtin_struct_idx(ctx, *ty).is_none() {
                        get_adt_struct_field_count(ctx, *ty).map(|fc| (*ty, fc))
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            let type_idx = if let Some(idx) = explicit_type_idx {
                // Explicit type_idx (from ATTR_TYPE_IDX or type_idx_by_type) takes priority
                idx
            } else if let Some(idx) = builtin_type_idx {
                idx
            } else if is_placeholder_type {
                // For placeholder types, use (type, field_count) as key
                let ty = match attrs.get(&ATTR_TYPE()) {
                    Some(Attribute::Type(ty)) => *ty,
                    _ => unreachable!("checked above"),
                };
                let Some(Attribute::IntBits(field_count)) = attrs.get(&Symbol::new("field_count"))
                else {
                    return Err(CompilationError::missing_attribute("field_count"));
                };
                let field_count = intbits_to_usize(*field_count)?;
                let key = (ty, field_count);
                if let Some(&idx) = placeholder_struct_type_idx.get(&key) {
                    idx
                } else {
                    // Allocate new type_idx for this placeholder, skipping reserved indices
                    let idx = next_available_idx(&mut next_type_idx, &reserved_indices);
                    placeholder_struct_type_idx.insert(key, idx);
                    debug!(
                        "GC: struct_get allocated type_idx={} for placeholder (field_count={})",
                        idx, field_count
                    );
                    idx
                }
            } else if let Some((adt_struct_ty, field_count)) = adt_struct_info {
                // For adt.struct types, first check type_idx_by_type (where function
                // param types are registered), then fall back to placeholder lookup.
                // This ensures struct_get uses the same type_idx as function params.
                if let Some(&idx) = type_idx_by_type.get(&adt_struct_ty) {
                    debug!(
                        "GC: struct_get reusing type_idx={} for adt.struct {:?} from type_idx_by_type",
                        idx, adt_struct_ty
                    );
                    idx
                } else {
                    // Fall back to placeholder lookup with (type, field_count) key
                    let key = (adt_struct_ty, field_count);
                    if let Some(&idx) = placeholder_struct_type_idx.get(&key) {
                        debug!(
                            "GC: struct_get reusing type_idx={} for adt.struct (field_count={})",
                            idx, field_count
                        );
                        idx
                    } else {
                        // Allocate new type_idx for this adt.struct type, skipping reserved indices
                        let idx = next_available_idx(&mut next_type_idx, &reserved_indices);
                        placeholder_struct_type_idx.insert(key, idx);
                        debug!(
                            "GC: struct_get allocated type_idx={} for adt.struct (field_count={})",
                            idx, field_count
                        );
                        idx
                    }
                }
            } else {
                // For regular types, use standard type_idx lookup
                let operands = ctx.op_operands(op).to_vec();
                let inferred_type = operands.first().map(|&v| helpers::value_type(ctx, v));
                let Some(idx) = get_type_idx(
                    ctx,
                    attrs,
                    &mut type_idx_by_type,
                    &mut next_type_idx,
                    inferred_type,
                    &reserved_indices,
                )?
                else {
                    continue;
                };
                idx
            };

            let field_idx = attr_field_idx(attrs)?;
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;

                // For placeholder types, set field_count from attribute if not already set
                if is_placeholder_type
                    && builder.field_count.is_none()
                    && let Some(Attribute::IntBits(fc)) = attrs.get(&Symbol::new("field_count"))
                {
                    let fc = intbits_to_usize(*fc)?;
                    builder.field_count = Some(fc);
                    if builder.fields.len() < fc {
                        builder.fields.resize_with(fc, || None);
                    }
                }

                // For adt.struct types, set field_count from the type's fields attribute
                if let Some((_, fc)) = adt_struct_info
                    && builder.field_count.is_none()
                {
                    builder.field_count = Some(fc);
                    if builder.fields.len() < fc {
                        builder.fields.resize_with(fc, || None);
                    }
                }

                if matches!(builder.field_count, Some(count) if field_idx as usize >= count) {
                    let count = builder.field_count.expect("count checked by matches");
                    return Err(CompilationError::type_error(format!(
                        "struct type index {type_idx} field index {field_idx} out of bounds (fields: {count})",
                    )));
                }
                let operands = ctx.op_operands(op).to_vec();
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                // Record field type from result type
                // Note: type variables should be resolved to concrete types before emit
                let result_types = ctx.op_result_types(op).to_vec();
                if let Some(&result_ty) = result_types.first() {
                    let result_data = ctx.types.get(result_ty);
                    debug!(
                        "GC: struct_get type_idx={} recording field {} with result_ty {}.{}",
                        type_idx, field_idx, result_data.dialect, result_data.name
                    );
                    record_struct_field(ctx, type_idx, builder, field_idx, result_ty)?;
                }
            }
        } else if wasm_dialect::StructSet::matches(ctx, op) {
            let attrs = &ctx.op(op).attributes;
            // Infer type from operand[0] (the struct ref)
            let operands = ctx.op_operands(op).to_vec();
            let inferred_type = operands.first().map(|&v| helpers::value_type(ctx, v));
            let Some(type_idx) = get_type_idx(
                ctx,
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
                &reserved_indices,
            )?
            else {
                continue;
            };
            let field_idx = attr_field_idx(attrs)?;
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
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                if let Some(&second_operand) = operands.get(1) {
                    let ty = helpers::value_type(ctx, second_operand);
                    record_struct_field(ctx, type_idx, builder, field_idx, ty)?;
                }
            }
        } else if wasm_dialect::ArrayNew::matches(ctx, op)
            || wasm_dialect::ArrayNewDefault::matches(ctx, op)
        {
            let attrs = &ctx.op(op).attributes;
            // Infer type from result type
            let result_types = ctx.op_result_types(op).to_vec();
            let inferred_type = result_types.first().copied();
            let Some(type_idx) = get_type_idx(
                ctx,
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
                &reserved_indices,
            )?
            else {
                continue;
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(&result_ty) = result_types.first() {
                    register_type(&mut type_idx_by_type, type_idx, result_ty);
                }
                let operands = ctx.op_operands(op).to_vec();
                if let Some(&second_operand) = operands.get(1) {
                    let ty = helpers::value_type(ctx, second_operand);
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if wasm_dialect::ArrayGet::matches(ctx, op)
            || wasm_dialect::ArrayGetS::matches(ctx, op)
            || wasm_dialect::ArrayGetU::matches(ctx, op)
        {
            let attrs = &ctx.op(op).attributes;
            // Infer type from operand[0] (the array ref)
            let operands = ctx.op_operands(op).to_vec();
            let inferred_type = operands.first().map(|&v| helpers::value_type(ctx, v));
            let Some(type_idx) = get_type_idx(
                ctx,
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
                &reserved_indices,
            )?
            else {
                continue;
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                // Record element type from result type
                // Note: type variables should be resolved to concrete types before emit
                let result_types = ctx.op_result_types(op).to_vec();
                if let Some(&result_ty) = result_types.first() {
                    record_array_elem(type_idx, builder, result_ty)?;
                }
            }
        } else if wasm_dialect::ArraySet::matches(ctx, op) {
            let attrs = &ctx.op(op).attributes;
            // Infer type from operand[0] (the array ref)
            let operands = ctx.op_operands(op).to_vec();
            let inferred_type = operands.first().map(|&v| helpers::value_type(ctx, v));
            let Some(type_idx) = get_type_idx(
                ctx,
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
                &reserved_indices,
            )?
            else {
                continue;
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(&first_operand) = operands.first() {
                    let ty = helpers::value_type(ctx, first_operand);
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                if let Some(&third_operand) = operands.get(2) {
                    let ty = helpers::value_type(ctx, third_operand);
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if wasm_dialect::ArrayCopy::matches(ctx, op) {
            // array_copy has dst_type_idx: u32 and src_type_idx: u32 attributes
            let attrs = &ctx.op(op).attributes;
            if let Some(&Attribute::IntBits(dst_idx)) = attrs.get(&Symbol::new("dst_type_idx")) {
                let dst_type_idx = intbits_to_u32(dst_idx)?;
                if let Some(builder) = try_get_builder(&mut builders, dst_type_idx) {
                    builder.kind = GcKind::Array;
                }
            }
            if let Some(&Attribute::IntBits(src_idx)) = attrs.get(&Symbol::new("src_type_idx")) {
                let src_type_idx = intbits_to_u32(src_idx)?;
                if let Some(builder) = try_get_builder(&mut builders, src_type_idx) {
                    builder.kind = GcKind::Array;
                }
            }
        } else if wasm_dialect::RefNull::matches(ctx, op)
            || wasm_dialect::RefCast::matches(ctx, op)
            || wasm_dialect::RefTest::matches(ctx, op)
        {
            let attrs = &ctx.op(op).attributes;
            // For ref_null: use result type as fallback
            // For ref_cast/ref_test: `type` attribute may differ from operand type, so keep it
            let result_types = ctx.op_result_types(op).to_vec();
            let inferred_type = result_types.first().copied();

            // Special handling for ref_cast with placeholder type (wasm.structref + field_count)
            // First check for explicit type_idx attribute (set by wasm_gc_type_assign pass)
            if wasm_dialect::RefCast::matches(ctx, op)
                && let Some(Attribute::Type(target_ty)) = attrs.get(&ATTR_TARGET_TYPE())
                && is_structref(ctx, *target_ty)
                && let Some(Attribute::IntBits(fc)) = attrs.get(&Symbol::new("field_count"))
            {
                let field_count = intbits_to_usize(*fc)?;
                let target_ty = *target_ty;

                // Check for explicit type_idx attribute first
                if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
                    let idx = intbits_to_u32(*idx)?;
                    next_type_idx = next_type_idx.max(idx.saturating_add(1));
                    if let Some(builder) = try_get_builder(&mut builders, idx) {
                        builder.kind = GcKind::Struct;
                        if builder.field_count.is_none() {
                            builder.field_count = Some(field_count);
                        }
                        if builder.fields.len() < field_count {
                            builder.fields.resize_with(field_count, || None);
                        }
                    }
                    debug!(
                        "GC: ref_cast using explicit type_idx={} from attribute (field_count={})",
                        idx, field_count
                    );
                    continue;
                }

                // Fall back to placeholder type handling, skipping reserved indices
                let key = (target_ty, field_count);
                placeholder_struct_type_idx.entry(key).or_insert_with(|| {
                    let idx = next_available_idx(&mut next_type_idx, &reserved_indices);
                    if let Some(builder) = try_get_builder(&mut builders, idx) {
                        builder.kind = GcKind::Struct;
                        builder.field_count = Some(field_count);
                        if builder.fields.len() < field_count {
                            builder.fields.resize_with(field_count, || None);
                        }
                    }
                    debug!(
                        "GC: ref_cast allocated type_idx={} for placeholder (field_count={})",
                        idx, field_count
                    );
                    idx
                });
                // Don't fall through to regular handling
                continue;
            }

            // Try specific attribute names first, then fall back to generic "type" attribute
            let type_idx = if wasm_dialect::RefNull::matches(ctx, op) {
                match attr_u32(attrs, ATTR_HEAP_TYPE()) {
                    Ok(idx) => Some(idx),
                    Err(_) => get_type_idx(
                        ctx,
                        attrs,
                        &mut type_idx_by_type,
                        &mut next_type_idx,
                        inferred_type,
                        &reserved_indices,
                    )?,
                }
            } else {
                match attr_u32(attrs, ATTR_TARGET_TYPE()) {
                    Ok(idx) => Some(idx),
                    Err(_) => {
                        // If target_type is an Attribute::Type, prefer it over inferred_type
                        let target_type_ref = attrs
                            .get(&ATTR_TARGET_TYPE())
                            .and_then(|a| {
                                if let Attribute::Type(ty) = a {
                                    Some(*ty)
                                } else {
                                    None
                                }
                            })
                            .or(inferred_type);
                        get_type_idx(
                            ctx,
                            attrs,
                            &mut type_idx_by_type,
                            &mut next_type_idx,
                            target_type_ref,
                            &reserved_indices,
                        )?
                    }
                }
            };
            let Some(type_idx) = type_idx else {
                continue;
            };
            if let Some(&result_ty) = result_types.first() {
                register_type(&mut type_idx_by_type, type_idx, result_ty);
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

    Ok((result, type_idx_by_type, placeholder_struct_type_idx))
}

/// Helper to get type_idx from attributes or inferred type.
/// Priority: type_idx attr > type attr > inferred_type (from result/operand)
///
/// Returns `Ok(Some(idx))` if a type_idx was found/allocated,
/// `Ok(None)` if no type information is available (abstract types, etc.),
/// or `Err` if an explicit attribute has an invalid value.
fn get_type_idx(
    ctx: &IrContext,
    attrs: &std::collections::BTreeMap<Symbol, Attribute>,
    type_idx_by_type: &mut HashMap<TypeRef, u32>,
    next_type_idx: &mut u32,
    inferred_type: Option<TypeRef>,
    reserved_indices: &std::collections::HashSet<u32>,
) -> CompilationResult<Option<u32>> {
    // Helper to get next available type_idx, skipping reserved indices
    let next_available_idx = |next_type_idx: &mut u32| -> u32 {
        while reserved_indices.contains(next_type_idx) {
            *next_type_idx += 1;
        }
        let idx = *next_type_idx;
        *next_type_idx += 1;
        idx
    };

    // First try type_idx attribute
    if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
        let idx = u32::try_from(*idx).map_err(|_| {
            CompilationError::invalid_attribute(format!(
                "type_idx attribute value {} out of u32 range",
                idx
            ))
        })?;
        // Advance next_type_idx to avoid collision with explicit indices
        *next_type_idx = (*next_type_idx).max(idx.saturating_add(1));
        return Ok(Some(idx));
    }
    // Fall back to type attribute (legacy, will be removed)
    if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
        if let Some(idx) = get_builtin_struct_idx(ctx, *ty) {
            return Ok(Some(idx));
        }
        if let Some(&idx) = type_idx_by_type.get(ty) {
            debug!(
                "GC: get_type_idx reusing type_idx={} for type {:?}",
                idx, ty
            );
            return Ok(Some(idx));
        }
        // Don't allocate concrete GC type indices for abstract WASM heap types.
        // Types like wasm.anyref, wasm.i31ref are built-in abstract types, not
        // user-defined struct/array definitions.
        if is_abstract_wasm_heap_type(ctx, *ty) {
            return Ok(None);
        }
        // Allocate new type_idx, skipping reserved indices
        let idx = next_available_idx(next_type_idx);
        debug!(
            "GC: get_type_idx allocated type_idx={} for type {:?}",
            idx, ty
        );
        type_idx_by_type.insert(*ty, idx);
        return Ok(Some(idx));
    }
    // Fall back to inferred type (from result or operand types)
    if let Some(ty) = inferred_type {
        if let Some(idx) = get_builtin_struct_idx(ctx, ty) {
            return Ok(Some(idx));
        }
        if let Some(&idx) = type_idx_by_type.get(&ty) {
            debug!(
                "GC: get_type_idx reusing type_idx={} for inferred type {:?}",
                idx, ty
            );
            return Ok(Some(idx));
        }
        // Don't allocate concrete GC type indices for abstract WASM heap types.
        if is_abstract_wasm_heap_type(ctx, ty) {
            return Ok(None);
        }
        // Allocate new type_idx, skipping reserved indices
        let idx = next_available_idx(next_type_idx);
        debug!(
            "GC: get_type_idx allocated type_idx={} for inferred type {:?}",
            idx, ty
        );
        type_idx_by_type.insert(ty, idx);
        return Ok(Some(idx));
    }
    Ok(None)
}
