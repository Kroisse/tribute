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
    self, CONTINUATION_IDX, EVIDENCE_IDX, FIRST_USER_TYPE_IDX, GcTypeDef, MARKER_IDX,
    RESUME_WRAPPER_IDX, STEP_IDX,
};
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
    let wasm_dialect = Symbol::new("wasm");
    let mut builders: Vec<GcTypeBuilder> = Vec::new();
    let mut type_idx_by_type: HashMap<TypeRef, u32> = HashMap::new();
    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;

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
            let struct_new =
                wasm_dialect::StructNew::from_op(ctx, op).expect("matched wasm.struct_new");
            let operands = ctx.op_operands(op).to_vec();
            let field_count = operands.len();
            let result_types = ctx.op_result_types(op).to_vec();
            let result_type = result_types.first().copied();
            let type_idx = struct_new.type_idx(ctx);

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
            let struct_get =
                wasm_dialect::StructGet::from_op(ctx, op).expect("matched wasm.struct_get");
            let type_idx = struct_get.type_idx(ctx);
            let field_idx = struct_get.field_idx(ctx);
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Struct;

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
            let struct_set =
                wasm_dialect::StructSet::from_op(ctx, op).expect("matched wasm.struct_set");
            let operands = ctx.op_operands(op).to_vec();
            let type_idx = struct_set.type_idx(ctx);
            let field_idx = struct_set.field_idx(ctx);
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
            let result_types = ctx.op_result_types(op).to_vec();
            let type_idx = wasm_dialect::ArrayNew::from_op(ctx, op)
                .map(|op| op.type_idx(ctx))
                .or_else(|_| {
                    wasm_dialect::ArrayNewDefault::from_op(ctx, op).map(|op| op.type_idx(ctx))
                })
                .expect("matched indexed array.new operation");
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
            let array_set =
                wasm_dialect::ArraySet::from_op(ctx, op).expect("matched wasm.array_set");
            let operands = ctx.op_operands(op).to_vec();
            let type_idx = array_set.type_idx(ctx);
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

    Ok((result, type_idx_by_type))
}
