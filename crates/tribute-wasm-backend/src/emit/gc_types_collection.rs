//! GC type collection from wasm dialect operations.
//!
//! This module traverses wasm operations to collect WebAssembly GC type
//! definitions (structs and arrays) and build the type index mappings.

use std::collections::HashMap;

use tracing::debug;

use tribute_ir::dialect::{adt, closure, tribute};
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{Attribute, BlockId, DialectOp, DialectType, Operation, Region, Symbol, Type};
use wasm_encoder::{FieldType, StorageType, ValType};

use crate::gc_types::{
    self, CLOSURE_STRUCT_IDX, FIRST_USER_TYPE_IDX, GcTypeDef, GcTypeRegistry, STEP_IDX,
};
use crate::{CompilationError, CompilationResult};

use super::helpers::{is_closure_struct_type, value_type};

// Symbol definitions - reuse from parent module
trunk_ir::symbols! {
    ATTR_TYPE_IDX => "type_idx",
    ATTR_TYPE => "type",
    ATTR_FIELD_IDX => "field_idx",
    ATTR_FIELD => "field",
    ATTR_TARGET_TYPE => "target_type",
    ATTR_HEAP_TYPE => "heap_type",
}

/// Result type for GC type collection.
/// Returns: (type_defs, type_idx_by_type, placeholder_struct_type_idx)
pub(crate) type GcTypesResult<'db> = (
    Vec<GcTypeDef>,
    HashMap<Type<'db>, u32>,
    HashMap<(Type<'db>, usize), u32>,
);

/// GC type kind enum
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GcKind {
    Struct,
    Array,
    Unknown,
}

/// Builder for GC types (struct or array)
struct GcTypeBuilder<'db> {
    kind: GcKind,
    fields: Vec<Option<Type<'db>>>,
    array_elem: Option<Type<'db>>,
    field_count: Option<usize>,
}

impl<'db> GcTypeBuilder<'db> {
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

/// Returns true if this is a built-in type (0-4) that shouldn't use a builder
fn is_builtin_type(idx: u32) -> bool {
    idx < FIRST_USER_TYPE_IDX
}

/// Get or create a builder for a user-defined type.
/// Returns None for built-in types (0-4) which are predefined.
fn try_get_builder<'db, 'a>(
    builders: &'a mut Vec<GcTypeBuilder<'db>>,
    idx: u32,
) -> Option<&'a mut GcTypeBuilder<'db>> {
    // Skip built-in types (0-4) as they are predefined
    if is_builtin_type(idx) {
        return None;
    }
    // Subtract FIRST_USER_TYPE_IDX because indices 0-4 are reserved for built-in types
    // User type indices start at FIRST_USER_TYPE_IDX
    let adjusted_idx = (idx - FIRST_USER_TYPE_IDX) as usize;
    if builders.len() <= adjusted_idx {
        builders.resize_with(adjusted_idx + 1, GcTypeBuilder::new);
    }
    Some(&mut builders[adjusted_idx])
}

/// Register a type in the type_idx_by_type map
fn register_type<'db>(type_idx_by_type: &mut HashMap<Type<'db>, u32>, idx: u32, ty: Type<'db>) {
    type_idx_by_type.entry(ty).or_insert(idx);
}

/// Record a struct field type
fn record_struct_field<'db>(
    type_idx: u32,
    builder: &mut GcTypeBuilder<'db>,
    field_idx: u32,
    ty: Type<'db>,
) -> CompilationResult<()> {
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
        if existing != ty {
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
fn record_array_elem<'db>(
    type_idx: u32,
    builder: &mut GcTypeBuilder<'db>,
    ty: Type<'db>,
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

/// Get attribute value as u32
fn attr_u32<'db>(
    attrs: &std::collections::BTreeMap<Symbol, Attribute<'db>>,
    key: Symbol,
) -> CompilationResult<u32> {
    match attrs.get(&key) {
        Some(Attribute::IntBits(bits)) => Ok(*bits as u32),
        _ => Err(CompilationError::missing_attribute("u32")),
    }
}

/// Get field index from attributes
fn attr_field_idx<'db>(
    attrs: &std::collections::BTreeMap<Symbol, Attribute<'db>>,
) -> CompilationResult<u32> {
    attr_u32(attrs, ATTR_FIELD_IDX()).or_else(|_| attr_u32(attrs, ATTR_FIELD()))
}

// ============================================================================
// Main collection function
// ============================================================================

/// Collect GC types from wasm dialect operations in a module.
///
/// Traverses all operations to identify struct and array types, recording their
/// field/element types. Returns type definitions, type index mappings, and
/// placeholder struct mappings.
pub(crate) fn collect_gc_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    block_arg_types: &HashMap<(BlockId, usize), Type<'db>>,
) -> CompilationResult<GcTypesResult<'db>> {
    let wasm_dialect = Symbol::new("wasm");
    let mut builders: Vec<GcTypeBuilder<'db>> = Vec::new();
    let mut type_idx_by_type: HashMap<Type<'db>, u32> = HashMap::new();
    // For placeholder types like wasm.structref, use (type, field_count) as key
    // to handle multiple structs with same placeholder type but different field counts
    let mut placeholder_struct_type_idx: HashMap<(Type<'db>, usize), u32> = HashMap::new();

    // Register builtin types in type_idx_by_type:
    // 0: BoxedF64, 1: BytesArray, 2: BytesStruct, 3: Step, 4: ClosureStruct
    // Step marker type needs to be registered so wasm.if can use it
    type_idx_by_type.insert(crate::gc_types::step_marker_type(db), STEP_IDX);

    // Start at FIRST_USER_TYPE_IDX since indices 0-4 are reserved for built-in types
    let mut next_type_idx: u32 = FIRST_USER_TYPE_IDX;

    // Helper to get type_idx from attributes or inferred type.
    // Priority: type_idx attr > type attr > inferred_type (from result/operand)
    let get_type_idx = |attrs: &std::collections::BTreeMap<Symbol, Attribute<'db>>,
                        type_idx_by_type: &mut HashMap<Type<'db>, u32>,
                        next_type_idx: &mut u32,
                        inferred_type: Option<Type<'db>>|
     -> Option<u32> {
        // First try type_idx attribute
        if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
            let idx = *idx as u32;
            // Advance next_type_idx to avoid collision with explicit indices
            *next_type_idx = (*next_type_idx).max(idx.saturating_add(1));
            return Some(idx);
        }
        // Fall back to type attribute (legacy, will be removed)
        if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
            // Special case: closure types use builtin CLOSURE_STRUCT_IDX
            if closure::Closure::from_type(db, *ty).is_some() || is_closure_struct_type(db, *ty) {
                return Some(CLOSURE_STRUCT_IDX);
            }
            if let Some(&idx) = type_idx_by_type.get(ty) {
                return Some(idx);
            }
            // Allocate new type_idx
            let idx = *next_type_idx;
            *next_type_idx += 1;
            type_idx_by_type.insert(*ty, idx);
            return Some(idx);
        }
        // Fall back to inferred type (from result or operand types)
        if let Some(ty) = inferred_type {
            // Special case: closure types use builtin CLOSURE_STRUCT_IDX
            if closure::Closure::from_type(db, ty).is_some() || is_closure_struct_type(db, ty) {
                return Some(CLOSURE_STRUCT_IDX);
            }
            if let Some(&idx) = type_idx_by_type.get(&ty) {
                return Some(idx);
            }
            // Allocate new type_idx
            let idx = *next_type_idx;
            *next_type_idx += 1;
            type_idx_by_type.insert(ty, idx);
            return Some(idx);
        }
        None
    };

    let mut visit_op = |op: &Operation<'db>| -> CompilationResult<()> {
        if op.dialect(db) != wasm_dialect {
            return Ok(());
        }
        if wasm::StructNew::matches(db, *op) {
            let attrs = op.attributes(db);
            let field_count = op.operands(db).len();
            let result_type = op.results(db).first().copied();

            // Check if this uses a placeholder type (wasm.structref) that allows
            // multiple structs with same type but different field counts.
            // Check both the `type` attribute and the result type for placeholder.
            let placeholder_type_from_attr = attrs.get(&ATTR_TYPE()).and_then(|attr| {
                if let Attribute::Type(ty) = attr {
                    if wasm::Structref::from_type(db, *ty).is_some() {
                        Some(*ty)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            let placeholder_type_from_result = result_type.and_then(|ty| {
                if wasm::Structref::from_type(db, ty).is_some() {
                    Some(ty)
                } else {
                    None
                }
            });

            let placeholder_type = placeholder_type_from_attr.or(placeholder_type_from_result);

            let type_idx = if let Some(ty) = placeholder_type {
                // For placeholder types, use (type, field_count) as key to allow
                // different field counts with same placeholder type
                let key = (ty, field_count);
                if let Some(&idx) = placeholder_struct_type_idx.get(&key) {
                    debug!(
                        "GC: struct_new reusing existing type_idx={} for placeholder (field_count={})",
                        idx, field_count
                    );
                    idx
                } else {
                    let idx = next_type_idx;
                    next_type_idx += 1;
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
                let inferred_type = op.results(db).first().copied();
                let Some(idx) = get_type_idx(
                    attrs,
                    &mut type_idx_by_type,
                    &mut next_type_idx,
                    inferred_type,
                ) else {
                    return Ok(());
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

                if let Some(result_ty) = op.results(db).first().copied() {
                    register_type(&mut type_idx_by_type, type_idx, result_ty);
                }
                for (field_idx, value) in op.operands(db).iter().enumerate() {
                    if let Some(ty) = value_type(db, *value, block_arg_types) {
                        debug!(
                            "GC: struct_new type_idx={} recording field {} with type {}.{}",
                            type_idx,
                            field_idx,
                            ty.dialect(db),
                            ty.name(db)
                        );
                        record_struct_field(type_idx, builder, field_idx as u32, ty)?;
                    } else {
                        debug!(
                            "GC: struct_new type_idx={} field {} has no type (value_type returned None)",
                            type_idx, field_idx
                        );
                    }
                }
            }
        } else if wasm::StructGet::matches(db, *op) {
            let attrs = op.attributes(db);

            // Check if this uses a placeholder type (wasm.structref) that allows
            // multiple structs with same type but different field counts
            let is_placeholder_type = attrs
                .get(&ATTR_TYPE())
                .map(|attr| {
                    if let Attribute::Type(ty) = attr {
                        wasm::Structref::from_type(db, *ty).is_some()
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            // Check if this is a _closure struct type (should use CLOSURE_STRUCT_IDX)
            let is_closure_type = attrs.get(&ATTR_TYPE()).is_some_and(|attr| {
                if let Attribute::Type(ty) = attr {
                    is_closure_struct_type(db, *ty)
                } else {
                    false
                }
            });

            // Check if this uses an adt.struct type (which should also use placeholder lookup)
            // Returns (adt_struct_type, field_count) if it's an adt.struct type
            let adt_struct_info = attrs.get(&ATTR_TYPE()).and_then(|attr| {
                if let Attribute::Type(ty) = attr {
                    if adt::is_struct_type(db, *ty) && !is_closure_struct_type(db, *ty) {
                        adt::get_struct_fields(db, *ty).map(|fields| (*ty, fields.len()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            let type_idx = if is_closure_type {
                // Closure struct types use the builtin CLOSURE_STRUCT_IDX
                CLOSURE_STRUCT_IDX
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
                let field_count = *field_count as usize;
                let key = (ty, field_count);
                if let Some(&idx) = placeholder_struct_type_idx.get(&key) {
                    idx
                } else {
                    // Allocate new type_idx for this placeholder
                    let idx = next_type_idx;
                    next_type_idx += 1;
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
                        "GC: struct_get reusing type_idx={} for adt.struct from type_idx_by_type",
                        idx
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
                        // Allocate new type_idx for this adt.struct type
                        let idx = next_type_idx;
                        next_type_idx += 1;
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
                let inferred_type = op
                    .operands(db)
                    .first()
                    .and_then(|v| value_type(db, *v, block_arg_types));
                let Some(idx) = get_type_idx(
                    attrs,
                    &mut type_idx_by_type,
                    &mut next_type_idx,
                    inferred_type,
                ) else {
                    return Ok(());
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
                    let fc = *fc as usize;
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
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                // Only record field type if result type is concrete (not a type variable).
                // Type variables map to ANYREF and would conflict with concrete types.
                if let Some(result_ty) = op.results(db).first().copied()
                    && !tribute::is_type_var(db, result_ty)
                {
                    debug!(
                        "GC: struct_get type_idx={} recording field {} with result_ty {}.{}",
                        type_idx,
                        field_idx,
                        result_ty.dialect(db),
                        result_ty.name(db)
                    );
                    record_struct_field(type_idx, builder, field_idx, result_ty)?;
                }
            }
        } else if wasm::StructSet::matches(db, *op) {
            let attrs = op.attributes(db);
            // Infer type from operand[0] (the struct ref)
            let inferred_type = op
                .operands(db)
                .first()
                .and_then(|v| value_type(db, *v, block_arg_types));
            let Some(type_idx) = get_type_idx(
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
            ) else {
                return Ok(());
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
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                if let Some(ty) = op
                    .operands(db)
                    .get(1)
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    record_struct_field(type_idx, builder, field_idx, ty)?;
                }
            }
        } else if wasm::ArrayNew::matches(db, *op) || wasm::ArrayNewDefault::matches(db, *op) {
            let attrs = op.attributes(db);
            // Infer type from result type
            let inferred_type = op.results(db).first().copied();
            let Some(type_idx) = get_type_idx(
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
            ) else {
                return Ok(());
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(result_ty) = op.results(db).first().copied() {
                    register_type(&mut type_idx_by_type, type_idx, result_ty);
                }
                if let Some(ty) = op
                    .operands(db)
                    .get(1)
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if wasm::ArrayGet::matches(db, *op)
            || wasm::ArrayGetS::matches(db, *op)
            || wasm::ArrayGetU::matches(db, *op)
        {
            let attrs = op.attributes(db);
            // Infer type from operand[0] (the array ref)
            let inferred_type = op
                .operands(db)
                .first()
                .and_then(|v| value_type(db, *v, block_arg_types));
            let Some(type_idx) = get_type_idx(
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
            ) else {
                return Ok(());
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                // Only record element type if result type is concrete (not a type variable).
                if let Some(result_ty) = op.results(db).first().copied()
                    && !tribute::is_type_var(db, result_ty)
                {
                    record_array_elem(type_idx, builder, result_ty)?;
                }
            }
        } else if wasm::ArraySet::matches(db, *op) {
            let attrs = op.attributes(db);
            // Infer type from operand[0] (the array ref)
            let inferred_type = op
                .operands(db)
                .first()
                .and_then(|v| value_type(db, *v, block_arg_types));
            let Some(type_idx) = get_type_idx(
                attrs,
                &mut type_idx_by_type,
                &mut next_type_idx,
                inferred_type,
            ) else {
                return Ok(());
            };
            if let Some(builder) = try_get_builder(&mut builders, type_idx) {
                builder.kind = GcKind::Array;
                if let Some(ty) = op
                    .operands(db)
                    .first()
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    register_type(&mut type_idx_by_type, type_idx, ty);
                }
                if let Some(ty) = op
                    .operands(db)
                    .get(2)
                    .copied()
                    .and_then(|value| value_type(db, value, block_arg_types))
                {
                    record_array_elem(type_idx, builder, ty)?;
                }
            }
        } else if wasm::ArrayCopy::matches(db, *op) {
            // array_copy has dst_type_idx: u32 and src_type_idx: u32 attributes
            let attrs = op.attributes(db);
            if let Some(&Attribute::IntBits(dst_idx)) = attrs.get(&Symbol::new("dst_type_idx")) {
                let dst_type_idx = dst_idx as u32;
                if let Some(builder) = try_get_builder(&mut builders, dst_type_idx) {
                    builder.kind = GcKind::Array;
                }
            }
            if let Some(&Attribute::IntBits(src_idx)) = attrs.get(&Symbol::new("src_type_idx")) {
                let src_type_idx = src_idx as u32;
                if let Some(builder) = try_get_builder(&mut builders, src_type_idx) {
                    builder.kind = GcKind::Array;
                }
            }
        } else if wasm::RefNull::matches(db, *op)
            || wasm::RefCast::matches(db, *op)
            || wasm::RefTest::matches(db, *op)
        {
            let attrs = op.attributes(db);
            // For ref_null: use result type as fallback
            // For ref_cast/ref_test: `type` attribute may differ from operand type, so keep it
            let inferred_type = op.results(db).first().copied();

            // Special handling for ref_cast with placeholder type (wasm.structref + field_count)
            if wasm::RefCast::matches(db, *op)
                && let Some(Attribute::Type(target_ty)) = attrs.get(&ATTR_TARGET_TYPE())
                && wasm::Structref::from_type(db, *target_ty).is_some()
                && let Some(Attribute::IntBits(fc)) = attrs.get(&Symbol::new("field_count"))
            {
                let field_count = *fc as usize;
                let key = (*target_ty, field_count);
                placeholder_struct_type_idx.entry(key).or_insert_with(|| {
                    let idx = next_type_idx;
                    next_type_idx += 1;
                    // Use try_get_builder to create/get the builder at the right index
                    // Initialize fields vec so struct_new can populate field types later
                    if let Some(builder) = try_get_builder(&mut builders, idx) {
                        builder.kind = GcKind::Struct;
                        builder.field_count = Some(field_count);
                        // Pre-allocate fields vec to be populated by struct_new
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
                return Ok(());
            }

            // Try specific attribute names first, then fall back to generic "type" attribute
            let type_idx = if wasm::RefNull::matches(db, *op) {
                attr_u32(attrs, ATTR_HEAP_TYPE()).ok().or_else(|| {
                    get_type_idx(
                        attrs,
                        &mut type_idx_by_type,
                        &mut next_type_idx,
                        inferred_type,
                    )
                })
            } else {
                attr_u32(attrs, ATTR_TARGET_TYPE()).ok().or_else(|| {
                    get_type_idx(
                        attrs,
                        &mut type_idx_by_type,
                        &mut next_type_idx,
                        inferred_type,
                    )
                })
            };
            let Some(type_idx) = type_idx else {
                return Ok(());
            };
            if let Some(result_ty) = op.results(db).first().copied() {
                register_type(&mut type_idx_by_type, type_idx, result_ty);
            }
            if let Some(builder) = try_get_builder(&mut builders, type_idx)
                && builder.kind == GcKind::Unknown
            {
                builder.kind = GcKind::Struct;
            }
        }
        Ok(())
    };

    // Recursively visit operations, including nested core.module operations.
    /// Recursively visit all operations in a region, including nested regions.
    fn visit_all_ops_recursive<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        visit_op: &mut impl FnMut(&Operation<'db>) -> CompilationResult<()>,
    ) -> CompilationResult<()> {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                visit_op(op)?;
                // Recursively visit nested regions (for wasm.if, wasm.block, etc.)
                for nested_region in op.regions(db).iter() {
                    visit_all_ops_recursive(db, nested_region, visit_op)?;
                }
            }
        }
        Ok(())
    }

    fn visit_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        visit_op: &mut impl FnMut(&Operation<'db>) -> CompilationResult<()>,
    ) -> CompilationResult<()> {
        let wasm_dialect = Symbol::new("wasm");
        let core_dialect = Symbol::new("core");
        let module_name = Symbol::new("module");
        let func_name = Symbol::new("func");

        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let dialect = op.dialect(db);
                let name = op.name(db);

                // Recurse into nested core.module operations
                if dialect == core_dialect && name == module_name {
                    for nested_region in op.regions(db).iter() {
                        visit_region(db, nested_region, visit_op)?;
                    }
                    continue;
                }

                // Visit wasm.func body (recursively including nested regions)
                if dialect == wasm_dialect && name == func_name {
                    if let Some(func_region) = op.regions(db).first() {
                        visit_all_ops_recursive(db, func_region, visit_op)?;
                    }
                } else {
                    visit_op(op)?;
                    // Also visit nested regions for control flow ops (wasm.if, wasm.block, etc.)
                    for nested_region in op.regions(db).iter() {
                        visit_all_ops_recursive(db, nested_region, visit_op)?;
                    }
                }
            }
        }
        Ok(())
    }

    visit_region(db, &module.body(db), &mut visit_op)?;

    // Create a registry view for type conversion (uses gc_types::type_to_field_type)
    let registry = GcTypeRegistry::from_type_maps(
        type_idx_by_type.clone(),
        placeholder_struct_type_idx.clone(),
    );

    // Build user-defined types from builders
    let mut user_types = Vec::new();
    for builder in builders {
        match builder.kind {
            GcKind::Array => {
                let elem = builder
                    .array_elem
                    .map(|ty| gc_types::type_to_field_type(db, ty, &registry))
                    .unwrap_or(FieldType {
                        element_type: StorageType::Val(ValType::I32),
                        mutable: false,
                    });
                user_types.push(GcTypeDef::Array(elem));
            }
            GcKind::Struct | GcKind::Unknown => {
                let fields = builder
                    .fields
                    .into_iter()
                    .map(|ty| {
                        ty.map(|ty| gc_types::type_to_field_type(db, ty, &registry))
                            .unwrap_or(FieldType {
                                element_type: StorageType::Val(ValType::I32),
                                mutable: false,
                            })
                    })
                    .collect::<Vec<_>>();
                user_types.push(GcTypeDef::Struct(fields));
            }
        }
    }

    // Combine builtin types (from GcTypeRegistry) with user-defined types
    let mut result = GcTypeRegistry::builtin_types();
    result.extend(user_types);

    Ok((result, type_idx_by_type, placeholder_struct_type_idx))
}
