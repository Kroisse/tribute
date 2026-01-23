//! Assign unique type_idx to GC struct types before emit.
//!
//! This pass runs before emit and ensures that:
//! 1. Each distinct struct type (by field types) gets a unique type_idx
//! 2. All struct_new, struct_get, and ref_cast operations have correct type_idx
//! 3. struct_get result types match the actual field types
//!
//! This removes the need for type inference at emit time.

use std::collections::HashMap;

use tracing::debug;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Type, Value,
    ValueDef,
};

use super::type_converter::wasm_type_converter;
use trunk_ir_wasm_backend::gc_types::FIRST_USER_TYPE_IDX;

trunk_ir::symbols! {
    ATTR_TYPE_IDX => "type_idx",
    ATTR_FIELD_IDX => "field_idx",
    ATTR_FIELD_COUNT => "field_count",
}

/// A struct type signature, identified by its field types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructSignature<'db> {
    field_types: Vec<Type<'db>>,
}

/// Registry of struct types and their assigned type_idx values.
#[derive(Clone)]
struct StructTypeRegistry<'db> {
    /// Map from struct signature to type_idx
    signature_to_idx: HashMap<StructSignature<'db>, u32>,
    /// Next available type_idx
    next_type_idx: u32,
    /// Map from struct_new operation result to its type_idx (for struct_get lookup)
    value_to_type_idx: HashMap<Value<'db>, u32>,
    /// Map from (type_idx, field_idx) to field type
    field_types: HashMap<(u32, u32), Type<'db>>,
    /// Map from BlockId to Block for resolving block argument types
    block_map: HashMap<BlockId, Block<'db>>,
}

impl<'db> StructTypeRegistry<'db> {
    fn new() -> Self {
        Self {
            signature_to_idx: HashMap::new(),
            next_type_idx: FIRST_USER_TYPE_IDX,
            value_to_type_idx: HashMap::new(),
            field_types: HashMap::new(),
            block_map: HashMap::new(),
        }
    }

    /// Register a block for later block argument type lookup.
    fn register_block(&mut self, db: &'db dyn salsa::Database, block: Block<'db>) {
        self.block_map.insert(block.id(db), block);
    }

    /// Get the type of a block argument.
    fn get_block_arg_type(
        &self,
        db: &'db dyn salsa::Database,
        block_id: BlockId,
        index: usize,
    ) -> Option<Type<'db>> {
        self.block_map
            .get(&block_id)
            .and_then(|block| block.args(db).get(index).map(|arg| arg.ty(db)))
    }

    /// Register a struct type and return its type_idx.
    fn register_struct(&mut self, field_types: Vec<Type<'db>>) -> u32 {
        let signature = StructSignature {
            field_types: field_types.clone(),
        };
        if let Some(&idx) = self.signature_to_idx.get(&signature) {
            return idx;
        }
        let idx = self.next_type_idx;
        self.next_type_idx += 1;
        self.signature_to_idx.insert(signature, idx);

        let field_count = field_types.len();
        // Record field types
        for (field_idx, field_ty) in field_types.into_iter().enumerate() {
            self.field_types.insert((idx, field_idx as u32), field_ty);
        }

        debug!(
            "wasm_gc_type_assign: registered struct type_idx={} with {} fields",
            idx, field_count
        );
        idx
    }

    /// Get the type of a field for a given struct type.
    fn get_field_type(&self, type_idx: u32, field_idx: u32) -> Option<Type<'db>> {
        self.field_types.get(&(type_idx, field_idx)).copied()
    }
}

/// Collect struct types from struct_new operations.
fn collect_struct_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> StructTypeRegistry<'db> {
    let mut registry = StructTypeRegistry::new();

    fn visit_region<'db>(
        db: &'db dyn salsa::Database,
        region: Region<'db>,
        registry: &mut StructTypeRegistry<'db>,
    ) {
        for block in region.blocks(db).iter() {
            visit_block(db, *block, registry);
        }
    }

    fn visit_block<'db>(
        db: &'db dyn salsa::Database,
        block: Block<'db>,
        registry: &mut StructTypeRegistry<'db>,
    ) {
        // Register block for block argument type lookup
        registry.register_block(db, block);

        for op in block.operations(db).iter() {
            // Check if this is a struct_new operation with placeholder result type
            if wasm::StructNew::matches(db, *op) {
                // Only process struct_new with placeholder result type (wasm.structref)
                let result_type = op.results(db).first().copied();
                let is_placeholder = result_type
                    .map(|ty| wasm::Structref::from_type(db, ty).is_some())
                    .unwrap_or(false);

                if is_placeholder {
                    let operands = op.operands(db);
                    let field_types: Vec<Type<'db>> = operands
                        .iter()
                        .map(|v| get_value_type(db, *v, registry))
                        .collect();

                    let type_idx = registry.register_struct(field_types);

                    // Record the mapping from result value to type_idx
                    let result = op.result(db, 0);
                    registry.value_to_type_idx.insert(result, type_idx);
                }
            }

            // Recursively visit nested regions
            for nested_region in op.regions(db).iter() {
                visit_region(db, *nested_region, registry);
            }
        }
    }

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        visit_block(db, *block, &mut registry);
    }

    registry
}

/// Get the type of a value.
fn get_value_type<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    registry: &StructTypeRegistry<'db>,
) -> Type<'db> {
    match value.def(db) {
        ValueDef::OpResult(op) => op
            .results(db)
            .get(value.index(db))
            .copied()
            .unwrap_or_else(|| wasm::Anyref::new(db).as_type()),
        ValueDef::BlockArg(block_id) => {
            // Look up block argument type from registry
            registry
                .get_block_arg_type(db, block_id, value.index(db))
                .unwrap_or_else(|| wasm::Anyref::new(db).as_type())
        }
    }
}

/// Trace a value back to find its type_idx if it came from a struct_new.
fn trace_value_to_type_idx<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    registry: &StructTypeRegistry<'db>,
) -> Option<u32> {
    // Direct lookup
    if let Some(&idx) = registry.value_to_type_idx.get(&value) {
        return Some(idx);
    }

    // Trace through ref_cast operations
    if let ValueDef::OpResult(op) = value.def(db)
        && wasm::RefCast::matches(db, op)
        && let Some(&operand) = op.operands(db).first()
    {
        return trace_value_to_type_idx(db, operand, registry);
    }
    None
}

/// Pattern to update struct_new operations with correct type_idx.
struct UpdateStructNewPattern<'db> {
    registry: StructTypeRegistry<'db>,
}

impl<'db> RewritePattern<'db> for UpdateStructNewPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if !wasm::StructNew::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Only process struct_new with placeholder result type (wasm.structref).
        // struct_new from adt → wasm conversion already has correct type_idx,
        // and their result type is the concrete adt type, not wasm.structref.
        let result_type = op.results(db).first().copied();
        let is_placeholder = result_type
            .map(|ty| wasm::Structref::from_type(db, ty).is_some())
            .unwrap_or(false);

        if !is_placeholder {
            // Not a placeholder type - leave it alone
            return RewriteResult::Unchanged;
        }

        // Get the type_idx for this struct
        let result = op.result(db, 0);
        let type_idx = self.registry.value_to_type_idx.get(&result).copied();

        let Some(type_idx) = type_idx else {
            return RewriteResult::Unchanged;
        };

        // Check if already has correct type_idx
        if let Some(Attribute::IntBits(existing_idx)) = op.attributes(db).get(&ATTR_TYPE_IDX())
            && *existing_idx as u32 == type_idx
        {
            return RewriteResult::Unchanged;
        }

        // Update the operation with the correct type_idx
        let new_op = op
            .modify(db)
            .attr(ATTR_TYPE_IDX(), Attribute::IntBits(type_idx as u64))
            .build();

        debug!(
            "wasm_gc_type_assign: updated struct_new with type_idx={}",
            type_idx
        );

        RewriteResult::Replace(new_op)
    }
}

/// Pattern to update struct_get operations with correct type_idx and result type.
struct UpdateStructGetPattern<'db> {
    registry: StructTypeRegistry<'db>,
}

impl<'db> RewritePattern<'db> for UpdateStructGetPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if !wasm::StructGet::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Get the struct operand and trace to find its type_idx
        let Some(&struct_val) = op.operands(db).first() else {
            return RewriteResult::Unchanged;
        };

        let Some(type_idx) = trace_value_to_type_idx(db, struct_val, &self.registry) else {
            // Can't determine type_idx, leave unchanged
            return RewriteResult::Unchanged;
        };

        // Get field_idx from attributes
        let field_idx = match op.attributes(db).get(&ATTR_FIELD_IDX()) {
            Some(Attribute::IntBits(idx)) => *idx as u32,
            _ => return RewriteResult::Unchanged,
        };

        // Get the correct field type
        let Some(field_type) = self.registry.get_field_type(type_idx, field_idx) else {
            return RewriteResult::Unchanged;
        };

        // Check if update is needed
        let current_type_idx = op
            .attributes(db)
            .get(&ATTR_TYPE_IDX())
            .and_then(|a| match a {
                Attribute::IntBits(idx) => Some(*idx as u32),
                _ => None,
            });
        let current_result_type = op.results(db).first().copied();

        if current_type_idx == Some(type_idx) && current_result_type == Some(field_type) {
            return RewriteResult::Unchanged;
        }

        // Update the operation
        let new_op = op
            .modify(db)
            .attr(ATTR_TYPE_IDX(), Attribute::IntBits(type_idx as u64))
            .results(IdVec::from(vec![field_type]))
            .build();

        debug!(
            "wasm_gc_type_assign: updated struct_get type_idx={} field_idx={} result_type={}.{}",
            type_idx,
            field_idx,
            field_type.dialect(db),
            field_type.name(db)
        );

        RewriteResult::Replace(new_op)
    }
}

/// Pattern to update ref_cast operations with correct type_idx.
struct UpdateRefCastPattern<'db> {
    registry: StructTypeRegistry<'db>,
}

impl<'db> RewritePattern<'db> for UpdateRefCastPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if !wasm::RefCast::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Get the source operand and try to trace its type_idx
        let Some(&source_val) = op.operands(db).first() else {
            return RewriteResult::Unchanged;
        };

        // If the source has a known type_idx, propagate it to the result
        if let Some(type_idx) = trace_value_to_type_idx(db, source_val, &self.registry) {
            // Check if field_count attribute exists and update type_idx
            if op.attributes(db).get(&ATTR_FIELD_COUNT()).is_some() {
                let current_type_idx =
                    op.attributes(db)
                        .get(&ATTR_TYPE_IDX())
                        .and_then(|a| match a {
                            Attribute::IntBits(idx) => Some(*idx as u32),
                            _ => None,
                        });

                if current_type_idx != Some(type_idx) {
                    let new_op = op
                        .modify(db)
                        .attr(ATTR_TYPE_IDX(), Attribute::IntBits(type_idx as u64))
                        .build();

                    debug!(
                        "wasm_gc_type_assign: updated ref_cast with type_idx={}",
                        type_idx
                    );

                    return RewriteResult::Replace(new_op);
                }
            }
        }

        RewriteResult::Unchanged
    }
}

/// Assign unique type_idx to all GC struct operations.
///
/// This pass runs before emit and ensures:
/// 1. Each distinct struct type gets a unique type_idx
/// 2. struct_new operations have type_idx attribute set
/// 3. struct_get operations have correct type_idx and result types
#[salsa::tracked]
pub fn assign_gc_type_indices<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Phase 1: Collect all struct types and assign type_idx
    let registry = collect_struct_types(db, module);

    debug!(
        "wasm_gc_type_assign: collected {} distinct struct types",
        registry.signature_to_idx.len()
    );

    // No specific conversion target - these are optimization passes
    let target = ConversionTarget::new();

    // Phase 2: Update struct_new operations
    let applicator =
        PatternApplicator::new(wasm_type_converter()).add_pattern(UpdateStructNewPattern {
            registry: registry.clone(),
        });
    let module = applicator.apply_partial(db, module, target.clone()).module;

    // Phase 3: Re-collect with updated struct_new to get value->type_idx mapping
    let registry = collect_struct_types(db, module);

    // Phase 4: Update struct_get operations
    let applicator =
        PatternApplicator::new(wasm_type_converter()).add_pattern(UpdateStructGetPattern {
            registry: registry.clone(),
        });
    let module = applicator.apply_partial(db, module, target.clone()).module;

    // Phase 5: Update ref_cast operations
    let applicator = PatternApplicator::new(wasm_type_converter())
        .add_pattern(UpdateRefCastPattern { registry });
    applicator.apply_partial(db, module, target).module
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_signature_equality() {
        let db = salsa::DatabaseImpl::default();
        let i32_ty = core::I32::new(&db).as_type();
        let anyref_ty = wasm::Anyref::new(&db).as_type();

        let sig1 = StructSignature {
            field_types: vec![i32_ty, anyref_ty],
        };
        let sig2 = StructSignature {
            field_types: vec![i32_ty, anyref_ty],
        };
        let sig3 = StructSignature {
            field_types: vec![anyref_ty, i32_ty],
        };

        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_registry_assigns_unique_indices() {
        let db = salsa::DatabaseImpl::default();
        let i32_ty = core::I32::new(&db).as_type();
        let anyref_ty = wasm::Anyref::new(&db).as_type();
        let funcref_ty = wasm::Funcref::new(&db).as_type();

        let mut registry = StructTypeRegistry::new();

        // Register first struct type
        let idx1 = registry.register_struct(vec![i32_ty, anyref_ty]);
        assert_eq!(idx1, FIRST_USER_TYPE_IDX);

        // Register same signature again - should return same idx
        let idx2 = registry.register_struct(vec![i32_ty, anyref_ty]);
        assert_eq!(idx2, FIRST_USER_TYPE_IDX);

        // Register different signature - should get new idx
        let idx3 = registry.register_struct(vec![funcref_ty, anyref_ty, i32_ty]);
        assert_eq!(idx3, FIRST_USER_TYPE_IDX + 1);

        // Verify field types are recorded
        assert_eq!(registry.get_field_type(idx1, 0), Some(i32_ty));
        assert_eq!(registry.get_field_type(idx1, 1), Some(anyref_ty));
        assert_eq!(registry.get_field_type(idx3, 0), Some(funcref_ty));
        assert_eq!(registry.get_field_type(idx3, 2), Some(i32_ty));
    }
}
