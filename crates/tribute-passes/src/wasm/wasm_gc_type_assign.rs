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
use trunk_ir::arena::ValueDef;
use trunk_ir::arena::context::{IrContext, OperationDataBuilder};
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::ir::Symbol;

use trunk_ir_wasm_backend::gc_types::FIRST_USER_TYPE_IDX;

const ATTR_TYPE_IDX: &str = "type_idx";
const ATTR_FIELD_IDX: &str = "field_idx";
const ATTR_FIELD_COUNT: &str = "field_count";

/// A struct type signature, identified by its field types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructSignature {
    field_types: Vec<TypeRef>,
}

/// Registry of struct types and their assigned type_idx values.
#[derive(Clone)]
struct StructTypeRegistry {
    /// Map from struct signature to type_idx
    signature_to_idx: HashMap<StructSignature, u32>,
    /// Next available type_idx
    next_type_idx: u32,
    /// Map from struct_new operation result to its type_idx (for struct_get lookup)
    value_to_type_idx: HashMap<ValueRef, u32>,
    /// Map from (type_idx, field_idx) to field type
    field_types: HashMap<(u32, u32), TypeRef>,
    /// Map from BlockRef to block arg types for resolving block argument types
    block_arg_types: HashMap<BlockRef, Vec<TypeRef>>,
}

impl StructTypeRegistry {
    fn new() -> Self {
        Self {
            signature_to_idx: HashMap::new(),
            next_type_idx: FIRST_USER_TYPE_IDX,
            value_to_type_idx: HashMap::new(),
            field_types: HashMap::new(),
            block_arg_types: HashMap::new(),
        }
    }

    /// Register a block for later block argument type lookup.
    fn register_block(&mut self, ctx: &IrContext, block: BlockRef) {
        let args = ctx.block_args(block);
        let arg_types: Vec<TypeRef> = args.iter().map(|&v| ctx.value_ty(v)).collect();
        self.block_arg_types.insert(block, arg_types);
    }

    /// Get the type of a block argument.
    fn get_block_arg_type(&self, block: BlockRef, index: usize) -> Option<TypeRef> {
        self.block_arg_types
            .get(&block)
            .and_then(|types| types.get(index).copied())
    }

    /// Register a struct type and return its type_idx.
    fn register_struct(&mut self, field_types: Vec<TypeRef>) -> u32 {
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
    fn get_field_type(&self, type_idx: u32, field_idx: u32) -> Option<TypeRef> {
        self.field_types.get(&(type_idx, field_idx)).copied()
    }
}

/// Collect struct types from struct_new operations.
fn collect_struct_types(
    ctx: &IrContext,
    module: ArenaModule,
    structref_ty: TypeRef,
) -> StructTypeRegistry {
    let mut registry = StructTypeRegistry::new();

    fn visit_region(
        ctx: &IrContext,
        region: RegionRef,
        registry: &mut StructTypeRegistry,
        structref_ty: TypeRef,
    ) {
        for &block in ctx.region(region).blocks.iter() {
            visit_block(ctx, block, registry, structref_ty);
        }
    }

    fn visit_block(
        ctx: &IrContext,
        block: BlockRef,
        registry: &mut StructTypeRegistry,
        structref_ty: TypeRef,
    ) {
        // Register block for block argument type lookup
        registry.register_block(ctx, block);

        for &op in ctx.block(block).ops.iter() {
            // Check if this is a struct_new operation with placeholder result type
            if arena_wasm::StructNew::matches(ctx, op) {
                // Only process struct_new with placeholder result type (wasm.structref)
                let result_types = ctx.op_result_types(op);
                let is_placeholder = result_types
                    .first()
                    .map(|&ty| ty == structref_ty)
                    .unwrap_or(false);

                if is_placeholder {
                    let operands = ctx.op_operands(op);
                    let field_types: Vec<TypeRef> = operands
                        .iter()
                        .map(|&v| get_value_type(ctx, v, registry, structref_ty))
                        .collect();

                    let type_idx = registry.register_struct(field_types);

                    // Record the mapping from result value to type_idx
                    let result = ctx.op_result(op, 0);
                    registry.value_to_type_idx.insert(result, type_idx);
                }
            }

            // Recursively visit nested regions
            for &nested_region in ctx.op(op).regions.iter() {
                visit_region(ctx, nested_region, registry, structref_ty);
            }
        }
    }

    if let Some(body) = module.body(ctx) {
        for &block in ctx.region(body).blocks.iter() {
            visit_block(ctx, block, &mut registry, structref_ty);
        }
    }

    registry
}

/// Get the type of a value.
fn get_value_type(
    ctx: &IrContext,
    value: ValueRef,
    registry: &StructTypeRegistry,
    fallback_ty: TypeRef,
) -> TypeRef {
    match ctx.value_def(value) {
        ValueDef::OpResult(op, idx) => {
            let result_types = ctx.op_result_types(op);
            result_types
                .get(idx as usize)
                .copied()
                .unwrap_or(fallback_ty)
        }
        ValueDef::BlockArg(block, idx) => registry
            .get_block_arg_type(block, idx as usize)
            .unwrap_or(fallback_ty),
    }
}

/// Trace a value back to find its type_idx if it came from a struct_new.
fn trace_value_to_type_idx(
    ctx: &IrContext,
    value: ValueRef,
    registry: &StructTypeRegistry,
) -> Option<u32> {
    // Direct lookup
    if let Some(&idx) = registry.value_to_type_idx.get(&value) {
        return Some(idx);
    }

    // Trace through ref_cast operations
    if let ValueDef::OpResult(op, _) = ctx.value_def(value) {
        if arena_wasm::RefCast::matches(ctx, op) {
            let operands = ctx.op_operands(op);
            if let Some(&operand) = operands.first() {
                return trace_value_to_type_idx(ctx, operand, registry);
            }
        }
    }
    None
}

/// Pattern to update struct_new operations with correct type_idx.
struct UpdateStructNewPattern {
    registry: StructTypeRegistry,
    structref_ty: TypeRef,
}

impl ArenaRewritePattern for UpdateStructNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_wasm::StructNew::matches(ctx, op) {
            return false;
        }

        // Only process struct_new with placeholder result type (wasm.structref).
        // struct_new from adt -> wasm conversion already has correct type_idx,
        // and their result type is the concrete adt type, not wasm.structref.
        let result_types = ctx.op_result_types(op);
        let is_placeholder = result_types
            .first()
            .map(|&ty| ty == self.structref_ty)
            .unwrap_or(false);

        if !is_placeholder {
            // Not a placeholder type - leave it alone
            return false;
        }

        // Get the type_idx for this struct
        let result = ctx.op_result(op, 0);
        let type_idx = self.registry.value_to_type_idx.get(&result).copied();

        let Some(type_idx) = type_idx else {
            return false;
        };

        // Check if already has correct type_idx
        if let Some(ArenaAttribute::IntBits(existing_idx)) =
            ctx.op(op).attributes.get(&Symbol::new(ATTR_TYPE_IDX))
        {
            if *existing_idx as u32 == type_idx {
                return false;
            }
        }

        // Update the operation with the correct type_idx
        let location = ctx.op(op).location;
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let results: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

        let mut builder =
            OperationDataBuilder::new(location, Symbol::new("wasm"), Symbol::new("struct_new"))
                .operands(operands)
                .results(results);

        // Copy existing attributes, updating type_idx
        for (key, val) in &ctx.op(op).attributes {
            if *key == Symbol::new(ATTR_TYPE_IDX) {
                continue;
            }
            builder = builder.attr(*key, val.clone());
        }
        builder = builder.attr(
            Symbol::new(ATTR_TYPE_IDX),
            ArenaAttribute::IntBits(type_idx as u64),
        );

        let new_data = builder.build(ctx);
        let new_op = ctx.create_op(new_data);

        debug!(
            "wasm_gc_type_assign: updated struct_new with type_idx={}",
            type_idx
        );

        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "UpdateStructNewPattern"
    }
}

/// Pattern to update struct_get operations with correct type_idx and result type.
struct UpdateStructGetPattern {
    registry: StructTypeRegistry,
}

impl ArenaRewritePattern for UpdateStructGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_wasm::StructGet::matches(ctx, op) {
            return false;
        }

        // Get the struct operand and trace to find its type_idx
        let operands = ctx.op_operands(op).to_vec();
        let Some(&struct_val) = operands.first() else {
            return false;
        };

        let Some(type_idx) = trace_value_to_type_idx(ctx, struct_val, &self.registry) else {
            // Can't determine type_idx, leave unchanged
            return false;
        };

        // Get field_idx from attributes
        let field_idx = match ctx.op(op).attributes.get(&Symbol::new(ATTR_FIELD_IDX)) {
            Some(ArenaAttribute::IntBits(idx)) => *idx as u32,
            _ => return false,
        };

        // Get the correct field type
        let Some(field_type) = self.registry.get_field_type(type_idx, field_idx) else {
            return false;
        };

        // Check if update is needed
        let current_type_idx = ctx
            .op(op)
            .attributes
            .get(&Symbol::new(ATTR_TYPE_IDX))
            .and_then(|a| match a {
                ArenaAttribute::IntBits(idx) => Some(*idx as u32),
                _ => None,
            });
        let current_result_type = ctx.op_result_types(op).first().copied();

        if current_type_idx == Some(type_idx) && current_result_type == Some(field_type) {
            return false;
        }

        // Update the operation
        let location = ctx.op(op).location;
        let mut builder =
            OperationDataBuilder::new(location, Symbol::new("wasm"), Symbol::new("struct_get"))
                .operands(operands)
                .result(field_type);

        // Copy existing attributes, updating type_idx
        for (key, val) in &ctx.op(op).attributes {
            if *key == Symbol::new(ATTR_TYPE_IDX) {
                continue;
            }
            builder = builder.attr(*key, val.clone());
        }
        builder = builder.attr(
            Symbol::new(ATTR_TYPE_IDX),
            ArenaAttribute::IntBits(type_idx as u64),
        );

        let new_data = builder.build(ctx);
        let new_op = ctx.create_op(new_data);

        debug!(
            "wasm_gc_type_assign: updated struct_get type_idx={} field_idx={}",
            type_idx, field_idx
        );

        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "UpdateStructGetPattern"
    }
}

/// Pattern to update ref_cast operations with correct type_idx.
struct UpdateRefCastPattern {
    registry: StructTypeRegistry,
}

impl ArenaRewritePattern for UpdateRefCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_wasm::RefCast::matches(ctx, op) {
            return false;
        }

        // Get the source operand and try to trace its type_idx
        let operands = ctx.op_operands(op).to_vec();
        let Some(&source_val) = operands.first() else {
            return false;
        };

        // If the source has a known type_idx, propagate it to the result
        if let Some(type_idx) = trace_value_to_type_idx(ctx, source_val, &self.registry) {
            // Check if field_count attribute exists and update type_idx
            if ctx
                .op(op)
                .attributes
                .get(&Symbol::new(ATTR_FIELD_COUNT))
                .is_some()
            {
                let current_type_idx = ctx
                    .op(op)
                    .attributes
                    .get(&Symbol::new(ATTR_TYPE_IDX))
                    .and_then(|a| match a {
                        ArenaAttribute::IntBits(idx) => Some(*idx as u32),
                        _ => None,
                    });

                if current_type_idx != Some(type_idx) {
                    let location = ctx.op(op).location;
                    let results: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

                    let mut builder = OperationDataBuilder::new(
                        location,
                        Symbol::new("wasm"),
                        Symbol::new("ref_cast"),
                    )
                    .operands(operands)
                    .results(results);

                    // Copy existing attributes, updating type_idx
                    for (key, val) in &ctx.op(op).attributes {
                        if *key == Symbol::new(ATTR_TYPE_IDX) {
                            continue;
                        }
                        builder = builder.attr(*key, val.clone());
                    }
                    builder = builder.attr(
                        Symbol::new(ATTR_TYPE_IDX),
                        ArenaAttribute::IntBits(type_idx as u64),
                    );

                    let new_data = builder.build(ctx);
                    let new_op = ctx.create_op(new_data);

                    debug!(
                        "wasm_gc_type_assign: updated ref_cast with type_idx={}",
                        type_idx
                    );

                    rewriter.replace_op(new_op);
                    return true;
                }
            }
        }

        false
    }

    fn name(&self) -> &'static str {
        "UpdateRefCastPattern"
    }
}

/// Assign unique type_idx to all GC struct operations.
///
/// This pass runs before emit and ensures:
/// 1. Each distinct struct type gets a unique type_idx
/// 2. struct_new operations have type_idx attribute set
/// 3. struct_get operations have correct type_idx and result types
pub fn assign_gc_type_indices(ctx: &mut IrContext, module: ArenaModule) {
    let structref_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("structref")).build());

    // Phase 1: Collect all struct types and assign type_idx
    let registry = collect_struct_types(ctx, module, structref_ty);

    debug!(
        "wasm_gc_type_assign: collected {} distinct struct types",
        registry.signature_to_idx.len()
    );

    // Phase 2: Update struct_new operations
    let applicator =
        PatternApplicator::new(ArenaTypeConverter::new()).add_pattern(UpdateStructNewPattern {
            registry: registry.clone(),
            structref_ty,
        });
    applicator.apply_partial(ctx, module);

    // Phase 3: Re-collect with updated struct_new to get value->type_idx mapping
    let registry = collect_struct_types(ctx, module, structref_ty);

    // Phase 4: Update struct_get operations
    let applicator =
        PatternApplicator::new(ArenaTypeConverter::new()).add_pattern(UpdateStructGetPattern {
            registry: registry.clone(),
        });
    applicator.apply_partial(ctx, module);

    // Phase 5: Update ref_cast operations
    let applicator = PatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(UpdateRefCastPattern { registry });
    applicator.apply_partial(ctx, module);
}
