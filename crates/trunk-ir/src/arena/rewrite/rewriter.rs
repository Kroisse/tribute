//! PatternRewriter for arena IR.
//!
//! Accumulates mutations from pattern rewrites. Unlike the Salsa version,
//! no value remapping is needed — RAUW handles that directly on the context.

use crate::arena::context::IrContext;
use crate::arena::refs::{OpRef, ValueRef};
use crate::arena::rewrite::type_converter::ArenaTypeConverter;

/// Accumulated mutations from a pattern rewrite.
pub(crate) struct Mutations {
    /// Operations to insert before the current op's position.
    pub(crate) prefix_ops: Vec<OpRef>,
    /// The replacement operation (if any).
    pub(crate) replacement: Option<OpRef>,
    /// If set, the operation is erased and its results mapped to these values.
    pub(crate) erase_values: Option<Vec<ValueRef>>,
    /// Operations to add at module level.
    pub(crate) module_ops: Vec<OpRef>,
}

/// Rewriter interface for arena IR patterns.
///
/// Patterns use this to record mutations which are applied by the
/// `PatternApplicator` after the pattern returns.
///
/// Unlike the Salsa-based `PatternRewriter`, there is no operand remapping —
/// operands are read directly from the context, and value replacements are
/// done via `IrContext::replace_all_uses`.
pub struct PatternRewriter<'a> {
    type_converter: &'a ArenaTypeConverter,
    prefix_ops: Vec<OpRef>,
    replacement: Option<OpRef>,
    erase_values: Option<Vec<ValueRef>>,
    module_ops: Vec<OpRef>,
}

impl<'a> PatternRewriter<'a> {
    /// Create a new empty rewriter with a reference to the type converter.
    pub(crate) fn new(type_converter: &'a ArenaTypeConverter) -> Self {
        Self {
            type_converter,
            prefix_ops: Vec::new(),
            replacement: None,
            erase_values: None,
            module_ops: Vec::new(),
        }
    }

    /// Get a reference to the type converter.
    pub fn type_converter(&self) -> &ArenaTypeConverter {
        self.type_converter
    }

    // === Mutations ===

    /// Insert an operation before the current operation.
    ///
    /// The op must already be created via `ctx.create_op()` but not yet
    /// attached to a block. Multiple calls accumulate operations in order.
    pub fn insert_op(&mut self, op: OpRef) {
        self.prefix_ops.push(op);
    }

    /// Replace the current operation with a new one.
    ///
    /// The applicator will RAUW old results → new results (1:1 by index),
    /// then remove the old op from its block and insert the new one.
    pub fn replace_op(&mut self, new_op: OpRef) {
        debug_assert!(
            self.replacement.is_none() && self.erase_values.is_none(),
            "replace_op called after replace_op or erase_op"
        );
        self.replacement = Some(new_op);
    }

    /// Erase the current operation, mapping its results to the given values.
    ///
    /// The replacement values must match the original result count.
    /// The applicator will RAUW each old result to the corresponding value.
    pub fn erase_op(&mut self, replacement_values: Vec<ValueRef>) {
        debug_assert!(
            self.replacement.is_none() && self.erase_values.is_none(),
            "erase_op called after replace_op or erase_op"
        );
        self.erase_values = Some(replacement_values);
    }

    /// Add an operation at module level (e.g., an outlined function).
    pub fn add_module_op(&mut self, op: OpRef) {
        self.module_ops.push(op);
    }

    // === Query ===

    /// Check if any mutation was recorded.
    pub(crate) fn has_mutations(&self) -> bool {
        !self.prefix_ops.is_empty()
            || self.replacement.is_some()
            || self.erase_values.is_some()
            || !self.module_ops.is_empty()
    }

    /// Consume the rewriter and return accumulated mutations.
    pub(crate) fn take_mutations(self) -> Mutations {
        Mutations {
            prefix_ops: self.prefix_ops,
            replacement: self.replacement,
            erase_values: self.erase_values,
            module_ops: self.module_ops,
        }
    }

    // === Convenience helpers ===

    /// Replace the current op and also insert prefix ops in one call.
    pub fn replace_with_prefix(&mut self, prefix: Vec<OpRef>, replacement: OpRef) {
        self.prefix_ops.extend(prefix);
        self.replace_op(replacement);
    }
}

/// Apply mutations to the IR context.
///
/// Called by the applicator after a pattern returns `true`.
pub(crate) fn apply_mutations(
    ctx: &mut IrContext,
    original_op: OpRef,
    mutations: Mutations,
    module_first_block: Option<crate::arena::refs::BlockRef>,
) {
    let parent_block = ctx.op(original_op).parent_block;

    // 1. Insert prefix ops before the original op
    if let Some(block) = parent_block {
        for prefix_op in &mutations.prefix_ops {
            ctx.insert_op_before(block, original_op, *prefix_op);
        }
    }

    // 2. Handle replacement or erasure
    if let Some(new_op) = mutations.replacement {
        // RAUW old results → new results
        let old_results: Vec<ValueRef> = ctx.op_results(original_op).to_vec();
        let new_results: Vec<ValueRef> = ctx.op_results(new_op).to_vec();
        debug_assert_eq!(
            old_results.len(),
            new_results.len(),
            "replace_op: result count mismatch ({} vs {})",
            old_results.len(),
            new_results.len()
        );
        for (old_v, new_v) in old_results.iter().zip(new_results.iter()) {
            ctx.replace_all_uses(*old_v, *new_v);
        }

        // Remove old from block, insert new in its place
        if let Some(block) = parent_block {
            // Insert new_op right after original_op, then remove original_op
            // We need to find position after the original to maintain ordering
            let ops = ctx.block(block).ops.to_vec();
            let pos = ops.iter().position(|&o| o == original_op);
            ctx.remove_op_from_block(block, original_op);
            if let Some(pos) = pos {
                let ops_after = ctx.block(block).ops.to_vec();
                if pos < ops_after.len() {
                    ctx.insert_op_before(block, ops_after[pos], new_op);
                } else {
                    ctx.push_op(block, new_op);
                }
            } else {
                ctx.push_op(block, new_op);
            }
        }

        // Clean up old op
        ctx.remove_op(original_op);
    } else if let Some(erase_values) = mutations.erase_values {
        // RAUW old results → replacement values
        let old_results: Vec<ValueRef> = ctx.op_results(original_op).to_vec();
        debug_assert_eq!(
            old_results.len(),
            erase_values.len(),
            "erase_op: replacement value count mismatch ({} vs {})",
            old_results.len(),
            erase_values.len()
        );
        for (old_v, new_v) in old_results.iter().zip(erase_values.iter()) {
            ctx.replace_all_uses(*old_v, *new_v);
        }

        // Remove from block and destroy
        if let Some(block) = parent_block {
            ctx.remove_op_from_block(block, original_op);
        }
        ctx.remove_op(original_op);
    }

    // 3. Add module-level ops
    if let Some(module_block) = module_first_block {
        for module_op in mutations.module_ops {
            ctx.push_op(module_block, module_op);
        }
    }
}
