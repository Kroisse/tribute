//! Value and block mapping for IR cloning and transformation.
//!
//! `IrMapping` tracks correspondences between old and new IR entities
//! during operations like deep cloning. Inspired by MLIR's `IRMapping`.

use std::collections::HashMap;

use crate::refs::{BlockRef, ValueRef};

/// Tracks value and block correspondences during IR transformations.
///
/// Used primarily by `IrContext::clone_op` and `IrContext::clone_region`
/// to remap SSA references in cloned IR. External values (those not in
/// the mapping) pass through unchanged via `lookup_or_default`.
#[derive(Clone, Debug, Default)]
pub struct IrMapping {
    values: HashMap<ValueRef, ValueRef>,
    blocks: HashMap<BlockRef, BlockRef>,
}

impl IrMapping {
    /// Create an empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mapping pre-populated with value correspondences.
    ///
    /// Useful for converting existing `HashMap<ValueRef, ValueRef>` into
    /// an `IrMapping`.
    pub fn from_values(iter: impl IntoIterator<Item = (ValueRef, ValueRef)>) -> Self {
        Self {
            values: iter.into_iter().collect(),
            blocks: HashMap::new(),
        }
    }

    /// Map an old value to a new value.
    pub fn map_value(&mut self, from: ValueRef, to: ValueRef) {
        self.values.insert(from, to);
    }

    /// Map an old block to a new block.
    pub fn map_block(&mut self, from: BlockRef, to: BlockRef) {
        self.blocks.insert(from, to);
    }

    /// Look up a value mapping. Returns `None` if not mapped.
    pub fn lookup_value(&self, v: ValueRef) -> Option<ValueRef> {
        self.values.get(&v).copied()
    }

    /// Look up a value, returning the original if not mapped.
    ///
    /// This is the key semantics for external references: values defined
    /// outside the cloned region are not in the mapping and pass through
    /// unchanged.
    pub fn lookup_value_or_default(&self, v: ValueRef) -> ValueRef {
        self.values.get(&v).copied().unwrap_or(v)
    }

    /// Look up a block mapping. Returns `None` if not mapped.
    pub fn lookup_block(&self, b: BlockRef) -> Option<BlockRef> {
        self.blocks.get(&b).copied()
    }

    /// Look up a block, returning the original if not mapped.
    pub fn lookup_block_or_default(&self, b: BlockRef) -> BlockRef {
        self.blocks.get(&b).copied().unwrap_or(b)
    }

    /// Check if a value is in the mapping.
    pub fn contains_value(&self, v: ValueRef) -> bool {
        self.values.contains_key(&v)
    }

    /// Check if a block is in the mapping.
    pub fn contains_block(&self, b: BlockRef) -> bool {
        self.blocks.contains_key(&b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::IrContext;
    use crate::location::Span;
    use crate::*;

    #[test]
    fn lookup_or_default_returns_original_when_unmapped() {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));

        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        let block = ctx.create_block(context::BlockData {
            location: loc,
            args: vec![context::BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let arg = ctx.block_args(block)[0];

        let mapping = IrMapping::new();
        assert_eq!(mapping.lookup_value_or_default(arg), arg);
        assert_eq!(mapping.lookup_block_or_default(block), block);
    }

    #[test]
    fn mapping_returns_mapped_value() {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));

        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        let block1 = ctx.create_block(context::BlockData {
            location: loc,
            args: vec![context::BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let block2 = ctx.create_block(context::BlockData {
            location: loc,
            args: vec![context::BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });

        let arg1 = ctx.block_args(block1)[0];
        let arg2 = ctx.block_args(block2)[0];

        let mut mapping = IrMapping::new();
        mapping.map_value(arg1, arg2);
        mapping.map_block(block1, block2);

        assert_eq!(mapping.lookup_value_or_default(arg1), arg2);
        assert_eq!(mapping.lookup_block_or_default(block1), block2);
        assert!(mapping.contains_value(arg1));
        assert!(mapping.contains_block(block1));
    }
}
