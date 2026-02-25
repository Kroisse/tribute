//! IrContext: arena-based mutable IR storage.
//!
//! All IR entities (operations, values, blocks, regions) are stored in
//! `PrimaryMap`s owned by `IrContext`. Entity lists (operands, results)
//! use `EntityList + ListPool` for compact 4-byte per-field storage.

use std::collections::BTreeMap;

use cranelift_entity::{EntityList, ListPool, PrimaryMap, SecondaryMap};
use smallvec::SmallVec;

use super::refs::*;
use super::types::*;
use crate::ir::Symbol;

// ============================================================================
// Use-chain
// ============================================================================

/// A single use of a value: which operation uses it, at which operand index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Use {
    pub user: OpRef,
    pub operand_index: u32,
}

// ============================================================================
// Entity data types
// ============================================================================

/// Data for a single operation in the arena.
pub struct OperationData {
    pub location: Location,
    pub dialect: Symbol,
    pub name: Symbol,
    pub operands: EntityList<ValueRef>,
    pub results: EntityList<TypeRef>,
    pub attributes: BTreeMap<Symbol, Attribute>,
    pub regions: SmallVec<[RegionRef; 4]>,
    pub successors: SmallVec<[BlockRef; 4]>,
    pub parent_block: Option<BlockRef>,
}

/// Data for a single SSA value.
pub struct ValueData {
    pub def: ValueDef,
    pub ty: TypeRef,
}

/// Data for a block argument (type + optional attributes).
#[derive(Clone, Debug)]
pub struct BlockArgData {
    pub ty: TypeRef,
    pub attrs: BTreeMap<Symbol, Attribute>,
}

/// Data for a basic block.
pub struct BlockData {
    pub location: Location,
    pub args: Vec<BlockArgData>,
    pub ops: SmallVec<[OpRef; 4]>,
    pub parent_region: Option<RegionRef>,
}

/// Data for a region (list of blocks).
pub struct RegionData {
    pub location: Location,
    pub blocks: SmallVec<[BlockRef; 4]>,
    pub parent_op: Option<OpRef>,
}

// ============================================================================
// IrContext
// ============================================================================

/// Arena-based mutable IR context.
///
/// Owns all IR entities and provides methods for creating, querying,
/// and mutating them. Use-chains are automatically maintained.
pub struct IrContext {
    ops: PrimaryMap<OpRef, OperationData>,
    values: PrimaryMap<ValueRef, ValueData>,
    blocks: PrimaryMap<BlockRef, BlockData>,
    regions: PrimaryMap<RegionRef, RegionData>,

    /// Use-chain: for each value, the list of operations that use it.
    uses: SecondaryMap<ValueRef, SmallVec<[Use; 2]>>,

    /// Type and path interners.
    pub types: TypeInterner,
    pub paths: PathInterner,

    /// Backing pools for EntityList storage.
    value_pool: ListPool<ValueRef>,
    type_pool: ListPool<TypeRef>,

    /// Mapping from operation to its result ValueRefs.
    result_values: SecondaryMap<OpRef, EntityList<ValueRef>>,
    /// Mapping from block to its argument ValueRefs.
    block_arg_values: SecondaryMap<BlockRef, EntityList<ValueRef>>,
}

impl IrContext {
    /// Create a new empty IR context.
    pub fn new() -> Self {
        Self {
            ops: PrimaryMap::new(),
            values: PrimaryMap::new(),
            blocks: PrimaryMap::new(),
            regions: PrimaryMap::new(),
            uses: SecondaryMap::new(),
            types: TypeInterner::new(),
            paths: PathInterner::new(),
            value_pool: ListPool::new(),
            type_pool: ListPool::new(),
            result_values: SecondaryMap::new(),
            block_arg_values: SecondaryMap::new(),
        }
    }

    // ========================================================================
    // Operation
    // ========================================================================

    /// Create a new operation and allocate result values for it.
    ///
    /// The operation's operands are registered in the use-chain.
    /// The operation must not have a `parent_block` set — use `push_op` to
    /// attach it to a block after creation.
    ///
    /// # Panics
    ///
    /// Panics if `data.parent_block` is `Some`, or if any region in
    /// `data.regions` already belongs to another operation.
    pub fn create_op(&mut self, data: OperationData) -> OpRef {
        assert!(
            data.parent_block.is_none(),
            "create_op: operation must not have parent_block set; \
             use push_op to attach to a block after creation",
        );

        // Register uses for operands
        let operand_slice: SmallVec<[ValueRef; 8]> =
            data.operands.as_slice(&self.value_pool).into();

        let result_types: SmallVec<[TypeRef; 4]> = data.results.as_slice(&self.type_pool).into();

        let regions: SmallVec<[RegionRef; 4]> = data.regions.clone();

        let op = self.ops.push(data);

        // Back-link owned regions to this operation
        for &r in &regions {
            if let Some(existing) = self.regions[r].parent_op {
                panic!(
                    "create_op: region {r} already belongs to operation {existing}; \
                     cannot reassign to {op}",
                );
            }
            self.regions[r].parent_op = Some(op);
        }

        // Register operand uses
        for (idx, &val) in operand_slice.iter().enumerate() {
            self.uses[val].push(Use {
                user: op,
                operand_index: idx as u32,
            });
        }

        // Allocate result values
        let mut result_value_list = EntityList::new();
        for (idx, &ty) in result_types.iter().enumerate() {
            let v = self.values.push(ValueData {
                def: ValueDef::OpResult(op, idx as u32),
                ty,
            });
            result_value_list.push(v, &mut self.value_pool);
        }
        self.result_values[op] = result_value_list;

        op
    }

    /// Get immutable reference to operation data.
    pub fn op(&self, op: OpRef) -> &OperationData {
        &self.ops[op]
    }

    /// Get mutable reference to operation data.
    ///
    /// **Warning**: Modifying operands directly will desync the use-chain.
    /// Prefer `replace_all_uses` or re-creating the operation.
    pub fn op_mut(&mut self, op: OpRef) -> &mut OperationData {
        &mut self.ops[op]
    }

    /// Get the operands of an operation as a slice.
    pub fn op_operands(&self, op: OpRef) -> &[ValueRef] {
        self.ops[op].operands.as_slice(&self.value_pool)
    }

    /// Get the result types of an operation as a slice.
    pub fn op_result_types(&self, op: OpRef) -> &[TypeRef] {
        self.ops[op].results.as_slice(&self.type_pool)
    }

    /// Get the i-th result value of an operation.
    pub fn op_result(&self, op: OpRef, index: u32) -> ValueRef {
        self.result_values[op].as_slice(&self.value_pool)[index as usize]
    }

    /// Get all result values of an operation.
    pub fn op_results(&self, op: OpRef) -> &[ValueRef] {
        self.result_values[op].as_slice(&self.value_pool)
    }

    /// Remove an operation, clearing its use-chain entries.
    ///
    /// Does NOT remove it from its parent block. Use `remove_op_from_block` first.
    ///
    /// # Panics
    ///
    /// Panics if any result value of the operation still has uses,
    /// as that would leave dangling references.
    pub fn remove_op(&mut self, op: OpRef) {
        // Refuse to remove an op that is still attached to a block
        assert!(
            self.ops[op].parent_block.is_none(),
            "remove_op: operation {op} is still attached to block {:?}; \
             call remove_op_from_block first",
            self.ops[op].parent_block.unwrap(),
        );

        // Check that result values have no remaining uses
        let results: SmallVec<[ValueRef; 4]> =
            self.result_values[op].as_slice(&self.value_pool).into();
        for &val in &results {
            assert!(
                self.uses[val].is_empty(),
                "remove_op: result value {val} still has {} use(s); \
                 replace all uses before removing the operation",
                self.uses[val].len()
            );
        }

        let operands: SmallVec<[ValueRef; 8]> =
            self.ops[op].operands.as_slice(&self.value_pool).into();
        for (idx, &val) in operands.iter().enumerate() {
            self.uses[val].retain(|u| !(u.user == op && u.operand_index == idx as u32));
        }
    }

    // ========================================================================
    // Value
    // ========================================================================

    /// Get immutable reference to value data.
    pub fn value(&self, v: ValueRef) -> &ValueData {
        &self.values[v]
    }

    /// Get the type of a value.
    pub fn value_ty(&self, v: ValueRef) -> TypeRef {
        self.values[v].ty
    }

    /// Get the definition of a value.
    pub fn value_def(&self, v: ValueRef) -> ValueDef {
        self.values[v].def
    }

    // ========================================================================
    // Block
    // ========================================================================

    /// Create a new block and allocate argument values for it.
    pub fn create_block(&mut self, data: BlockData) -> BlockRef {
        let arg_types: Vec<TypeRef> = data.args.iter().map(|a| a.ty).collect();
        let block = self.blocks.push(data);

        // Allocate block argument values
        let mut arg_value_list = EntityList::new();
        for (idx, ty) in arg_types.into_iter().enumerate() {
            let v = self.values.push(ValueData {
                def: ValueDef::BlockArg(block, idx as u32),
                ty,
            });
            arg_value_list.push(v, &mut self.value_pool);
        }
        self.block_arg_values[block] = arg_value_list;

        block
    }

    /// Get immutable reference to block data.
    pub fn block(&self, b: BlockRef) -> &BlockData {
        &self.blocks[b]
    }

    /// Get mutable reference to block data.
    pub fn block_mut(&mut self, b: BlockRef) -> &mut BlockData {
        &mut self.blocks[b]
    }

    /// Get the i-th block argument value.
    pub fn block_arg(&self, b: BlockRef, index: u32) -> ValueRef {
        self.block_arg_values[b].as_slice(&self.value_pool)[index as usize]
    }

    /// Get all block argument values.
    pub fn block_args(&self, b: BlockRef) -> &[ValueRef] {
        self.block_arg_values[b].as_slice(&self.value_pool)
    }

    /// Append an operation to the end of a block.
    ///
    /// # Panics
    ///
    /// Panics if the operation already belongs to a block.
    pub fn push_op(&mut self, block: BlockRef, op: OpRef) {
        assert!(
            self.ops[op].parent_block.is_none(),
            "push_op: operation {op} already belongs to block {:?}; \
             remove it from the old block first",
            self.ops[op].parent_block.unwrap(),
        );
        self.ops[op].parent_block = Some(block);
        self.blocks[block].ops.push(op);
    }

    /// Insert an operation before `before` in the given block.
    ///
    /// # Panics
    ///
    /// Panics if the operation already belongs to a block, or if `before`
    /// is not found in the block.
    pub fn insert_op_before(&mut self, block: BlockRef, before: OpRef, op: OpRef) {
        assert!(
            self.ops[op].parent_block.is_none(),
            "insert_op_before: operation {op} already belongs to block {:?}; \
             remove it from the old block first",
            self.ops[op].parent_block.unwrap(),
        );
        let ops = &mut self.blocks[block].ops;
        let pos = ops
            .iter()
            .position(|&o| o == before)
            .expect("insert_op_before: `before` op not found in block");
        ops.insert(pos, op);
        self.ops[op].parent_block = Some(block);
    }

    /// Remove an operation from a block (does not destroy the operation).
    ///
    /// Only clears the operation's `parent_block` if it matches the given block.
    pub fn remove_op_from_block(&mut self, block: BlockRef, op: OpRef) {
        self.blocks[block].ops.retain(|o| *o != op);
        if self.ops[op].parent_block == Some(block) {
            self.ops[op].parent_block = None;
        }
    }

    // ========================================================================
    // Region
    // ========================================================================

    /// Create a new region.
    ///
    /// # Panics
    ///
    /// Panics if any block in `data.blocks` already belongs to another region.
    pub fn create_region(&mut self, data: RegionData) -> RegionRef {
        let region = self.regions.push(data);

        // Set parent_region on all blocks in this region
        let blocks: SmallVec<[BlockRef; 4]> = self.regions[region].blocks.clone();
        for &b in &blocks {
            if let Some(existing) = self.blocks[b].parent_region {
                panic!(
                    "create_region: block {b} already belongs to region {existing}; \
                     cannot reassign to {region}",
                );
            }
            self.blocks[b].parent_region = Some(region);
        }

        region
    }

    /// Get immutable reference to region data.
    pub fn region(&self, r: RegionRef) -> &RegionData {
        &self.regions[r]
    }

    /// Get mutable reference to region data.
    pub fn region_mut(&mut self, r: RegionRef) -> &mut RegionData {
        &mut self.regions[r]
    }

    // ========================================================================
    // Use-chain
    // ========================================================================

    /// Get all uses of a value.
    pub fn uses(&self, v: ValueRef) -> &[Use] {
        &self.uses[v]
    }

    /// Check if a value has any uses.
    pub fn has_uses(&self, v: ValueRef) -> bool {
        !self.uses[v].is_empty()
    }

    // ========================================================================
    // RAUW (Replace All Uses With)
    // ========================================================================

    /// Replace all uses of `old` with `new` in all operations.
    ///
    /// Updates both operand lists and the use-chain.
    pub fn replace_all_uses(&mut self, old: ValueRef, new: ValueRef) {
        if old == new {
            return;
        }
        // Take the old use list
        let old_uses = std::mem::take(&mut self.uses[old]);

        for u in &old_uses {
            // Update the operand in the operation
            let operands = &mut self.ops[u.user].operands;
            let slice = operands.as_mut_slice(&mut self.value_pool);
            debug_assert_eq!(slice[u.operand_index as usize], old);
            slice[u.operand_index as usize] = new;

            // Add this use to the new value's use-chain
            self.uses[new].push(Use {
                user: u.user,
                operand_index: u.operand_index,
            });
        }
    }
}

impl Default for IrContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper constructors for OperationData
// ============================================================================

impl OperationData {
    /// Create a new OperationData with the given basic fields.
    /// Operands and results must be added via the pool-backed EntityList.
    pub fn new(location: Location, dialect: Symbol, name: Symbol) -> Self {
        Self {
            location,
            dialect,
            name,
            operands: EntityList::new(),
            results: EntityList::new(),
            attributes: BTreeMap::new(),
            regions: SmallVec::new(),
            successors: SmallVec::new(),
            parent_block: None,
        }
    }
}

/// Builder for constructing `OperationData` with pool-backed lists.
///
/// Collects operands and result types into `Vec`s, then packs them
/// into `EntityList`s on `build()`.
pub struct OperationDataBuilder {
    location: Location,
    dialect: Symbol,
    name: Symbol,
    operands: Vec<ValueRef>,
    results: Vec<TypeRef>,
    attributes: BTreeMap<Symbol, Attribute>,
    regions: SmallVec<[RegionRef; 4]>,
    successors: SmallVec<[BlockRef; 4]>,
}

impl OperationDataBuilder {
    pub fn new(location: Location, dialect: Symbol, name: Symbol) -> Self {
        Self {
            location,
            dialect,
            name,
            operands: Vec::new(),
            results: Vec::new(),
            attributes: BTreeMap::new(),
            regions: SmallVec::new(),
            successors: SmallVec::new(),
        }
    }

    pub fn operand(mut self, v: ValueRef) -> Self {
        self.operands.push(v);
        self
    }

    pub fn operands(mut self, vs: impl IntoIterator<Item = ValueRef>) -> Self {
        self.operands.extend(vs);
        self
    }

    pub fn result(mut self, ty: TypeRef) -> Self {
        self.results.push(ty);
        self
    }

    pub fn results(mut self, tys: impl IntoIterator<Item = TypeRef>) -> Self {
        self.results.extend(tys);
        self
    }

    pub fn attr(mut self, key: impl Into<Symbol>, val: Attribute) -> Self {
        self.attributes.insert(key.into(), val);
        self
    }

    pub fn region(mut self, r: RegionRef) -> Self {
        self.regions.push(r);
        self
    }

    pub fn successor(mut self, b: BlockRef) -> Self {
        self.successors.push(b);
        self
    }

    /// Build the `OperationData`, packing vecs into `EntityList`s using
    /// the context's pools.
    pub fn build(self, ctx: &mut IrContext) -> OperationData {
        let mut operands = EntityList::new();
        for v in self.operands {
            operands.push(v, &mut ctx.value_pool);
        }
        let mut results = EntityList::new();
        for ty in self.results {
            results.push(ty, &mut ctx.type_pool);
        }
        OperationData {
            location: self.location,
            dialect: self.dialect,
            name: self.name,
            operands,
            results,
            attributes: self.attributes,
            regions: self.regions,
            successors: self.successors,
            parent_block: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i32"),
            params: smallvec![],
            attrs: BTreeMap::new(),
        })
    }

    #[test]
    fn create_op_and_read_back() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .attr("value", Attribute::IntBits(42))
            .build(&mut ctx);

        let op = ctx.create_op(data);

        assert_eq!(ctx.op(op).dialect, Symbol::new("arith"));
        assert_eq!(ctx.op(op).name, Symbol::new("const"));
        assert_eq!(ctx.op_result_types(op), &[i32_ty]);
        assert_eq!(
            ctx.op(op).attributes.get(&Symbol::new("value")),
            Some(&Attribute::IntBits(42))
        );
    }

    #[test]
    fn op_result_values() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("multi"))
            .result(i32_ty)
            .result(i32_ty)
            .build(&mut ctx);

        let op = ctx.create_op(data);

        let results = ctx.op_results(op);
        assert_eq!(results.len(), 2);

        let r0 = ctx.op_result(op, 0);
        let r1 = ctx.op_result(op, 1);
        assert_ne!(r0, r1);

        assert_eq!(ctx.value_ty(r0), i32_ty);
        assert_eq!(ctx.value_ty(r1), i32_ty);
        assert_eq!(ctx.value_def(r0), ValueDef::OpResult(op, 0));
        assert_eq!(ctx.value_def(r1), ValueDef::OpResult(op, 1));
    }

    #[test]
    fn block_args() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: i32_ty,
                    attrs: BTreeMap::new(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: BTreeMap::new(),
                },
            ],
            ops: SmallVec::new(),
            parent_region: None,
        });

        let args = ctx.block_args(block);
        assert_eq!(args.len(), 2);

        let a0 = ctx.block_arg(block, 0);
        let a1 = ctx.block_arg(block, 1);
        assert_ne!(a0, a1);

        assert_eq!(ctx.value_ty(a0), i32_ty);
        assert_eq!(ctx.value_def(a0), ValueDef::BlockArg(block, 0));
        assert_eq!(ctx.value_def(a1), ValueDef::BlockArg(block, 1));
    }

    #[test]
    fn use_chain_tracking() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        // Create op1 with a result
        let data1 = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(data1);
        let v1 = ctx.op_result(op1, 0);

        // No uses yet
        assert!(!ctx.has_uses(v1));
        assert!(ctx.uses(v1).is_empty());

        // Create op2 that uses v1
        let data2 = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("neg"))
            .operand(v1)
            .result(i32_ty)
            .build(&mut ctx);
        let op2 = ctx.create_op(data2);

        // v1 now has one use
        assert!(ctx.has_uses(v1));
        let uses = ctx.uses(v1);
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].user, op2);
        assert_eq!(uses[0].operand_index, 0);
    }

    #[test]
    fn rauw() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        // Create two constants
        let data1 = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(data1);
        let v_old = ctx.op_result(op1, 0);

        let data2 = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op2 = ctx.create_op(data2);
        let v_new = ctx.op_result(op2, 0);

        // Create op3 that uses v_old twice
        let data3 = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("add"))
            .operand(v_old)
            .operand(v_old)
            .result(i32_ty)
            .build(&mut ctx);
        let op3 = ctx.create_op(data3);

        // Verify initial uses
        assert_eq!(ctx.uses(v_old).len(), 2);
        assert!(!ctx.has_uses(v_new));

        // RAUW
        ctx.replace_all_uses(v_old, v_new);

        // v_old should have no uses
        assert!(!ctx.has_uses(v_old));

        // v_new should have two uses
        assert_eq!(ctx.uses(v_new).len(), 2);

        // Operands of op3 should now point to v_new
        let operands = ctx.op_operands(op3);
        assert_eq!(operands[0], v_new);
        assert_eq!(operands[1], v_new);
    }

    #[test]
    fn parent_tracking() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        // Create a block
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });

        // Create an op
        let data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(data);

        // Not in a block yet
        assert_eq!(ctx.op(op).parent_block, None);

        // Push into block
        ctx.push_op(block, op);
        assert_eq!(ctx.op(op).parent_block, Some(block));
        assert_eq!(ctx.block(block).ops.as_slice(), &[op]);

        // Create a region containing the block
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        assert_eq!(ctx.block(block).parent_region, Some(region));
    }

    #[test]
    fn insert_op_before() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });

        let mk_op = |ctx: &mut IrContext, name: &'static str| {
            let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new(name))
                .result(i32_ty)
                .build(ctx);
            ctx.create_op(data)
        };

        let op_a = mk_op(&mut ctx, "a");
        let op_c = mk_op(&mut ctx, "c");
        ctx.push_op(block, op_a);
        ctx.push_op(block, op_c);

        // Insert op_b before op_c
        let op_b = mk_op(&mut ctx, "b");
        ctx.insert_op_before(block, op_c, op_b);

        assert_eq!(ctx.block(block).ops.as_slice(), &[op_a, op_b, op_c]);
        assert_eq!(ctx.op(op_b).parent_block, Some(block));
    }

    #[test]
    fn remove_op_from_block() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });

        let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("x"))
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(data);
        ctx.push_op(block, op);

        assert_eq!(ctx.block(block).ops.len(), 1);
        ctx.remove_op_from_block(block, op);
        assert!(ctx.block(block).ops.is_empty());
        assert_eq!(ctx.op(op).parent_block, None);
    }

    #[test]
    fn entity_ref_display() {
        use cranelift_entity::EntityRef;

        let op = OpRef::new(0);
        assert_eq!(format!("{op}"), "op0");

        let v = ValueRef::new(5);
        assert_eq!(format!("{v}"), "v5");

        let b = BlockRef::new(2);
        assert_eq!(format!("{b}"), "block2");

        let r = RegionRef::new(1);
        assert_eq!(format!("{r}"), "region1");

        let ty = TypeRef::new(3);
        assert_eq!(format!("{ty}"), "ty3");

        let p = PathRef::new(0);
        assert_eq!(format!("{p}"), "path0");
    }

    #[test]
    #[should_panic(expected = "still has")]
    fn remove_op_panics_when_result_has_uses() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        // Create op1 with a result
        let data1 = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("a"))
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(data1);
        let v1 = ctx.op_result(op1, 0);

        // Create op2 that uses op1's result
        let data2 = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("b"))
            .operand(v1)
            .result(i32_ty)
            .build(&mut ctx);
        let _op2 = ctx.create_op(data2);

        // This should panic because v1 still has uses
        ctx.remove_op(op1);
    }

    #[test]
    fn create_op_sets_region_parent() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        // Create a block and region first
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        // Region has no parent yet
        assert_eq!(ctx.region(region).parent_op, None);

        // Create an op that owns the region
        let data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .result(i32_ty)
            .region(region)
            .build(&mut ctx);
        let op = ctx.create_op(data);

        // Now the region should be back-linked to the op
        assert_eq!(ctx.region(region).parent_op, Some(op));
    }

    #[test]
    #[should_panic(expected = "must not have parent_block set")]
    fn create_op_panics_when_parent_block_preset() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });

        // Manually construct OperationData with parent_block set
        let mut data = OperationData::new(loc, Symbol::new("test"), Symbol::new("x"));
        data.results.push(i32_ty, &mut ctx.type_pool);
        data.parent_block = Some(block);

        // Should panic
        ctx.create_op(data);
    }

    #[test]
    #[should_panic(expected = "already belongs to operation")]
    fn create_op_panics_when_region_already_owned() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        // First op takes ownership of the region
        let data1 = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .result(i32_ty)
            .region(region)
            .build(&mut ctx);
        let _op1 = ctx.create_op(data1);

        // Second op tries to take the same region — should panic
        let data2 = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .result(i32_ty)
            .region(region)
            .build(&mut ctx);
        ctx.create_op(data2);
    }

    #[test]
    #[should_panic(expected = "already belongs to region")]
    fn create_region_panics_when_block_already_owned() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });

        // First region takes ownership of the block
        let _r1 = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        // Second region tries to take the same block — should panic
        ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
    }
}
