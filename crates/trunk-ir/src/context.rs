//! IrContext: arena-based mutable IR storage.
//!
//! All IR entities (operations, values, blocks, regions) are stored in
//! `PrimaryMap`s owned by `IrContext`. Entity lists (operands, results)
//! use `EntityList + ListPool` for compact 4-byte per-field storage.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use cranelift_entity::{EntityList, ListPool, PrimaryMap, SecondaryMap};
use smallvec::SmallVec;

use super::refs::*;
use super::types::*;
use crate::diagnostic::{Diagnostic, DiagnosticSeverity};
use crate::ir_mapping::IrMapping;
use crate::location::Span;
use crate::symbol::Symbol;

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
///
/// The value's type is not stored here — it is derived from the defining
/// operation (`OperationData.results`) or block (`BlockArgData.ty`) via
/// [`IrContext::value_ty`].  This eliminates the possibility of type
/// information going out of sync between a value and its definition site.
pub struct ValueData {
    pub def: ValueDef,
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
/// Opaque identifier uniquely tagging an [`IrContext`] instance.
///
/// Produced at context construction from a process-global monotonic
/// counter. Stable for the context's lifetime regardless of moves,
/// and distinct across different contexts. Used to verify that data
/// holding context-local indices (e.g. [`OpRef`](crate::refs::OpRef))
/// is not fed an unrelated context.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IrContextId(u64);

pub struct IrContext {
    id: IrContextId,

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

    /// Type aliases: ordered list of `(name, type)` for stable output.
    type_aliases: Vec<(Symbol, TypeRef)>,
    /// Lookup alias by name.
    type_alias_by_name: HashMap<Symbol, TypeRef>,
    /// Reverse lookup: type → alias name (for printer).
    type_alias_by_type: HashMap<TypeRef, Symbol>,

    /// Diagnostics collected during validation and transformation passes.
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl IrContext {
    /// Create a new empty IR context.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: IrContextId(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
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
            type_aliases: Vec::new(),
            type_alias_by_name: HashMap::new(),
            type_alias_by_type: HashMap::new(),
            diagnostics: RefCell::new(Vec::new()),
        }
    }

    /// Stable identifier for this context instance. See [`IrContextId`].
    pub fn id(&self) -> IrContextId {
        self.id
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /// Report a diagnostic.
    pub fn report(&self, diag: Diagnostic) {
        self.diagnostics.borrow_mut().push(diag);
    }

    /// Report an error diagnostic.
    pub fn report_error(&self, span: Span, message: impl Into<String>) {
        self.report(Diagnostic {
            message: message.into(),
            span,
            severity: DiagnosticSeverity::Error,
            labels: Box::default(),
            note: None,
        });
    }

    /// Report a warning diagnostic.
    pub fn report_warning(&self, span: Span, message: impl Into<String>) {
        self.report(Diagnostic {
            message: message.into(),
            span,
            severity: DiagnosticSeverity::Warning,
            labels: Box::default(),
            note: None,
        });
    }

    /// Take all collected diagnostics, leaving the internal buffer empty.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.diagnostics.get_mut().drain(..).collect()
    }

    /// Return a reference to the collected diagnostics.
    pub fn diagnostics(&self) -> std::cell::Ref<'_, [Diagnostic]> {
        std::cell::Ref::map(self.diagnostics.borrow(), |v| v.as_slice())
    }

    /// Check whether any diagnostics have been reported.
    pub fn has_diagnostics(&self) -> bool {
        !self.diagnostics.borrow().is_empty()
    }

    // ========================================================================
    // Type aliases
    // ========================================================================

    /// Register a type alias. If a name is already registered, it is replaced.
    pub fn register_type_alias(&mut self, name: Symbol, ty: TypeRef) {
        if let Some(old_ty) = self.type_alias_by_name.insert(name, ty) {
            // Remove old reverse mapping
            self.type_alias_by_type.remove(&old_ty);
            // Update ordered list in-place
            if let Some(entry) = self.type_aliases.iter_mut().find(|(n, _)| *n == name) {
                entry.1 = ty;
            }
        } else {
            self.type_aliases.push((name, ty));
        }
        self.type_alias_by_type.insert(ty, name);
    }

    /// Look up a type alias by name.
    pub fn type_alias_by_name(&self, name: Symbol) -> Option<TypeRef> {
        self.type_alias_by_name.get(&name).copied()
    }

    /// Look up an alias name for a given type (reverse lookup for printer).
    pub fn type_alias_by_type(&self, ty: TypeRef) -> Option<Symbol> {
        self.type_alias_by_type.get(&ty).copied()
    }

    /// Get the ordered list of type aliases.
    pub fn type_aliases(&self) -> &[(Symbol, TypeRef)] {
        &self.type_aliases
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

        let num_results = data.results.as_slice(&self.type_pool).len();

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
        for idx in 0..num_results {
            let v = self.values.push(ValueData {
                def: ValueDef::OpResult(op, idx as u32),
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

    /// Set the type of the i-th result of an operation.
    pub fn set_op_result_type(&mut self, op: OpRef, index: u32, new_ty: TypeRef) {
        self.ops[op].results.as_mut_slice(&mut self.type_pool)[index as usize] = new_ty;
    }

    /// Set the i-th operand of an operation, updating use-chains.
    pub fn set_op_operand(&mut self, op: OpRef, index: u32, new_val: ValueRef) {
        let old_val = self.ops[op].operands.as_slice(&self.value_pool)[index as usize];
        if old_val == new_val {
            return;
        }
        self.uses[old_val].retain(|u| !(u.user == op && u.operand_index == index));
        self.ops[op].operands.as_mut_slice(&mut self.value_pool)[index as usize] = new_val;
        self.uses[new_val].push(Use {
            user: op,
            operand_index: index,
        });
    }

    /// Append an operand to an operation's operand list.
    pub fn push_op_operand(&mut self, op: OpRef, val: ValueRef) {
        let index = self.ops[op].operands.len(&self.value_pool) as u32;
        self.ops[op].operands.push(val, &mut self.value_pool);
        self.uses[val].push(Use {
            user: op,
            operand_index: index,
        });
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

    /// Get the type of a value, derived from its definition site.
    ///
    /// For operation results, this reads from `OperationData.results`.
    /// For block arguments, this reads from `BlockArgData.ty`.
    pub fn value_ty(&self, v: ValueRef) -> TypeRef {
        match self.values[v].def {
            ValueDef::OpResult(op, idx) => {
                self.ops[op].results.as_slice(&self.type_pool)[idx as usize]
            }
            ValueDef::BlockArg(block, idx) => self.blocks[block].args[idx as usize].ty,
        }
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
        let num_args = data.args.len();
        let block = self.blocks.push(data);

        // Allocate block argument values
        let mut arg_value_list = EntityList::new();
        for idx in 0..num_args {
            let v = self.values.push(ValueData {
                def: ValueDef::BlockArg(block, idx as u32),
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

    /// Add a new argument to an existing block and return its `ValueRef`.
    pub fn add_block_arg(&mut self, block: BlockRef, arg: BlockArgData) -> ValueRef {
        let index = self.blocks[block].args.len() as u32;
        self.blocks[block].args.push(arg);
        let v = self.values.push(ValueData {
            def: ValueDef::BlockArg(block, index),
        });
        self.block_arg_values[block].push(v, &mut self.value_pool);
        v
    }

    /// Prepend a new argument at position 0 of an existing block.
    ///
    /// Shifts existing block arg `ValueDef` indices by +1, inserts the new
    /// `BlockArgData` at position 0, and returns the `ValueRef` for the new
    /// argument. Existing `ValueRef`s remain valid — only their `ValueDef`
    /// indices are updated.
    pub fn prepend_block_arg(&mut self, block: BlockRef, arg: BlockArgData) -> ValueRef {
        // Shift existing block arg ValueDef indices by +1
        let existing_args = self.block_arg_values[block].as_slice(&self.value_pool);
        let existing_refs: SmallVec<[ValueRef; 8]> = existing_args.into();
        for &v in &existing_refs {
            if let ValueDef::BlockArg(b, idx) = self.values[v].def {
                debug_assert_eq!(b, block);
                self.values[v].def = ValueDef::BlockArg(b, idx + 1);
            }
        }

        // Insert new BlockArgData at position 0
        self.blocks[block].args.insert(0, arg);

        // Create new value for the prepended arg
        let new_value = self.values.push(ValueData {
            def: ValueDef::BlockArg(block, 0),
        });

        // Rebuild block_arg_values EntityList with new value prepended
        let mut new_list = EntityList::new();
        new_list.push(new_value, &mut self.value_pool);
        for &v in &existing_refs {
            new_list.push(v, &mut self.value_pool);
        }
        self.block_arg_values[block] = new_list;

        new_value
    }

    /// Set the type of a block argument.
    pub fn set_block_arg_type(&mut self, block: BlockRef, index: u32, new_ty: TypeRef) {
        self.blocks[block].args[index as usize].ty = new_ty;
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

    /// Detach an operation from its parent block.
    ///
    /// This is a convenience wrapper around `remove_op_from_block` that uses
    /// the operation's own `parent_block` field. Does nothing if the operation
    /// is not attached to any block.
    pub fn detach_op(&mut self, op: OpRef) {
        if let Some(block) = self.ops[op].parent_block {
            self.remove_op_from_block(block, op);
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

    /// Detach a region from its parent operation.
    ///
    /// Clears both the region's `parent_op` back-link and removes the region
    /// from the parent operation's region list. Does nothing if the region
    /// has no parent.
    pub fn detach_region(&mut self, region: RegionRef) {
        if let Some(parent_op) = self.regions[region].parent_op.take() {
            self.ops[parent_op].regions.retain(|r| *r != region);
        }
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

    // ========================================================================
    // Deep Clone
    // ========================================================================

    /// Clone an operation, remapping operands, successors, and nested regions.
    ///
    /// - Operands are remapped via `mapping.lookup_value_or_default()`.
    /// - Successors are remapped via `mapping.lookup_block_or_default()`.
    /// - Nested regions are recursively deep-cloned via [`clone_region`].
    /// - Result values are automatically registered in the mapping
    ///   (old results → new results).
    ///
    /// The cloned operation is **not** attached to any block. Use
    /// [`push_op`] or [`clone_op_into_block`] to insert it.
    pub fn clone_op(&mut self, src_op: OpRef, mapping: &mut IrMapping) -> OpRef {
        // Copy all data from src_op into locals to avoid borrow conflicts.
        let data = &self.ops[src_op];
        let loc = data.location;
        let dialect = data.dialect;
        let name = data.name;
        let attrs = data.attributes.clone();
        let regions: SmallVec<[RegionRef; 4]> = data.regions.clone();
        let successors: SmallVec<[BlockRef; 4]> = data.successors.clone();
        let operands: SmallVec<[ValueRef; 8]> = data.operands.as_slice(&self.value_pool).into();
        let result_types: SmallVec<[TypeRef; 4]> = data.results.as_slice(&self.type_pool).into();

        // Build new operation with remapped operands and successors.
        let mut builder = OperationDataBuilder::new(loc, dialect, name);
        for &v in &operands {
            builder = builder.operand(mapping.lookup_value_or_default(v));
        }
        for ty in &result_types {
            builder = builder.result(*ty);
        }
        for (k, v) in attrs {
            builder = builder.attr(k, v);
        }
        for &r in &regions {
            let cloned_region = self.clone_region(r, mapping);
            builder = builder.region(cloned_region);
        }
        for &s in &successors {
            builder = builder.successor(mapping.lookup_block_or_default(s));
        }

        let op_data = builder.build(self);
        let new_op = self.create_op(op_data);

        // Register result value mappings: old results → new results.
        let old_results: SmallVec<[ValueRef; 4]> =
            self.result_values[src_op].as_slice(&self.value_pool).into();
        let new_results: SmallVec<[ValueRef; 4]> =
            self.result_values[new_op].as_slice(&self.value_pool).into();
        for (old_r, new_r) in old_results.into_iter().zip(new_results) {
            mapping.map_value(old_r, new_r);
        }

        new_op
    }

    /// Deep-clone a region, recursively remapping all values and blocks.
    ///
    /// Uses a 2-pass approach to handle forward block references:
    /// 1. Create all block headers (with arguments), register block/arg
    ///    mappings.
    /// 2. Clone operations in each block using [`clone_op`].
    ///
    /// Values not in the mapping (external references) pass through
    /// unchanged.
    pub fn clone_region(&mut self, src_region: RegionRef, mapping: &mut IrMapping) -> RegionRef {
        let loc = self.regions[src_region].location;
        let src_blocks: SmallVec<[BlockRef; 4]> = self.regions[src_region].blocks.clone();

        let mut new_blocks = Vec::with_capacity(src_blocks.len());

        // Pass 1: create block headers and register block/arg mappings.
        for &src_block in &src_blocks {
            let src_args: SmallVec<[ValueRef; 4]> = self.block_arg_values[src_block]
                .as_slice(&self.value_pool)
                .into();
            let arg_data: Vec<BlockArgData> = self.blocks[src_block]
                .args
                .iter()
                .map(|a| BlockArgData {
                    ty: a.ty,
                    attrs: a.attrs.clone(),
                })
                .collect();
            let block_loc = self.blocks[src_block].location;

            let new_block = self.create_block(BlockData {
                location: block_loc,
                args: arg_data,
                ops: SmallVec::new(),
                parent_region: None,
            });

            mapping.map_block(src_block, new_block);

            let new_args: SmallVec<[ValueRef; 4]> = self.block_arg_values[new_block]
                .as_slice(&self.value_pool)
                .into();
            for (old_arg, new_arg) in src_args.into_iter().zip(new_args) {
                mapping.map_value(old_arg, new_arg);
            }

            new_blocks.push(new_block);
        }

        // Pass 2: clone operations in each block.
        for (&src_block, &new_block) in src_blocks.iter().zip(new_blocks.iter()) {
            let src_ops: SmallVec<[OpRef; 4]> = self.blocks[src_block].ops.clone();
            for &op in &src_ops {
                let new_op = self.clone_op(op, mapping);
                self.push_op(new_block, new_op);
            }
        }

        self.create_region(RegionData {
            location: loc,
            blocks: new_blocks.into(),
            parent_op: None,
        })
    }

    /// Clone an operation and append it to a destination block.
    ///
    /// Convenience wrapper around [`clone_op`] + [`push_op`].
    pub fn clone_op_into_block(
        &mut self,
        dest: BlockRef,
        src_op: OpRef,
        mapping: &mut IrMapping,
    ) -> OpRef {
        let new_op = self.clone_op(src_op, mapping);
        self.push_op(dest, new_op);
        new_op
    }

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
    use crate::location::Span;
    use crate::symbol::Symbol;
    use smallvec::smallvec;

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    #[test]
    fn create_op_and_read_back() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .attr("value", Attribute::Int(42))
            .build(&mut ctx);

        let op = ctx.create_op(data);

        assert_eq!(ctx.op(op).dialect, Symbol::new("arith"));
        assert_eq!(ctx.op(op).name, Symbol::new("const"));
        assert_eq!(ctx.op_result_types(op), &[i32_ty]);
        assert_eq!(
            ctx.op(op).attributes.get(&Symbol::new("value")),
            Some(&Attribute::Int(42))
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
    fn prepend_block_arg() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);
        let f64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());

        // Create a block with 2 args of type i32
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

        let old_a0 = ctx.block_arg(block, 0);
        let old_a1 = ctx.block_arg(block, 1);

        // Prepend an f64 arg
        let new_arg = ctx.prepend_block_arg(
            block,
            BlockArgData {
                ty: f64_ty,
                attrs: BTreeMap::new(),
            },
        );

        // New arg should be at index 0
        assert_eq!(ctx.block_args(block).len(), 3);
        assert_eq!(ctx.block_arg(block, 0), new_arg);
        assert_eq!(ctx.value_ty(new_arg), f64_ty);
        assert_eq!(ctx.value_def(new_arg), ValueDef::BlockArg(block, 0));

        // Old args should now have shifted indices
        assert_eq!(ctx.block_arg(block, 1), old_a0);
        assert_eq!(ctx.block_arg(block, 2), old_a1);
        assert_eq!(ctx.value_def(old_a0), ValueDef::BlockArg(block, 1));
        assert_eq!(ctx.value_def(old_a1), ValueDef::BlockArg(block, 2));

        // Types should be preserved
        assert_eq!(ctx.value_ty(old_a0), i32_ty);
        assert_eq!(ctx.value_ty(old_a1), i32_ty);

        // BlockArgData should also match
        assert_eq!(ctx.block(block).args[0].ty, f64_ty);
        assert_eq!(ctx.block(block).args[1].ty, i32_ty);
        assert_eq!(ctx.block(block).args[2].ty, i32_ty);
    }

    #[test]
    fn prepend_block_arg_empty_block() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        });

        let new_arg = ctx.prepend_block_arg(
            block,
            BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            },
        );

        assert_eq!(ctx.block_args(block).len(), 1);
        assert_eq!(ctx.block_arg(block, 0), new_arg);
        assert_eq!(ctx.value_def(new_arg), ValueDef::BlockArg(block, 0));
    }

    #[test]
    fn set_op_result_type_updates_value_ty() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);
        let f64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());

        let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("op"))
            .result(i32_ty)
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(data);

        let r0 = ctx.op_result(op, 0);
        let r1 = ctx.op_result(op, 1);
        assert_eq!(ctx.value_ty(r0), i32_ty);
        assert_eq!(ctx.value_ty(r1), i32_ty);

        // Mutate the second result type
        ctx.set_op_result_type(op, 1, f64_ty);

        // The existing ValueRef should now reflect the new type
        assert_eq!(ctx.value_ty(r0), i32_ty); // unchanged
        assert_eq!(ctx.value_ty(r1), f64_ty); // updated
        assert_eq!(ctx.op_result_types(op), &[i32_ty, f64_ty]);
    }

    #[test]
    fn set_block_arg_type_updates_value_ty() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);
        let f64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());

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

        let a0 = ctx.block_arg(block, 0);
        let a1 = ctx.block_arg(block, 1);
        assert_eq!(ctx.value_ty(a0), i32_ty);
        assert_eq!(ctx.value_ty(a1), i32_ty);

        // Mutate the first arg type
        ctx.set_block_arg_type(block, 0, f64_ty);

        // The existing ValueRef should now reflect the new type
        assert_eq!(ctx.value_ty(a0), f64_ty); // updated
        assert_eq!(ctx.value_ty(a1), i32_ty); // unchanged
        assert_eq!(ctx.block(block).args[0].ty, f64_ty);
        assert_eq!(ctx.block(block).args[1].ty, i32_ty);
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
    fn set_op_operand_updates_use_chains() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        // Create two constants: v_old and v_new
        let data_old = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op_old = ctx.create_op(data_old);
        let v_old = ctx.op_result(op_old, 0);

        let data_new = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op_new = ctx.create_op(data_new);
        let v_new = ctx.op_result(op_new, 0);

        // Create op that uses v_old at operand 0
        let data_user = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("neg"))
            .operand(v_old)
            .result(i32_ty)
            .build(&mut ctx);
        let op_user = ctx.create_op(data_user);

        // Verify initial state
        assert_eq!(ctx.uses(v_old).len(), 1);
        assert_eq!(
            ctx.uses(v_old)[0],
            Use {
                user: op_user,
                operand_index: 0
            }
        );
        assert!(!ctx.has_uses(v_new));

        // Set operand to v_new
        ctx.set_op_operand(op_user, 0, v_new);

        // v_old should have no uses
        assert!(!ctx.has_uses(v_old));
        // v_new should have the use
        assert_eq!(ctx.uses(v_new).len(), 1);
        assert_eq!(
            ctx.uses(v_new)[0],
            Use {
                user: op_user,
                operand_index: 0
            }
        );
        // Operand should reflect the change
        assert_eq!(ctx.op_operands(op_user)[0], v_new);

        // Set same value again (no-op)
        ctx.set_op_operand(op_user, 0, v_new);
        assert_eq!(ctx.uses(v_new).len(), 1); // unchanged
        assert!(!ctx.has_uses(v_old)); // still no uses
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

    // ====================================================================
    // Deep clone tests
    // ====================================================================

    /// Helper: parse IR, clone the first func's body region, add the
    /// clone as a new func, and print the whole module.
    fn clone_first_func_region(input: &str, new_name: &'static str) -> String {
        use crate::parser::parse_test_module;
        use crate::printer::print_module;

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        // Find the first func.func in the module's top-level block
        let module_region = ctx.op(module.op()).regions[0];
        let module_block = ctx.region(module_region).blocks[0];
        let func_op = ctx
            .block(module_block)
            .ops
            .iter()
            .copied()
            .find(|&op| {
                ctx.op(op).dialect == Symbol::new("func") && ctx.op(op).name == Symbol::new("func")
            })
            .expect("no func.func found");

        // Clone the body region
        let body_region = ctx.op(func_op).regions[0];
        let mut mapping = IrMapping::new();
        let cloned_region = ctx.clone_region(body_region, &mut mapping);

        // Build a new func.func with the cloned region
        let func_ty = ctx
            .op(func_op)
            .attributes
            .get(&Symbol::new("type"))
            .cloned()
            .unwrap();
        let new_func = OperationDataBuilder::new(
            ctx.op(func_op).location,
            Symbol::new("func"),
            Symbol::new("func"),
        )
        .attr("sym_name", Attribute::Symbol(Symbol::new(new_name)))
        .attr("type", func_ty)
        .region(cloned_region)
        .build(&mut ctx);
        let new_func_op = ctx.create_op(new_func);

        // Add to module
        let module_region = ctx.op(module.op()).regions[0];
        let module_block = ctx.region(module_region).blocks[0];
        ctx.push_op(module_block, new_func_op);

        print_module(&ctx, module.op())
    }

    #[test]
    fn clone_flat_ops() {
        let output = clone_first_func_region(
            r#"core.module @test {
  func.func @original() -> core.i32 {
    %0 = arith.const {value = 7} : core.i32
    %1 = arith.negi %0 : core.i32
    func.return %1
  }
}"#,
            "cloned",
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn clone_op_with_nested_region() {
        let output = clone_first_func_region(
            r#"core.module @test {
  func.func @original(%0: core.i1) -> core.i32 {
    %1 = scf.if %0 : core.i32 {
      %2 = arith.const {value = 1} : core.i32
      scf.yield %2
    } {
      %3 = arith.const {value = 2} : core.i32
      scf.yield %3
    }
    func.return %1
  }
}"#,
            "cloned",
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn clone_region_with_forward_block_ref() {
        let output = clone_first_func_region(
            r#"core.module @test {
  func.func @original() -> core.i32 {
  ^bb0:
    %0 = arith.const {value = 1} : core.i32
    cf.br %0 [^bb1]
  ^bb1(%1: core.i32):
    func.return %1
  }
}"#,
            "cloned",
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn clone_external_values_passthrough() {
        // External values (defined outside the cloned region) pass through
        // unchanged. We test this by cloning a flat op directly.
        use crate::printer::print_op;

        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let ext_data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let ext_op = ctx.create_op(ext_data);
        let ext_val = ctx.op_result(ext_op, 0);

        let op_data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("neg"))
            .operand(ext_val)
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(op_data);

        let mut mapping = IrMapping::new();
        let cloned = ctx.clone_op(op, &mut mapping);

        // Cloned op should reference the same external value
        assert_eq!(ctx.op_operands(cloned), &[ext_val]);

        // Both original and clone should print the same textual form
        assert_eq!(print_op(&ctx, op), print_op(&ctx, cloned));
    }

    #[test]
    fn clone_op_result_mapping() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("multi"))
            .result(i32_ty)
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(data);
        let r0 = ctx.op_result(op, 0);
        let r1 = ctx.op_result(op, 1);

        let mut mapping = IrMapping::new();
        let cloned = ctx.clone_op(op, &mut mapping);

        let cr0 = ctx.op_result(cloned, 0);
        let cr1 = ctx.op_result(cloned, 1);

        // Old results should map to new results
        assert_eq!(mapping.lookup_value(r0), Some(cr0));
        assert_eq!(mapping.lookup_value(r1), Some(cr1));
        assert_ne!(r0, cr0);
        assert_ne!(r1, cr1);
    }
}
