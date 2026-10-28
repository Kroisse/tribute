//! IrContext: arena-based mutable IR storage.
//!
//! All IR entities (operations, values, blocks, regions) are stored in
//! `PrimaryMap`s owned by `IrContext`. Entity lists (operands, results)
//! use `EntityList + ListPool` for compact 4-byte per-field storage.

use rustc_hash::FxHashMap as HashMap;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

use cranelift_entity::{EntityList, EntityRef, ListPool, PrimaryMap, SecondaryMap};
use smallvec::SmallVec;

use super::refs::*;
use super::types::*;
use crate::diagnostic::{Diagnostic, DiagnosticSeverity};
use crate::ir_mapping::IrMapping;
use crate::location::Span;
use crate::op_def::{OpName, OpNameArg};
use crate::symbol::Symbol;
use crate::type_def::TypeName;

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

/// The regions of an operation, or a snapshot of them.
pub type RegionList = SmallVec<[RegionRef; 4]>;

/// The blocks of a region or the successors of an operation, or a snapshot
/// of them.
pub type BlockList = SmallVec<[BlockRef; 4]>;

/// The operations of a block, or a snapshot of them.
pub type OpList = SmallVec<[OpRef; 4]>;

/// The successors of an operation.
///
/// Most operations have none, and branches have one or two, so up to two are
/// stored inline. Longer lists live in the context's block pool.
#[derive(Clone)]
pub(crate) enum SuccessorList {
    Inline { len: u8, blocks: [BlockRef; 2] },
    Spilled(EntityList<BlockRef>),
}

impl Default for SuccessorList {
    fn default() -> Self {
        // Slots past `len` are never read.
        Self::Inline {
            len: 0,
            blocks: [BlockRef::new(0); 2],
        }
    }
}

impl SuccessorList {
    fn from_slice(blocks: &[BlockRef], pool: &mut ListPool<BlockRef>) -> Self {
        match *blocks {
            [] => Self::default(),
            [first] => Self::Inline {
                len: 1,
                blocks: [first, first],
            },
            [first, second] => Self::Inline {
                len: 2,
                blocks: [first, second],
            },
            _ => Self::Spilled(EntityList::from_slice(blocks, pool)),
        }
    }

    fn as_slice<'a>(&'a self, pool: &'a ListPool<BlockRef>) -> &'a [BlockRef] {
        match self {
            Self::Inline { len, blocks } => &blocks[..usize::from(*len)],
            Self::Spilled(list) => list.as_slice(pool),
        }
    }

    fn as_mut_slice<'a>(&'a mut self, pool: &'a mut ListPool<BlockRef>) -> &'a mut [BlockRef] {
        match self {
            Self::Inline { len, blocks } => &mut blocks[..usize::from(*len)],
            Self::Spilled(list) => list.as_mut_slice(pool),
        }
    }

    /// Replace the list, returning a spilled list's storage to the pool.
    fn replace(&mut self, blocks: &[BlockRef], pool: &mut ListPool<BlockRef>) {
        if let Self::Spilled(list) = self {
            list.clear(pool);
        }
        *self = Self::from_slice(blocks, pool);
    }
}

/// Data for a single operation in the arena.
#[derive(Clone)]
pub struct OperationData {
    pub location: Location,
    pub name: OpName,
    pub operands: EntityList<ValueRef>,
    pub results: EntityList<TypeRef>,
    pub attributes: AttributeMap,
    /// Head of the operation's region list, linked through
    /// `IrContext::next_region`. Read through [`IrContext::op_regions`] and
    /// change through the context's region methods.
    pub(crate) first_region: Option<RegionRef>,
    /// Read through [`IrContext::op_successors`] and change through the
    /// context's successor methods.
    pub(crate) successors: SuccessorList,
    pub parent_block: Option<BlockRef>,
}

/// Data for a single SSA value.
///
/// The value's type is not stored here — it is derived from the defining
/// operation (`OperationData.results`) or block (`BlockArgData.ty`) via
/// [`IrContext::value_ty`].  This eliminates the possibility of type
/// information going out of sync between a value and its definition site.
#[derive(Clone)]
pub struct ValueData {
    pub def: ValueDef,
}

/// Data for a block argument (type + optional attributes).
#[derive(Clone, Debug)]
pub struct BlockArgData {
    pub ty: TypeRef,
    pub attrs: AttributeMap,
}

/// Data for a basic block.
#[derive(Clone)]
pub struct BlockData {
    pub location: Location,
    pub args: Vec<BlockArgData>,
    pub ops: OpList,
    pub parent_region: Option<RegionRef>,
}

/// Data for a region (list of blocks).
#[derive(Clone)]
pub struct RegionData {
    pub location: Location,
    pub blocks: BlockList,
    pub parent_op: Option<OpRef>,
}

// ============================================================================
// IrContext
// ============================================================================

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Arena-based mutable IR context.
///
/// Owns all IR entities and provides methods for creating, querying,
/// and mutating them. Use-chains are automatically maintained.
pub struct IrContext {
    identity: u64,
    revision: u64,
    ops: PrimaryMap<OpRef, OperationData>,
    values: PrimaryMap<ValueRef, ValueData>,
    blocks: PrimaryMap<BlockRef, BlockData>,
    regions: PrimaryMap<RegionRef, RegionData>,

    /// Use-chain: for each value, the list of operations that use it.
    uses: SecondaryMap<ValueRef, SmallVec<[Use; 2]>>,

    /// Type and path interners.
    types: TypeInterner,
    paths: PathInterner,
    /// Pool for string attribute values.
    strings: StringPool,

    /// Backing pools for EntityList storage.
    value_pool: ListPool<ValueRef>,
    type_pool: ListPool<TypeRef>,
    /// Successor lists longer than [`SuccessorList`] keeps inline.
    block_pool: ListPool<BlockRef>,

    /// The region after each region in its operation's region list.
    next_region: SecondaryMap<RegionRef, Option<RegionRef>>,

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

    /// Entities read while an analysis computation is active.
    reads: ReadRecorder,
    /// The revision of the last change in each anchor's scope.
    anchor_revisions: SecondaryMap<OpRef, u64>,
    /// The revision of the last change whose scope is not classified.
    unclassified_revision: u64,
}

/// One IR entity an analysis computation read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ReadEntity {
    Op(OpRef),
    Block(BlockRef),
    Region(RegionRef),
    Value(ValueRef),
}

/// The anchors whose counters cover a computation's reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReadFootprint {
    /// Some read had no anchor chain.
    WholeContext,
    /// The outermost anchors read; none is an ancestor of another.
    Anchors(SmallVec<[OpRef; 2]>),
}

/// Records reads through `&self` accessors during nested analysis
/// computations. Each computation takes the log segment it started.
#[derive(Default)]
struct ReadRecorder {
    depth: Cell<u32>,
    log: RefCell<Vec<ReadEntity>>,
}

/// Advance an IR revision; a field-level helper so interning can advance it
/// while an interner entry is still borrowed.
fn advance_revision(revision: &mut u64) {
    *revision = revision.checked_add(1).expect("IR revision exhausted");
}

/// Allocate an identity no other context in this process has.
fn next_context_identity() -> u64 {
    NEXT_CONTEXT_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .expect("IrContext identity exhausted")
}

fn primary_with_headroom<K: EntityRef, V: Clone>(map: &PrimaryMap<K, V>) -> PrimaryMap<K, V> {
    let mut copy = PrimaryMap::with_capacity(map.len().next_power_of_two());
    copy.extend(map.values().cloned());
    copy
}

fn secondary_with_capacity<K: EntityRef, V: Clone + Default>(
    map: &SecondaryMap<K, V>,
) -> SecondaryMap<K, V> {
    let mut copy = SecondaryMap::with_capacity(map.capacity());
    for (key, value) in map.iter() {
        copy[key] = value.clone();
    }
    copy
}

/// Copies the whole IR, so a compilation can continue on independent
/// branches from one intermediate result. The cost is proportional to the
/// IR size.
///
/// The clone gets a fresh identity: analyses are cached by identity and
/// revision, and the two contexts diverge from the same revision.
impl Clone for IrContext {
    fn clone(&self) -> Self {
        Self {
            identity: next_context_identity(),
            revision: self.revision,
            ops: self.ops.clone(),
            values: self.values.clone(),
            blocks: self.blocks.clone(),
            regions: self.regions.clone(),
            uses: self.uses.clone(),
            types: self.types.clone(),
            paths: self.paths.clone(),
            strings: self.strings.clone(),
            value_pool: self.value_pool.clone(),
            type_pool: self.type_pool.clone(),
            block_pool: self.block_pool.clone(),
            next_region: self.next_region.clone(),
            result_values: self.result_values.clone(),
            block_arg_values: self.block_arg_values.clone(),
            type_aliases: self.type_aliases.clone(),
            type_alias_by_name: self.type_alias_by_name.clone(),
            type_alias_by_type: self.type_alias_by_type.clone(),
            diagnostics: self.diagnostics.clone(),
            reads: ReadRecorder::default(),
            anchor_revisions: self.anchor_revisions.clone(),
            unclassified_revision: self.unclassified_revision,
        }
    }
}

impl AsRef<StringPool> for IrContext {
    fn as_ref(&self) -> &StringPool {
        &self.strings
    }
}

impl IrContext {
    /// Create a new empty IR context.
    pub fn new() -> Self {
        Self {
            identity: next_context_identity(),
            revision: 0,
            ops: PrimaryMap::new(),
            values: PrimaryMap::new(),
            blocks: PrimaryMap::new(),
            regions: PrimaryMap::new(),
            uses: SecondaryMap::new(),
            types: TypeInterner::new(),
            paths: PathInterner::new(),
            strings: StringPool::new(),
            value_pool: ListPool::new(),
            type_pool: ListPool::new(),
            block_pool: ListPool::new(),
            next_region: SecondaryMap::new(),
            result_values: SecondaryMap::new(),
            block_arg_values: SecondaryMap::new(),
            type_aliases: Vec::new(),
            type_alias_by_name: HashMap::default(),
            type_alias_by_type: HashMap::default(),
            diagnostics: RefCell::new(Vec::new()),
            reads: ReadRecorder::default(),
            anchor_revisions: SecondaryMap::new(),
            unclassified_revision: 0,
        }
    }

    /// Like [`Clone::clone`], but every arena keeps room to grow as it would
    /// after being built by insertion, so the clone's first insertions do not
    /// reallocate.
    pub fn clone_with_headroom(&self) -> Self {
        Self {
            identity: next_context_identity(),
            revision: self.revision,
            ops: primary_with_headroom(&self.ops),
            values: primary_with_headroom(&self.values),
            blocks: primary_with_headroom(&self.blocks),
            regions: primary_with_headroom(&self.regions),
            uses: secondary_with_capacity(&self.uses),
            types: self.types.clone(),
            paths: self.paths.clone(),
            strings: self.strings.clone(),
            value_pool: self.value_pool.clone(),
            type_pool: self.type_pool.clone(),
            block_pool: self.block_pool.clone(),
            next_region: secondary_with_capacity(&self.next_region),
            result_values: secondary_with_capacity(&self.result_values),
            block_arg_values: secondary_with_capacity(&self.block_arg_values),
            type_aliases: self.type_aliases.clone(),
            type_alias_by_name: self.type_alias_by_name.clone(),
            type_alias_by_type: self.type_alias_by_type.clone(),
            diagnostics: self.diagnostics.clone(),
            reads: ReadRecorder::default(),
            anchor_revisions: secondary_with_capacity(&self.anchor_revisions),
            unclassified_revision: self.unclassified_revision,
        }
    }

    pub(crate) fn analysis_stamp(&self) -> (u64, u64) {
        (self.identity, self.revision)
    }

    fn bump_revision(&mut self) {
        advance_revision(&mut self.revision);
    }

    /// The revision of the last change in `anchor`'s scope.
    pub(crate) fn anchor_revision(&self, anchor: OpRef) -> u64 {
        self.anchor_revisions[anchor]
    }

    /// The revision of the last change whose scope is not classified.
    pub(crate) fn unclassified_revision(&self) -> u64 {
        self.unclassified_revision
    }

    /// Advance the revision for a change no anchor chain describes.
    fn bump_unclassified(&mut self) {
        self.bump_revision();
        self.unclassified_revision = self.revision;
    }

    /// Mark every anchor from `op` to its root as changed at the current
    /// revision. `OpRef`s are never reused, so a removed anchor's counter
    /// stays with it.
    ///
    /// An anchor already at the current revision was marked earlier in this
    /// mutation together with every anchor above it, so the walk stops there.
    fn touch_chain(&mut self, op: OpRef) {
        let mut current = Some(op);
        while let Some(op) = current {
            if self.is_anchor(op) {
                if self.anchor_revisions[op] == self.revision {
                    return;
                }
                self.anchor_revisions[op] = self.revision;
            }
            current = self.parent_op(op);
        }
    }

    fn touch_block_owner(&mut self, block: BlockRef) {
        if let Some(owner) = self.block_owner(block) {
            self.touch_chain(owner);
        }
    }

    /// Mark the chain of `value`'s definition, whose use-list changes.
    fn touch_value_def(&mut self, value: ValueRef) {
        match self.values[value].def {
            ValueDef::OpResult(op, _) => self.touch_chain(op),
            ValueDef::BlockArg(block, _) => self.touch_block_owner(block),
        }
    }

    #[inline(always)]
    fn record(&self, entity: ReadEntity) {
        if self.reads.depth.get() != 0 {
            self.record_read(entity);
        }
    }

    #[inline(never)]
    fn record_read(&self, entity: ReadEntity) {
        let mut log = self.reads.log.borrow_mut();
        if log.last() != Some(&entity) {
            log.push(entity);
        }
    }

    /// Start recording reads for one computation; returns its log start.
    pub(crate) fn begin_reads(&self) -> usize {
        self.reads.depth.set(self.reads.depth.get() + 1);
        self.reads.log.borrow().len()
    }

    /// Stop recording the computation that started at `start` and return
    /// its reads, excluding those of computations nested in it.
    pub(crate) fn end_reads(&self, start: usize) -> Vec<ReadEntity> {
        let depth = self.reads.depth.get();
        assert!(depth != 0, "end_reads without begin_reads");
        self.reads.depth.set(depth - 1);
        self.reads.log.borrow_mut().split_off(start)
    }

    fn block_owner(&self, block: BlockRef) -> Option<OpRef> {
        self.blocks[block]
            .parent_region
            .and_then(|region| self.regions[region].parent_op)
    }

    fn parent_op(&self, op: OpRef) -> Option<OpRef> {
        self.ops[op]
            .parent_block
            .and_then(|block| self.block_owner(block))
    }

    fn is_anchor(&self, op: OpRef) -> bool {
        self.ops[op]
            .name
            .def()
            .is_some_and(crate::op_def::OpDef::has_flag::<crate::op_flag::IsolatedFromAbove>)
    }

    /// The innermost anchor at or above `block`'s owner, or `None` when the
    /// block's root is not an anchor.
    fn anchor_of_block(
        &self,
        block: BlockRef,
        memo: &mut HashMap<BlockRef, Option<OpRef>>,
    ) -> Option<OpRef> {
        if let Some(&anchor) = memo.get(&block) {
            return anchor;
        }
        let anchor = match self.block_owner(block) {
            None => None,
            Some(owner) => {
                let above = self.ops[owner]
                    .parent_block
                    .map(|parent| self.anchor_of_block(parent, memo));
                match above {
                    Some(None) => None,
                    _ if self.is_anchor(owner) => Some(owner),
                    Some(above) => above,
                    None => None,
                }
            }
        };
        memo.insert(block, anchor);
        anchor
    }

    /// Map recorded reads to the outermost anchors that cover them. Reads
    /// fields directly, so an enclosing recording computation is unaffected.
    pub(crate) fn read_footprint(&self, reads: &[ReadEntity]) -> ReadFootprint {
        // Operations in one block share the innermost anchor at or above
        // the block's owner, so the walk to the root runs once per block.
        let mut block_anchors: HashMap<BlockRef, Option<OpRef>> = HashMap::default();
        let mut anchors: SmallVec<[OpRef; 2]> = SmallVec::new();
        let mut last_block = None;
        for &entity in reads {
            let (op, block) = match entity {
                ReadEntity::Op(op) => (Some(op), self.ops[op].parent_block),
                ReadEntity::Block(block) => (None, Some(block)),
                ReadEntity::Region(region) => match self.regions[region].parent_op {
                    Some(op) => (Some(op), self.ops[op].parent_block),
                    None => return ReadFootprint::WholeContext,
                },
                ReadEntity::Value(value) => match self.values[value].def {
                    ValueDef::OpResult(op, _) => (Some(op), self.ops[op].parent_block),
                    ValueDef::BlockArg(block, _) => (None, Some(block)),
                },
            };
            if let Some(op) = op
                && self.is_anchor(op)
            {
                match block {
                    Some(block) if self.anchor_of_block(block, &mut block_anchors).is_none() => {
                        return ReadFootprint::WholeContext;
                    }
                    None => {}
                    Some(_) => {}
                }
                if anchors.last() != Some(&op) {
                    anchors.push(op);
                }
                continue;
            }
            let Some(block) = block else {
                return ReadFootprint::WholeContext;
            };
            if last_block == Some(block) {
                continue;
            }
            last_block = Some(block);
            match self.anchor_of_block(block, &mut block_anchors) {
                Some(anchor) => {
                    if anchors.last() != Some(&anchor) {
                        anchors.push(anchor);
                    }
                }
                None => return ReadFootprint::WholeContext,
            }
        }
        anchors.sort_unstable();
        anchors.dedup();
        let outermost = anchors
            .iter()
            .copied()
            .filter(|&anchor| {
                let mut current = anchor;
                while let Some(parent) = self.parent_op(current) {
                    if anchors.binary_search(&parent).is_ok() {
                        return false;
                    }
                    current = parent;
                }
                true
            })
            .collect();
        ReadFootprint::Anchors(outermost)
    }

    /// Read interned types without granting mutable access to the IR.
    pub fn types(&self) -> &TypeInterner {
        &self.types
    }

    /// Read the data for an interned type.
    pub fn get_type(&self, ty: TypeRef) -> &TypeData {
        self.types.get(ty)
    }

    /// Read interned paths without granting mutable access to the IR.
    pub fn paths(&self) -> &PathInterner {
        &self.paths
    }

    /// Intern a type, advancing the revision only for a new entry.
    pub fn intern_type(&mut self, data: TypeData) -> TypeRef {
        debug_assert!(
            data.validate_param_attrs(self).is_ok(),
            "{}: {}",
            data.name.display(self),
            data.validate_param_attrs(self).unwrap_err()
        );
        match self.types.entry(&data) {
            InternEntry::Occupied(ty) => ty,
            InternEntry::Vacant(entry) => {
                advance_revision(&mut self.revision);
                entry.insert(data)
            }
        }
    }

    /// Intern a path, advancing the revision only for a new entry.
    ///
    /// A path already interned is found without allocating; a new path is
    /// copied into the interner once.
    /// The pool holding string attribute values.
    pub fn strings(&self) -> &StringPool {
        &self.strings
    }

    /// Intern a string attribute value, advancing the revision only for a new entry.
    pub fn intern_str(&mut self, text: &str) -> StringRef {
        if let Some(existing) = self.strings.lookup(text) {
            return existing;
        }
        advance_revision(&mut self.revision);
        self.strings.intern(text)
    }

    /// The pooled handle for a builder's string argument.
    pub fn intern_string_arg(&mut self, arg: StringArg) -> StringRef {
        match arg {
            StringArg::Ref(r) => r,
            StringArg::Text(text) => self.intern_str(&text),
        }
    }

    /// Intern the text of `symbol` as a string value.
    ///
    /// For names that reach the IR as symbols, such as frontend declaration
    /// names, but are stored as string attributes.
    pub fn intern_symbol_text(&mut self, symbol: &Symbol) -> StringRef {
        symbol.with_str(|text| self.intern_str(text))
    }

    /// The handle of the operation name written `dialect.name`: its
    /// registered definition, or strings in this context.
    ///
    /// # Panics
    ///
    /// If `text` has no `.` separating a non-empty dialect and name.
    pub fn intern_op_name(&mut self, text: &str) -> OpName {
        OpName::intern(&mut self.strings, text)
    }

    /// The handle of the type name written `dialect.name`; see
    /// [`intern_op_name`](Self::intern_op_name).
    pub fn intern_type_name(&mut self, text: &str) -> TypeName {
        TypeName::intern(&mut self.strings, text)
    }

    /// The handle of the operation name written `dialect.name`, or `None`
    /// if it is neither registered nor used in this context.
    pub fn lookup_op_name(&self, text: &str) -> Option<OpName> {
        OpName::lookup(&self.strings, text)
    }

    /// The handle of the type name written `dialect.name`, or `None` if it
    /// is neither registered nor used in this context.
    pub fn lookup_type_name(&self, text: &str) -> Option<TypeName> {
        TypeName::lookup(&self.strings, text)
    }

    /// The attributes of `op`, read through this context.
    #[inline]
    pub fn op_attrs(&self, op: OpRef) -> AttributeView<'_> {
        self.record(ReadEntity::Op(op));
        self.ops[op].attributes.view(self)
    }

    /// The attributes of type `ty`, read through this context.
    pub fn type_attrs(&self, ty: TypeRef) -> AttributeView<'_> {
        self.types.get(ty).attrs.view(self)
    }

    /// The string pool, for borrowing it beside another part of the context.
    pub(crate) fn strings_mut(&mut self) -> &mut StringPool {
        &mut self.strings
    }

    /// The attributes of `op`, changed through this context.
    pub fn op_attrs_mut(&mut self, op: OpRef) -> AttributeViewMut<'_> {
        self.bump_revision();
        self.touch_chain(op);
        AttributeViewMut::new(&mut self.ops[op].attributes, &mut self.strings)
    }

    /// The attributes of argument `index` of `block`, changed through this
    /// context.
    pub fn block_arg_attrs_mut(&mut self, block: BlockRef, index: usize) -> AttributeViewMut<'_> {
        self.bump_revision();
        self.touch_block_owner(block);
        AttributeViewMut::new(&mut self.blocks[block].args[index].attrs, &mut self.strings)
    }

    /// Find a pooled string without interning it.
    pub fn lookup_str(&self, text: &str) -> Option<StringRef> {
        self.strings.lookup(text)
    }

    /// The text of a pooled string.
    pub fn str(&self, r: StringRef) -> &str {
        self.strings.get(r)
    }

    /// A string attribute holding `text`.
    pub fn string_attr(&mut self, text: &str) -> Attribute {
        Attribute::String(self.intern_str(text))
    }

    pub fn intern_path(&mut self, path: &str) -> PathRef {
        match self.paths.entry(path) {
            InternEntry::Occupied(existing) => existing,
            InternEntry::Vacant(entry) => {
                advance_revision(&mut self.revision);
                entry.insert(path.to_owned())
            }
        }
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
        std::mem::take(self.diagnostics.get_mut())
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
        if self.type_alias_by_name.get(&name) == Some(&ty) {
            return;
        }
        self.bump_unclassified();
        if let Some(old_ty) = self.type_alias_by_name.insert(name.clone(), ty) {
            // Remove old reverse mapping
            self.type_alias_by_type.remove(&old_ty);
            // Update ordered list in-place
            if let Some(entry) = self.type_aliases.iter_mut().find(|(n, _)| *n == name) {
                entry.1 = ty;
            }
        } else {
            self.type_aliases.push((name.clone(), ty));
        }
        self.type_alias_by_type.insert(ty, name);
    }

    /// Look up a type alias by name.
    pub fn type_alias_by_name(&self, name: &Symbol) -> Option<TypeRef> {
        self.type_alias_by_name.get(name).copied()
    }

    /// Look up a type alias by the text of its name, such as a nominal
    /// type's string name.
    pub fn type_alias_by_text(&self, name: &str) -> Option<TypeRef> {
        self.type_alias_by_name(&Symbol::new(name))
    }

    /// Look up an alias name for a given type (reverse lookup for printer).
    pub fn type_alias_by_type(&self, ty: TypeRef) -> Option<Symbol> {
        self.type_alias_by_type.get(&ty).cloned()
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

        let first_region = data.first_region;

        self.bump_revision();
        let op = self.ops.push(data);

        // Back-link owned regions to this operation
        let mut next = first_region;
        while let Some(r) = next {
            if let Some(existing) = self.regions[r].parent_op {
                panic!(
                    "create_op: region {r} already belongs to operation {existing}; \
                     cannot reassign to {op}",
                );
            }
            self.regions[r].parent_op = Some(op);
            next = self.next_region[r];
        }

        // Register operand uses
        for &val in &operand_slice {
            self.touch_value_def(val);
        }
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
    #[inline]
    pub fn op(&self, op: OpRef) -> &OperationData {
        self.record(ReadEntity::Op(op));
        &self.ops[op]
    }

    /// Get mutable reference to operation data.
    ///
    /// **Warning**: Modifying operands directly will desync the use-chain.
    /// Prefer `replace_all_uses` or re-creating the operation.
    pub fn op_mut(&mut self, op: OpRef) -> &mut OperationData {
        self.bump_unclassified();
        &mut self.ops[op]
    }

    /// Get the operands of an operation as a slice.
    #[inline]
    pub fn op_operands(&self, op: OpRef) -> &[ValueRef] {
        self.record(ReadEntity::Op(op));
        self.ops[op].operands.as_slice(&self.value_pool)
    }

    /// Get the result types of an operation as a slice.
    #[inline]
    pub fn op_result_types(&self, op: OpRef) -> &[TypeRef] {
        self.record(ReadEntity::Op(op));
        self.ops[op].results.as_slice(&self.type_pool)
    }

    /// Set the type of the i-th result of an operation.
    pub fn set_op_result_type(&mut self, op: OpRef, index: u32, new_ty: TypeRef) {
        if self.op_result_types(op)[index as usize] == new_ty {
            return;
        }
        self.bump_revision();
        self.touch_chain(op);
        self.ops[op].results.as_mut_slice(&mut self.type_pool)[index as usize] = new_ty;
    }

    /// Set the i-th operand of an operation, updating use-chains.
    pub fn set_op_operand(&mut self, op: OpRef, index: u32, new_val: ValueRef) {
        let old_val = self.ops[op].operands.as_slice(&self.value_pool)[index as usize];
        if old_val == new_val {
            return;
        }
        self.bump_revision();
        self.touch_chain(op);
        self.touch_value_def(old_val);
        self.touch_value_def(new_val);
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
        self.bump_revision();
        self.touch_chain(op);
        self.touch_value_def(val);
        self.ops[op].operands.push(val, &mut self.value_pool);
        self.uses[val].push(Use {
            user: op,
            operand_index: index,
        });
    }

    /// Remove the i-th operand from an operation, updating use-chains.
    pub fn remove_op_operand(&mut self, op: OpRef, index: u32) {
        let old_operands: SmallVec<[ValueRef; 8]> =
            self.ops[op].operands.as_slice(&self.value_pool).into();
        let index_usize = index as usize;
        assert!(
            index_usize < old_operands.len(),
            "remove_op_operand: operand index {index} out of bounds for operation {op}",
        );

        let removed = old_operands[index_usize];
        self.bump_revision();
        self.touch_chain(op);
        // The removed use and every later use change their operand index.
        for &val in &old_operands[index_usize..] {
            self.touch_value_def(val);
        }
        self.uses[removed].retain(|u| !(u.user == op && u.operand_index == index));

        for (old_idx, &val) in old_operands.iter().enumerate().skip(index_usize + 1) {
            for use_entry in &mut self.uses[val] {
                if use_entry.user == op && use_entry.operand_index == old_idx as u32 {
                    use_entry.operand_index -= 1;
                    break;
                }
            }
        }

        let mut operands = EntityList::new();
        for (idx, &val) in old_operands.iter().enumerate() {
            if idx != index_usize {
                operands.push(val, &mut self.value_pool);
            }
        }
        self.ops[op].operands = operands;
    }

    /// Get the i-th result value of an operation.
    #[inline]
    pub fn op_result(&self, op: OpRef, index: u32) -> ValueRef {
        self.record(ReadEntity::Op(op));
        self.result_values[op].as_slice(&self.value_pool)[index as usize]
    }

    /// Get all result values of an operation.
    #[inline]
    pub fn op_results(&self, op: OpRef) -> &[ValueRef] {
        self.record(ReadEntity::Op(op));
        self.result_values[op].as_slice(&self.value_pool)
    }

    /// The regions of an operation, in order.
    #[inline]
    pub fn op_regions(&self, op: OpRef) -> Regions<'_> {
        self.record(ReadEntity::Op(op));
        Regions {
            ctx: self,
            next: self.ops[op].first_region,
        }
    }

    /// The `index`-th region of an operation, if it has that many.
    pub fn op_region(&self, op: OpRef, index: usize) -> Option<RegionRef> {
        self.op_regions(op).nth(index)
    }

    /// Whether an operation has any region, without walking its region list.
    #[inline]
    pub fn op_has_regions(&self, op: OpRef) -> bool {
        self.record(ReadEntity::Op(op));
        self.ops[op].first_region.is_some()
    }

    /// The number of regions of an operation.
    pub fn op_region_count(&self, op: OpRef) -> usize {
        self.op_regions(op).count()
    }

    /// Append `region` to an operation's regions and make the operation its
    /// parent.
    ///
    /// # Panics
    ///
    /// Panics if `region` already belongs to an operation.
    pub fn push_op_region(&mut self, op: OpRef, region: RegionRef) {
        if let Some(existing) = self.regions[region].parent_op {
            panic!(
                "push_op_region: region {region} already belongs to operation {existing}; \
                 cannot reassign to {op}",
            );
        }
        self.bump_revision();
        self.touch_chain(op);
        self.regions[region].parent_op = Some(op);
        self.next_region[region] = None;
        match self.op_regions(op).last() {
            Some(last) => self.next_region[last] = Some(region),
            None => self.ops[op].first_region = Some(region),
        }
    }

    /// Detach every region from an operation, leaving the regions without a
    /// parent.
    pub fn clear_op_regions(&mut self, op: OpRef) {
        self.bump_revision();
        self.touch_chain(op);
        let mut next = self.ops[op].first_region.take();
        while let Some(region) = next {
            next = self.next_region[region].take();
            self.regions[region].parent_op = None;
        }
    }

    /// The successor blocks of an operation, in order.
    #[inline]
    pub fn op_successors(&self, op: OpRef) -> Successors<'_> {
        self.record(ReadEntity::Op(op));
        self.ops[op]
            .successors
            .as_slice(&self.block_pool)
            .iter()
            .copied()
    }

    /// The `index`-th successor of an operation, if it has that many.
    pub fn op_successor(&self, op: OpRef, index: usize) -> Option<BlockRef> {
        self.op_successors(op).nth(index)
    }

    /// Whether an operation has any successor.
    #[inline]
    pub fn op_has_successors(&self, op: OpRef) -> bool {
        self.record(ReadEntity::Op(op));
        !self.ops[op]
            .successors
            .as_slice(&self.block_pool)
            .is_empty()
    }

    /// The number of successors of an operation.
    #[inline]
    pub fn op_successor_count(&self, op: OpRef) -> usize {
        self.record(ReadEntity::Op(op));
        self.ops[op].successors.as_slice(&self.block_pool).len()
    }

    /// Replace the `index`-th successor of an operation.
    ///
    /// # Panics
    ///
    /// Panics if the operation has no `index`-th successor.
    pub fn set_op_successor(&mut self, op: OpRef, index: usize, block: BlockRef) {
        self.bump_revision();
        self.touch_chain(op);
        self.ops[op].successors.as_mut_slice(&mut self.block_pool)[index] = block;
    }

    /// Keep only the first `len` successors of an operation.
    pub fn truncate_op_successors(&mut self, op: OpRef, len: usize) {
        self.bump_revision();
        self.touch_chain(op);
        let kept: BlockList = self.op_successors(op).take(len).collect();
        self.ops[op].successors.replace(&kept, &mut self.block_pool);
    }

    /// Remove an operation and its entire region subtree, clearing all
    /// operand use-chain entries the subtree contributed.
    ///
    /// Does NOT remove it from its parent block. Use `remove_op_from_block` first.
    ///
    /// The op's body regions are cleaned up recursively: every descendant op's
    /// operand use-chain entries are removed, so disposing a region-bearing op
    /// (e.g. `closure.lambda`, `scf.if`) does not leave stale uses on values
    /// defined outside the subtree.
    ///
    /// Only the op's *own* result values are checked for remaining uses. A
    /// descendant result that is still consumed from outside the subtree is the
    /// disposing caller's responsibility to RAUW beforehand; this method does
    /// not enforce it (doing so would turn pre-existing leaks into hard panics
    /// on error-recovery paths).
    ///
    /// # Panics
    ///
    /// Panics if any result value of `op` itself still has uses, as that would
    /// leave dangling references.
    pub fn remove_op(&mut self, op: OpRef) {
        // Refuse to remove an op that is still attached to a block
        assert!(
            self.ops[op].parent_block.is_none(),
            "remove_op: operation {op} is still attached to block {:?}; \
             call remove_op_from_block first",
            self.ops[op].parent_block.unwrap(),
        );

        // Check that the op's own result values have no remaining uses.
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

        self.bump_revision();
        // Clear operand uses for the whole subtree (op + all descendants in its
        // regions), so a disposed region-bearing op leaves no stale uses on
        // values defined outside its body. `walk_op` visits `op` first, then
        // recurses into its regions.
        let mut subtree: Vec<OpRef> = Vec::new();
        let _ = crate::walk::walk_op::<()>(self, op, &mut |o| {
            subtree.push(o);
            std::ops::ControlFlow::Continue(crate::walk::WalkAction::Advance)
        });
        // Mark the subtree's anchors and the definitions whose use-lists
        // lose entries before those entries are cleared.
        for &o in &subtree {
            if self.is_anchor(o) {
                self.anchor_revisions[o] = self.revision;
            }
            let operands: SmallVec<[ValueRef; 8]> =
                self.ops[o].operands.as_slice(&self.value_pool).into();
            for val in operands {
                self.touch_value_def(val);
            }
        }
        for &o in &subtree {
            self.clear_operand_uses_of(o);
        }
    }

    /// Remove every use-chain entry contributed by `op`'s own operands.
    fn clear_operand_uses_of(&mut self, op: OpRef) {
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
    #[inline]
    pub fn value(&self, v: ValueRef) -> &ValueData {
        self.record(ReadEntity::Value(v));
        &self.values[v]
    }

    /// Get the type of a value, derived from its definition site.
    ///
    /// For operation results, this reads from `OperationData.results`.
    /// For block arguments, this reads from `BlockArgData.ty`.
    #[inline]
    pub fn value_ty(&self, v: ValueRef) -> TypeRef {
        self.record(ReadEntity::Value(v));
        match self.values[v].def {
            ValueDef::OpResult(op, idx) => {
                self.ops[op].results.as_slice(&self.type_pool)[idx as usize]
            }
            ValueDef::BlockArg(block, idx) => self.blocks[block].args[idx as usize].ty,
        }
    }

    /// Get the definition of a value.
    #[inline]
    pub fn value_def(&self, v: ValueRef) -> ValueDef {
        self.record(ReadEntity::Value(v));
        self.values[v].def
    }

    // ========================================================================
    // Block
    // ========================================================================

    /// Create a new block and allocate argument values for it.
    pub fn create_block(&mut self, data: BlockData) -> BlockRef {
        let num_args = data.args.len();
        self.bump_revision();
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
    #[inline]
    pub fn block(&self, b: BlockRef) -> &BlockData {
        self.record(ReadEntity::Block(b));
        &self.blocks[b]
    }

    /// Get mutable reference to block data.
    pub fn block_mut(&mut self, b: BlockRef) -> &mut BlockData {
        self.bump_unclassified();
        &mut self.blocks[b]
    }

    /// Get the i-th block argument value.
    #[inline]
    pub fn block_arg(&self, b: BlockRef, index: u32) -> ValueRef {
        self.record(ReadEntity::Block(b));
        self.block_arg_values[b].as_slice(&self.value_pool)[index as usize]
    }

    /// Get all block argument values.
    #[inline]
    pub fn block_args(&self, b: BlockRef) -> &[ValueRef] {
        self.record(ReadEntity::Block(b));
        self.block_arg_values[b].as_slice(&self.value_pool)
    }

    /// Add a new argument to an existing block and return its `ValueRef`.
    pub fn add_block_arg(&mut self, block: BlockRef, arg: BlockArgData) -> ValueRef {
        let index = self.blocks[block].args.len() as u32;
        self.bump_revision();
        self.touch_block_owner(block);
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
        self.bump_revision();
        self.touch_block_owner(block);
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
        if self.blocks[block].args[index as usize].ty == new_ty {
            return;
        }
        self.bump_revision();
        self.touch_block_owner(block);
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
        self.bump_revision();
        self.ops[op].parent_block = Some(block);
        self.blocks[block].ops.push(op);
        // The chain now runs through the new owner and includes `op` itself
        // when it is an anchor.
        self.touch_chain(op);
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
        let pos = self.blocks[block]
            .ops
            .iter()
            .position(|&o| o == before)
            .expect("insert_op_before: `before` op not found in block");
        self.bump_revision();
        self.blocks[block].ops.insert(pos, op);
        self.ops[op].parent_block = Some(block);
        self.touch_chain(op);
    }

    /// Remove an operation from a block (does not destroy the operation).
    ///
    /// Only clears the operation's `parent_block` if it matches the given block.
    pub fn remove_op_from_block(&mut self, block: BlockRef, op: OpRef) {
        self.bump_revision();
        if self.ops[op].parent_block == Some(block) {
            self.touch_chain(op);
        } else {
            self.touch_block_owner(block);
        }
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
        self.bump_revision();
        let region = self.regions.push(data);

        // Set parent_region on all blocks in this region
        let blocks: BlockList = self.regions[region].blocks.clone();
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
        if let Some(parent_op) = self.regions[region].parent_op {
            self.bump_revision();
            self.touch_chain(parent_op);
            self.regions[region].parent_op = None;
            let after = self.next_region[region].take();
            if self.ops[parent_op].first_region == Some(region) {
                self.ops[parent_op].first_region = after;
            } else {
                let before = self
                    .op_regions(parent_op)
                    .find(|&r| self.next_region[r] == Some(region))
                    .expect("detach_region: region is not linked from its parent");
                self.next_region[before] = after;
            }
        }
    }

    /// Get immutable reference to region data.
    #[inline]
    pub fn region(&self, r: RegionRef) -> &RegionData {
        self.record(ReadEntity::Region(r));
        &self.regions[r]
    }

    /// Get mutable reference to region data.
    pub fn region_mut(&mut self, r: RegionRef) -> &mut RegionData {
        self.bump_unclassified();
        &mut self.regions[r]
    }

    // ========================================================================
    // Use-chain
    // ========================================================================

    /// Get all uses of a value.
    #[inline]
    pub fn uses(&self, v: ValueRef) -> &[Use] {
        self.record(ReadEntity::Value(v));
        &self.uses[v]
    }

    /// Check if a value has any uses.
    #[inline]
    pub fn has_uses(&self, v: ValueRef) -> bool {
        self.record(ReadEntity::Value(v));
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
    /// - Nested regions are recursively deep-cloned via [`clone_region`](Self::clone_region).
    /// - Result values are automatically registered in the mapping
    ///   (old results → new results).
    ///
    /// The cloned operation is **not** attached to any block. Use
    /// [`push_op`](Self::push_op) or [`clone_op_into_block`](Self::clone_op_into_block) to insert it.
    pub fn clone_op(&mut self, src_op: OpRef, mapping: &mut IrMapping) -> OpRef {
        // Copy all data from src_op into locals to avoid borrow conflicts.
        let data = &self.ops[src_op];
        let loc = data.location;
        let name = data.name;
        let attrs = data.attributes.clone();
        let operands: SmallVec<[ValueRef; 8]> = data.operands.as_slice(&self.value_pool).into();
        let result_types: SmallVec<[TypeRef; 4]> = data.results.as_slice(&self.type_pool).into();
        let regions: RegionList = self.op_regions(src_op).collect();
        let successors: BlockList = self.op_successors(src_op).collect();

        // Build new operation with remapped operands and successors.
        let mut builder = OperationDataBuilder::new(loc, name);
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
    /// 2. Clone operations in each block using [`clone_op`](Self::clone_op).
    ///
    /// Values not in the mapping (external references) pass through
    /// unchanged.
    pub fn clone_region(&mut self, src_region: RegionRef, mapping: &mut IrMapping) -> RegionRef {
        let loc = self.regions[src_region].location;
        let src_blocks: BlockList = self.regions[src_region].blocks.clone();

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
            let src_ops: OpList = self.blocks[src_block].ops.clone();
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
    /// Convenience wrapper around [`clone_op`](Self::clone_op) + [`push_op`](Self::push_op).
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
        self.bump_revision();
        self.touch_value_def(old);
        self.touch_value_def(new);
        let users: SmallVec<[OpRef; 8]> = self.uses[old].iter().map(|u| u.user).collect();
        for user in users {
            self.touch_chain(user);
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
// Region and successor iterators
// ============================================================================

/// Iterator over the regions of an operation; see [`IrContext::op_regions`].
#[derive(Clone)]
pub struct Regions<'a> {
    ctx: &'a IrContext,
    next: Option<RegionRef>,
}

impl Iterator for Regions<'_> {
    type Item = RegionRef;

    fn next(&mut self) -> Option<RegionRef> {
        let region = self.next?;
        self.next = self.ctx.next_region[region];
        Some(region)
    }
}

impl std::iter::FusedIterator for Regions<'_> {}

/// Iterator over the successors of an operation; see
/// [`IrContext::op_successors`].
pub type Successors<'a> = std::iter::Copied<std::slice::Iter<'a, BlockRef>>;

// ============================================================================
// Helper constructors for OperationData
// ============================================================================

impl OperationData {
    /// Create a new OperationData with the given basic fields.
    /// Operands and results must be added via the pool-backed EntityList.
    pub fn new(location: Location, name: OpName) -> Self {
        Self {
            location,
            name,
            operands: EntityList::new(),
            results: EntityList::new(),
            attributes: AttributeMap::new(),
            first_region: None,
            successors: SuccessorList::default(),
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
    name: OpNameArg,
    operands: Vec<ValueRef>,
    results: Vec<TypeRef>,
    /// Keys are interned when the operation data is built.
    attributes: Vec<(StringArg, Attribute)>,
    regions: RegionList,
    successors: BlockList,
}

impl OperationDataBuilder {
    /// Start an operation named by its definition, handle, or
    /// `dialect.name` text; text is interned when the data is built.
    pub fn new(location: Location, name: impl Into<OpNameArg>) -> Self {
        Self {
            location,
            name: name.into(),
            operands: Vec::new(),
            results: Vec::new(),
            attributes: Vec::new(),
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

    pub fn attr(mut self, key: impl Into<StringArg>, val: impl Into<Attribute>) -> Self {
        self.attributes.push((key.into(), val.into()));
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
        // Link the regions in order. Linking writes their `next_region`
        // entries, so a region that already belongs to an operation (whose
        // list would be corrupted) or appears twice (a cycle) is rejected
        // here rather than in `create_op`.
        for (index, &region) in self.regions.iter().enumerate() {
            if let Some(owner) = ctx.regions[region].parent_op {
                panic!(
                    "OperationDataBuilder: region {region} already belongs to operation {owner}"
                );
            }
            assert!(
                !self.regions[..index].contains(&region),
                "OperationDataBuilder: region {region} appears more than once",
            );
        }
        for pair in self.regions.windows(2) {
            ctx.next_region[pair[0]] = Some(pair[1]);
        }
        if let Some(&last) = self.regions.last() {
            ctx.next_region[last] = None;
        }
        let attributes = AttributeMap::from_entries(ctx, self.attributes);
        OperationData {
            location: self.location,
            name: self.name.resolve(&mut ctx.strings),
            operands,
            results,
            attributes,
            first_region: self.regions.first().copied(),
            successors: SuccessorList::from_slice(&self.successors, &mut ctx.block_pool),
            parent_block: None,
        }
    }
}

#[cfg(test)]
mod prop_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::Span;

    use smallvec::{smallvec, smallvec_inline};

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.intern_path("file:///test.trb");
        Location::new(path, Span::new(0, 0))
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    fn empty_region(ctx: &mut IrContext, loc: Location) -> RegionRef {
        ctx.create_region(RegionData {
            location: loc,
            blocks: SmallVec::new(),
            parent_op: None,
        })
    }

    fn empty_block(ctx: &mut IrContext, loc: Location) -> BlockRef {
        ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: SmallVec::new(),
            parent_region: None,
        })
    }

    fn region_ids(ctx: &IrContext, op: OpRef) -> RegionList {
        ctx.op_regions(op).collect()
    }

    fn successor_ids(ctx: &IrContext, op: OpRef) -> BlockList {
        ctx.op_successors(op).collect()
    }

    #[test]
    fn region_list_keeps_order_through_push_and_detach() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let [a, b, c] = [(); 3].map(|()| empty_region(&mut ctx, loc));
        let data = OperationDataBuilder::new(loc, "test.op")
            .region(a)
            .region(b)
            .build(&mut ctx);
        let op = ctx.create_op(data);
        ctx.push_op_region(op, c);
        assert_eq!(region_ids(&ctx, op), smallvec_inline![a, b, c]);
        assert_eq!(ctx.op_region_count(op), 3);
        assert_eq!(ctx.op_region(op, 2), Some(c));
        assert!(
            [a, b, c]
                .iter()
                .all(|&r| ctx.region(r).parent_op == Some(op))
        );

        // Detaching from the middle, the head, and the tail keeps the rest linked.
        ctx.detach_region(b);
        assert_eq!(region_ids(&ctx, op), smallvec_inline![a, c]);
        ctx.detach_region(a);
        assert_eq!(region_ids(&ctx, op), smallvec_inline![c]);
        ctx.push_op_region(op, b);
        ctx.detach_region(b);
        assert_eq!(region_ids(&ctx, op), smallvec_inline![c]);
        assert_eq!(ctx.region(b).parent_op, None);

        // A detached region can be attached elsewhere.
        let name = ctx.intern_op_name("test.op");
        let other = ctx.create_op(OperationData::new(loc, name));
        ctx.push_op_region(other, a);
        ctx.push_op_region(other, b);
        assert_eq!(region_ids(&ctx, other), smallvec_inline![a, b]);

        ctx.clear_op_regions(op);
        assert!(region_ids(&ctx, op).is_empty());
        assert_eq!(ctx.region(c).parent_op, None);
        assert_eq!(region_ids(&ctx, other), smallvec_inline![a, b]);
    }

    #[test]
    fn cloned_operation_gets_its_own_region_list() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let regions = [(); 3].map(|()| empty_region(&mut ctx, loc));
        let mut builder = OperationDataBuilder::new(loc, "test.op");
        for region in regions {
            builder = builder.region(region);
        }
        let data = builder.build(&mut ctx);
        let op = ctx.create_op(data);
        let clone = ctx.clone_op(op, &mut IrMapping::new());
        assert_eq!(region_ids(&ctx, op)[..], regions);
        let cloned = region_ids(&ctx, clone);
        assert_eq!(cloned.len(), 3);
        assert!(cloned.iter().all(|r| !regions.contains(r)));
        assert!(
            cloned
                .iter()
                .all(|&r| ctx.region(r).parent_op == Some(clone))
        );
    }

    #[test]
    #[should_panic(expected = "appears more than once")]
    fn builder_rejects_a_repeated_region() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let region = empty_region(&mut ctx, loc);
        OperationDataBuilder::new(loc, "test.op")
            .region(region)
            .region(region)
            .build(&mut ctx);
    }

    #[test]
    #[should_panic(expected = "already belongs to operation")]
    fn builder_rejects_an_owned_region() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let region = empty_region(&mut ctx, loc);
        let data = OperationDataBuilder::new(loc, "test.op")
            .region(region)
            .build(&mut ctx);
        ctx.create_op(data);
        OperationDataBuilder::new(loc, "test.op")
            .region(region)
            .build(&mut ctx);
    }

    #[test]
    fn successors_spill_past_two_and_shrink_back() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let blocks = [(); 4].map(|()| empty_block(&mut ctx, loc));
        for len in 0..=blocks.len() {
            let mut builder = OperationDataBuilder::new(loc, "test.br");
            for &block in &blocks[..len] {
                builder = builder.successor(block);
            }
            let data = builder.build(&mut ctx);
            let op = ctx.create_op(data);
            assert_eq!(successor_ids(&ctx, op)[..], blocks[..len]);
            assert_eq!(ctx.op_successor_count(op), len);
            assert_eq!(ctx.op_successors(op).len(), len);
            assert!(
                ctx.op_successors(op)
                    .rev()
                    .eq(blocks[..len].iter().rev().copied())
            );
        }

        let mut builder = OperationDataBuilder::new(loc, "test.br");
        for block in blocks {
            builder = builder.successor(block);
        }
        let data = builder.build(&mut ctx);
        let op = ctx.create_op(data);
        ctx.set_op_successor(op, 3, blocks[0]);
        assert_eq!(
            successor_ids(&ctx, op),
            smallvec_inline![blocks[0], blocks[1], blocks[2], blocks[0]]
        );
        let clone = ctx.clone_op(op, &mut IrMapping::new());
        ctx.truncate_op_successors(op, 1);
        assert_eq!(successor_ids(&ctx, op), smallvec_inline![blocks[0]]);
        assert_eq!(
            successor_ids(&ctx, clone),
            smallvec_inline![blocks[0], blocks[1], blocks[2], blocks[0]]
        );
        ctx.set_op_successor(op, 0, blocks[3]);
        assert_eq!(successor_ids(&ctx, op), smallvec_inline![blocks[3]]);
    }

    #[test]
    fn create_op_and_read_back() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let data = OperationDataBuilder::new(loc, "arith.const")
            .result(i32_ty)
            .attr("value", Attribute::Int(42))
            .build(&mut ctx);

        let op = ctx.create_op(data);

        assert_eq!(
            ctx.op(op).name,
            <crate::dialect::arith::Const as crate::ops::DialectOp>::NAME
        );
        assert_eq!(ctx.op_result_types(op), &[i32_ty]);
        assert_eq!(ctx.op_attrs(op).get("value"), Some(&Attribute::Int(42)));
    }

    #[test]
    fn op_result_values() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = i32_type(&mut ctx);

        let data = OperationDataBuilder::new(loc, "test.multi")
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
                    attrs: AttributeMap::new(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: AttributeMap::new(),
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
        let data = OperationDataBuilder::new(loc, "arith.const")
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
            let data = OperationDataBuilder::new(loc, name)
                .result(i32_ty)
                .build(ctx);
            ctx.create_op(data)
        };

        let op_a = mk_op(&mut ctx, "test.a");
        let op_c = mk_op(&mut ctx, "test.c");
        ctx.push_op(block, op_a);
        ctx.push_op(block, op_c);

        // Insert op_b before op_c
        let op_b = mk_op(&mut ctx, "test.b");
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

        let data = OperationDataBuilder::new(loc, "test.x")
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
        let f64_ty = TypeDataBuilder::new("core.f64").intern(&mut ctx);

        // Create a block with 2 args of type i32
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: i32_ty,
                    attrs: AttributeMap::new(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: AttributeMap::new(),
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
                attrs: AttributeMap::new(),
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
                attrs: AttributeMap::new(),
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
        let f64_ty = TypeDataBuilder::new("core.f64").intern(&mut ctx);

        let data = OperationDataBuilder::new(loc, "test.op")
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
        let f64_ty = TypeDataBuilder::new("core.f64").intern(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: i32_ty,
                    attrs: AttributeMap::new(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: AttributeMap::new(),
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
        let data1 = OperationDataBuilder::new(loc, "test.a")
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(data1);
        let v1 = ctx.op_result(op1, 0);

        // Create op2 that uses op1's result
        let data2 = OperationDataBuilder::new(loc, "test.b")
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
        let data = OperationDataBuilder::new(loc, "func.func")
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
        let name = ctx.intern_op_name("test.x");
        let mut data = OperationData::new(loc, name);
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
        let data1 = OperationDataBuilder::new(loc, "func.func")
            .result(i32_ty)
            .region(region)
            .build(&mut ctx);
        let _op1 = ctx.create_op(data1);

        // Second op tries to take the same region — should panic
        let data2 = OperationDataBuilder::new(loc, "func.func")
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

    // ====================================================================
    // Deep clone tests
    // ====================================================================

    #[test]
    fn cloned_context_is_independent() {
        assert_clone_is_independent(IrContext::clone);
    }

    #[test]
    fn context_cloned_with_headroom_is_independent() {
        assert_clone_is_independent(IrContext::clone_with_headroom);
    }

    fn assert_clone_is_independent(clone_context: fn(&IrContext) -> IrContext) {
        use crate::parser::parse_test_module;
        use crate::printer::print_module;

        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = arith.const {value = 2} : core.i32
    %2 = arith.negi %0 : core.i32
    func.return %2
  }
}"#,
        );
        let original_text = print_module(&ctx, module.op());

        let mut clone = clone_context(&ctx);
        assert_ne!(clone.analysis_stamp().0, ctx.analysis_stamp().0);
        assert_eq!(clone.analysis_stamp().1, ctx.analysis_stamp().1);
        assert_eq!(print_module(&clone, module.op()), original_text);

        // Redirect the negation to the other constant in the clone only.
        let func_body = clone.op_region(module.op(), 0).unwrap();
        let func_op = clone.block(clone.region(func_body).blocks[0]).ops[0];
        let entry = clone.region(clone.op_region(func_op, 0).unwrap()).blocks[0];
        let [first, second, ..] = clone.block(entry).ops[..] else {
            panic!("expected two constants");
        };
        let one = clone.op_results(first)[0];
        let two = clone.op_results(second)[0];
        clone.replace_all_uses(one, two);
        TypeDataBuilder::new("core.i64").intern(&mut clone);

        assert_eq!(print_module(&ctx, module.op()), original_text);
        assert_eq!(ctx.uses(one).len(), 1);
        assert!(ctx.uses(two).is_empty());
        assert!(clone.uses(one).is_empty());
        assert_eq!(clone.uses(two).len(), 1);
        assert_ne!(print_module(&clone, module.op()), original_text);
    }

    /// Helper: parse IR, clone the first func's body region, add the
    /// clone as a new func, and print the whole module.
    fn clone_first_func_region(input: &str, new_name: &'static str) -> String {
        use crate::parser::parse_test_module;
        use crate::printer::print_module;

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        // Find the first func.func in the module's top-level block
        let module_region = ctx.op_region(module.op(), 0).unwrap();
        let module_block = ctx.region(module_region).blocks[0];
        let func_op = ctx
            .block(module_block)
            .ops
            .iter()
            .copied()
            .find(|&op| ctx.op(op).name.is(&ctx, "func.func"))
            .expect("no func.func found");

        // Clone the body region
        let body_region = ctx.op_region(func_op, 0).unwrap();
        let mut mapping = IrMapping::new();
        let cloned_region = ctx.clone_region(body_region, &mut mapping);

        // Build a new func.func with the cloned region
        let func_ty = ctx.op_attrs(func_op).get("type").cloned().unwrap();
        let new_func = OperationDataBuilder::new(ctx.op(func_op).location, "func.func")
            .attr("sym_name", Attribute::String(ctx.intern_str(new_name)))
            .attr("type", func_ty)
            .region(cloned_region)
            .build(&mut ctx);
        let new_func_op = ctx.create_op(new_func);

        // Add to module
        let module_region = ctx.op_region(module.op(), 0).unwrap();
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

        let ext_data = OperationDataBuilder::new(loc, "arith.const")
            .result(i32_ty)
            .build(&mut ctx);
        let ext_op = ctx.create_op(ext_data);
        let ext_val = ctx.op_result(ext_op, 0);

        let op_data = OperationDataBuilder::new(loc, "arith.neg")
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

        let data = OperationDataBuilder::new(loc, "test.multi")
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
