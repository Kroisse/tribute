//! PatternRewriter — unified mutation + operand access interface.
//!
//! The `PatternRewriter` is the single interface exposed to patterns for
//! both reading (remapped operands, converted types) and writing (mutations).
//!
//! Patterns receive the **remapped** operation (intra-iteration operand remappings
//! applied) so that `op.modify(db).build()` preserves up-to-date operands.
//! Use the rewriter for type-converted operand access:
//! - `rewriter.operand(i)` — remap+cast applied operand
//! - `rewriter.replace_op(new_op)` — replace the current operation
//! - `rewriter.insert_op(op)` — insert before the replacement
//! - `rewriter.add_module_op(op)` — add a top-level function to the module

use crate::{IdVec, Location, Operation, Type, Value};

use super::context::RewriteContext;
use super::type_converter::{MaterializeResult, TypeConverter};

/// Accumulated mutations from a pattern rewrite.
pub(crate) struct Mutations<'db> {
    /// Operations to insert before the replacement.
    pub(crate) prefix_ops: Vec<Operation<'db>>,
    /// The replacement operation (if any).
    pub(crate) replacement: Option<Operation<'db>>,
    /// If set, the operation is erased and its results mapped to these values.
    pub(crate) erase_values: Option<Vec<Value<'db>>>,
    /// Operations to add at module level (e.g., outlined functions).
    pub(crate) module_ops: Vec<Operation<'db>>,
}

/// Unified interface for pattern rewriting.
///
/// Provides access to remapped+casted operands and accumulates mutations.
/// Patterns receive this as `&mut PatternRewriter` alongside the original operation.
///
/// # Operand Access
///
/// Use `rewriter.operand(i)` to get the remapped and cast-applied operand.
/// Do **not** use `op.operands(db)` — those are the original (possibly stale) operands.
///
/// # Mutations
///
/// - [`insert_op`](Self::insert_op) — insert an operation before the replacement
/// - [`replace_op`](Self::replace_op) — replace the current operation
/// - [`erase_op`](Self::erase_op) — erase the operation, mapping results to given values
/// - [`add_module_op`](Self::add_module_op) — add a top-level operation to the module
pub struct PatternRewriter<'db, 'ctx> {
    /// Remap + cast applied operands.
    remapped_operands: IdVec<Value<'db>>,
    /// Pre-converted operand types.
    operand_types: Vec<Option<Type<'db>>>,
    /// Reference to the rewrite context (value lookup).
    ctx: &'ctx RewriteContext<'db>,
    /// Reference to the type converter.
    type_converter: &'ctx TypeConverter,

    // === Accumulated mutations ===
    prefix_ops: Vec<Operation<'db>>,
    replacement: Option<Operation<'db>>,
    erase_values: Option<Vec<Value<'db>>>,
    module_ops: Vec<Operation<'db>>,
}

impl<'db, 'ctx> PatternRewriter<'db, 'ctx> {
    /// Create a new rewriter. Called internally by the applicator.
    pub(crate) fn new(
        remapped_operands: IdVec<Value<'db>>,
        operand_types: Vec<Option<Type<'db>>>,
        ctx: &'ctx RewriteContext<'db>,
        type_converter: &'ctx TypeConverter,
    ) -> Self {
        Self {
            remapped_operands,
            operand_types,
            ctx,
            type_converter,
            prefix_ops: Vec::new(),
            replacement: None,
            erase_values: None,
            module_ops: Vec::new(),
        }
    }

    // === Operand access (remap+cast applied) ===

    /// Get all remapped operands.
    pub fn operands(&self) -> &IdVec<Value<'db>> {
        &self.remapped_operands
    }

    /// Get a specific remapped operand by index.
    pub fn operand(&self, index: usize) -> Option<Value<'db>> {
        self.remapped_operands.get(index).copied()
    }

    /// Get the number of operands.
    pub fn num_operands(&self) -> usize {
        self.remapped_operands.len()
    }

    // === Type access (type conversion applied) ===

    /// Get the type of a remapped operand by index (with type conversion applied).
    pub fn operand_type(&self, index: usize) -> Option<Type<'db>> {
        self.operand_types.get(index).copied().flatten()
    }

    /// Get the result type at the given index with automatic type conversion.
    ///
    /// Reads from the **original** operation's result types and applies conversion.
    pub fn result_type(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        index: usize,
    ) -> Option<Type<'db>> {
        op.results(db)
            .get(index)
            .map(|ty| self.type_converter.convert_type(db, *ty).unwrap_or(*ty))
    }

    /// Get all result types with automatic type conversion applied.
    pub fn result_types(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> IdVec<Type<'db>> {
        self.type_converter.convert_types(db, op.results(db))
    }

    /// Get the type of a value with automatic type conversion applied.
    pub fn get_value_type(
        &self,
        db: &'db dyn salsa::Database,
        value: Value<'db>,
    ) -> Option<Type<'db>> {
        let raw_ty = self.ctx.get_value_type(db, value)?;
        Some(
            self.type_converter
                .convert_type(db, raw_ty)
                .unwrap_or(raw_ty),
        )
    }

    /// Get the raw (unconverted) type of a value.
    pub fn get_raw_value_type(
        &self,
        db: &'db dyn salsa::Database,
        value: Value<'db>,
    ) -> Option<Type<'db>> {
        self.ctx.get_value_type(db, value)
    }

    // === Value resolution ===

    /// Look up the final mapped value for a given value.
    pub fn lookup_value(&self, value: Value<'db>) -> Value<'db> {
        self.ctx.lookup(value)
    }

    /// Materialize a value conversion from one type to another.
    pub fn materialize(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: Value<'db>,
        from_ty: Type<'db>,
        to_ty: Type<'db>,
    ) -> Option<MaterializeResult<'db>> {
        self.type_converter
            .materialize(db, location, value, from_ty, to_ty)
    }

    /// Get a reference to the type converter.
    pub fn type_converter(&self) -> &'ctx TypeConverter {
        self.type_converter
    }

    // === Mutations ===

    /// Insert an operation before the replacement.
    ///
    /// Multiple calls accumulate operations in order.
    pub fn insert_op(&mut self, op: Operation<'db>) {
        self.prefix_ops.push(op);
    }

    /// Replace the current operation with a new one.
    ///
    /// The new operation's results are mapped 1:1 to the original's results.
    /// Can only be called once per rewrite.
    pub fn replace_op(&mut self, new_op: Operation<'db>) {
        debug_assert!(
            self.replacement.is_none() && self.erase_values.is_none(),
            "replace_op called after replace_op or erase_op"
        );
        self.replacement = Some(new_op);
    }

    /// Erase the current operation, mapping its results to the given values.
    ///
    /// The replacement values must match the original result count.
    pub fn erase_op(&mut self, replacement_values: Vec<Value<'db>>) {
        debug_assert!(
            self.replacement.is_none() && self.erase_values.is_none(),
            "erase_op called after replace_op or erase_op"
        );
        self.erase_values = Some(replacement_values);
    }

    /// Add an operation at module level (e.g., an outlined function).
    ///
    /// The operation will be added to the module's first block after
    /// the current iteration completes. All value references within the
    /// operation will be remapped using the full value map at that point.
    pub fn add_module_op(&mut self, op: Operation<'db>) {
        self.module_ops.push(op);
    }

    // === Internal ===

    /// Check if any mutation was recorded.
    pub(crate) fn has_mutations(&self) -> bool {
        !self.prefix_ops.is_empty()
            || self.replacement.is_some()
            || self.erase_values.is_some()
            || !self.module_ops.is_empty()
    }

    /// Consume the rewriter and return accumulated mutations.
    pub(crate) fn take_mutations(self) -> Mutations<'db> {
        Mutations {
            prefix_ops: self.prefix_ops,
            replacement: self.replacement,
            erase_values: self.erase_values,
            module_ops: self.module_ops,
        }
    }
}
