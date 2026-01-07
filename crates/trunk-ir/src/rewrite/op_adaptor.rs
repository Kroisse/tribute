//! Operation adaptor for pattern matching with type information.
//!
//! The `OpAdaptor` provides access to remapped operands and value types
//! during pattern application, delegating to `RewriteContext` internally.
//!
//! When a `TypeConverter` is configured on `PatternApplicator`, `operand_type()`
//! returns already-converted types (e.g., `tribute.int` → `core.i32`).

use crate::{IdVec, Operation, Type, Value};

use super::RewriteContext;

/// Adaptor providing access to remapped operands and type information.
///
/// `OpAdaptor` wraps an operation and its remapped operands, providing
/// convenient access to value types including block arguments. It uses
/// the delegation pattern to hide the internal `RewriteContext`.
///
/// When a `TypeConverter` is provided, `operand_type()` returns converted types
/// automatically (e.g., `tribute.int` → `core.i32`).
///
/// # Example
///
/// ```ignore
/// fn match_and_rewrite<'db>(
///     &self,
///     db: &'db dyn Database,
///     op: &Operation<'db>,
///     adaptor: &OpAdaptor<'db, '_>,
/// ) -> RewriteResult<'db> {
///     // Get remapped operand
///     let operand = adaptor.operand(0)?;
///
///     // Get type of operand (already converted by TypeConverter if configured)
///     let ty = adaptor.operand_type(0);
///
///     // ...
/// }
/// ```
pub struct OpAdaptor<'db, 'ctx> {
    /// The original operation being matched.
    op: Operation<'db>,
    /// Operands with value mappings applied.
    remapped_operands: IdVec<Value<'db>>,
    /// Pre-converted operand types (type conversion already applied).
    operand_types: Vec<Option<Type<'db>>>,
    /// Reference to the rewrite context (private).
    ctx: &'ctx RewriteContext<'db>,
}

impl<'db, 'ctx> OpAdaptor<'db, 'ctx> {
    /// Create a new adaptor for an operation.
    ///
    /// The `operand_types` parameter should contain pre-converted types for each operand.
    /// Use `None` for operands whose type could not be determined.
    pub fn new(
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
        operand_types: Vec<Option<Type<'db>>>,
        ctx: &'ctx RewriteContext<'db>,
    ) -> Self {
        Self {
            op,
            remapped_operands,
            operand_types,
            ctx,
        }
    }

    /// Get the original operation.
    pub fn operation(&self) -> Operation<'db> {
        self.op
    }

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

    /// Get the type of a value, including block arguments.
    ///
    /// This delegates to `RewriteContext::get_value_type`.
    pub fn get_value_type(
        &self,
        db: &'db dyn salsa::Database,
        value: Value<'db>,
    ) -> Option<Type<'db>> {
        self.ctx.get_value_type(db, value)
    }

    /// Look up the final mapped value for a given value.
    ///
    /// This follows the chain of value mappings to get the final value.
    pub fn lookup_value(&self, value: Value<'db>) -> Value<'db> {
        self.ctx.lookup(value)
    }

    /// Get the type of a remapped operand by index (with type conversion applied).
    ///
    /// This returns the pre-converted type that was computed when the adaptor was created.
    /// For example, `tribute.int` → `core.i32` when using the WASM type converter.
    pub fn operand_type(&self, index: usize) -> Option<Type<'db>> {
        self.operand_types.get(index).copied().flatten()
    }
}
