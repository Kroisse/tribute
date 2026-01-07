//! Operation adaptor for pattern matching with type information.
//!
//! The `OpAdaptor` provides access to remapped operands and value types
//! during pattern application, delegating to `RewriteContext` internally.

use crate::{IdVec, Operation, Type, Value};

use super::RewriteContext;

/// Adaptor providing access to remapped operands and type information.
///
/// `OpAdaptor` wraps an operation and its remapped operands, providing
/// convenient access to value types including block arguments. It uses
/// the delegation pattern to hide the internal `RewriteContext`.
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
///     // Get type of any value (including block args)
///     let ty = adaptor.get_value_type(db, operand);
///
///     // ...
/// }
/// ```
pub struct OpAdaptor<'db, 'ctx> {
    /// The original operation being matched.
    op: Operation<'db>,
    /// Operands with value mappings applied.
    remapped_operands: IdVec<Value<'db>>,
    /// Reference to the rewrite context (private).
    ctx: &'ctx RewriteContext<'db>,
}

impl<'db, 'ctx> OpAdaptor<'db, 'ctx> {
    /// Create a new adaptor for an operation.
    pub fn new(
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
        ctx: &'ctx RewriteContext<'db>,
    ) -> Self {
        Self {
            op,
            remapped_operands,
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

    /// Get the type of a remapped operand by index.
    ///
    /// Convenience method combining `operand()` and `get_value_type()`.
    pub fn operand_type(&self, db: &'db dyn salsa::Database, index: usize) -> Option<Type<'db>> {
        self.operand(index).and_then(|v| self.get_value_type(db, v))
    }
}
