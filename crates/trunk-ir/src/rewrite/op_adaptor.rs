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
/// All type accessors (`operand_type()`, `get_value_type()`, `result_type()`)
/// automatically apply type conversion via the `TypeConverter`.
///
/// # Example
///
/// ```no_run
/// use trunk_ir::Operation;
/// use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
///
/// struct MyPattern;
///
/// impl<'db> RewritePattern<'db> for MyPattern {
///     fn match_and_rewrite(
///         &self,
///         db: &'db dyn salsa::Database,
///         op: &Operation<'db>,
///         adaptor: &OpAdaptor<'db, '_>,
///     ) -> RewriteResult<'db> {
///         // Get remapped operand
///         let operand = adaptor.operand(0).unwrap();
///
///         // Get type of operand (automatically converted)
///         let _ty = adaptor.operand_type(0);
///
///         // Get result type (automatically converted)
///         let _result_ty = adaptor.result_type(db, 0);
///
///         RewriteResult::Unchanged
///     }
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
    /// Reference to the type converter for automatic type conversion.
    type_converter: &'ctx super::TypeConverter,
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
        type_converter: &'ctx super::TypeConverter,
    ) -> Self {
        Self {
            op,
            remapped_operands,
            operand_types,
            ctx,
            type_converter,
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

    /// Get the type of a value with automatic type conversion applied.
    ///
    /// This looks up the type via `RewriteContext::get_value_type` and then
    /// applies type conversion via the `TypeConverter`.
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
    ///
    /// Use this when you need the original type without conversion.
    pub fn get_raw_value_type(
        &self,
        db: &'db dyn salsa::Database,
        value: Value<'db>,
    ) -> Option<Type<'db>> {
        self.ctx.get_value_type(db, value)
    }

    /// Get the result type at the given index with automatic type conversion.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn result_type(&self, db: &'db dyn salsa::Database, index: usize) -> Option<Type<'db>> {
        self.op
            .results(db)
            .get(index)
            .map(|ty| self.type_converter.convert_type(db, *ty).unwrap_or(*ty))
    }

    /// Get all result types with automatic type conversion applied.
    pub fn result_types(&self, db: &'db dyn salsa::Database) -> IdVec<Type<'db>> {
        self.type_converter.convert_types(db, self.op.results(db))
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
