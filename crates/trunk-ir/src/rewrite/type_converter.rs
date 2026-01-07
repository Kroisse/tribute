//! Type conversion infrastructure for dialect lowering.
//!
//! Provides MLIR-style type conversion with support for:
//! - Type conversion registration via closures
//! - Materialization for generating conversion IR

use smallvec::SmallVec;

use crate::{Location, Operation, Type, Value};

/// Stack-optimized vector for materialization operations.
/// Most materializations produce 0-2 operations.
pub type OpVec<'db> = SmallVec<[Operation<'db>; 4]>;

/// Result of a materialization attempt.
#[derive(Debug, Clone)]
pub enum MaterializeResult<'db> {
    /// This materializer does not handle this conversion - try the next one.
    Skip,
    /// Conversion handled, no IR operations needed (type-only change).
    NoOp,
    /// Conversion handled, insert these operations.
    Ops(OpVec<'db>),
}

impl<'db> MaterializeResult<'db> {
    /// Create a result with a single operation.
    pub fn single(op: Operation<'db>) -> Self {
        let mut v = SmallVec::new();
        v.push(op);
        MaterializeResult::Ops(v)
    }

    /// Create a result with multiple operations.
    pub fn ops(ops: impl IntoIterator<Item = Operation<'db>>) -> Self {
        MaterializeResult::Ops(ops.into_iter().collect())
    }
}

/// Type conversion function trait using higher-ranked trait bounds.
/// The closure works with any `'db` lifetime.
pub trait TypeConversionFn:
    for<'db> Fn(&'db dyn salsa::Database, Type<'db>) -> Option<Type<'db>> + Send + Sync
{
}

impl<F> TypeConversionFn for F where
    F: for<'db> Fn(&'db dyn salsa::Database, Type<'db>) -> Option<Type<'db>> + Send + Sync
{
}

/// Materialization function trait using higher-ranked trait bounds.
pub trait MaterializeFn:
    for<'db> Fn(
        &'db dyn salsa::Database,
        Location<'db>,
        Value<'db>,
        Type<'db>,
        Type<'db>,
    ) -> MaterializeResult<'db>
    + Send
    + Sync
{
}

impl<F> MaterializeFn for F where
    F: for<'db> Fn(
            &'db dyn salsa::Database,
            Location<'db>,
            Value<'db>,
            Type<'db>,
            Type<'db>,
        ) -> MaterializeResult<'db>
        + Send
        + Sync
{
}

/// Type converter for dialect lowering.
///
/// Manages type conversions and materializations during IR transformation.
/// Conversions are registered via closures. The converter is `'static` and
/// can be stored in patterns or shared across passes.
///
/// # Example
///
/// ```ignore
/// let converter = TypeConverter::new()
///     .add_conversion(|db, ty| {
///         if is_high_level_type(db, ty) {
///             Some(low_level_type(db))
///         } else {
///             None
///         }
///     })
///     .add_materialization(|db, loc, value, from_ty, to_ty| {
///         if needs_cast(from_ty, to_ty) {
///             MaterializeResult::single(create_cast(db, loc, value, to_ty))
///         } else {
///             MaterializeResult::Skip
///         }
///     });
/// ```
pub struct TypeConverter {
    conversions: Vec<Box<dyn TypeConversionFn>>,
    materializations: Vec<Box<dyn MaterializeFn>>,
}

impl TypeConverter {
    /// Create a new empty type converter.
    pub fn new() -> Self {
        Self {
            conversions: Vec::new(),
            materializations: Vec::new(),
        }
    }

    /// Register a type conversion function.
    ///
    /// Conversion functions are tried in registration order.
    /// Return `Some(converted_type)` if this function handles the type,
    /// or `None` to try the next converter.
    pub fn add_conversion<F>(mut self, f: F) -> Self
    where
        F: TypeConversionFn + 'static,
    {
        self.conversions.push(Box::new(f));
        self
    }

    /// Register a materialization function.
    ///
    /// Materialization functions generate IR operations to convert values
    /// from one type to another at runtime. They are tried in registration order.
    ///
    /// Return:
    /// - `MaterializeResult::Skip` to try the next materializer
    /// - `MaterializeResult::NoOp` if no IR is needed
    /// - `MaterializeResult::Ops(ops)` to insert the given operations
    pub fn add_materialization<M>(mut self, m: M) -> Self
    where
        M: MaterializeFn + 'static,
    {
        self.materializations.push(Box::new(m));
        self
    }

    /// Convert a type using registered converters.
    ///
    /// Returns `Some(converted_type)` if the type should be converted,
    /// or `None` if the type is already legal (no conversion needed).
    pub fn convert_type<'db>(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
    ) -> Option<Type<'db>> {
        self.conversions.iter().find_map(|f| f(db, ty))
    }

    /// Check if a type is legal (needs no conversion).
    pub fn is_legal<'db>(&self, db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
        self.convert_type(db, ty).is_none()
    }

    /// Materialize a value conversion.
    ///
    /// Generates IR operations to convert a value from `from_ty` to `to_ty`.
    /// Tries registered materializers in order until one handles the conversion.
    ///
    /// Returns `None` if no materializer handles this conversion.
    pub fn materialize<'db>(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: Value<'db>,
        from_ty: Type<'db>,
        to_ty: Type<'db>,
    ) -> Option<MaterializeResult<'db>> {
        for m in &self.materializations {
            let result = m(db, location, value, from_ty, to_ty);
            match result {
                MaterializeResult::Skip => continue,
                _ => return Some(result),
            }
        }
        None
    }
}

impl Default for TypeConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    // Basic tests would go here, but they require a database setup
    // which is complex for unit tests. Integration tests are preferred.
}
