//! Type conversion infrastructure for dialect lowering.
//!
//! Provides MLIR-style type conversion with support for:
//! - Type conversion registration via closures
//! - Materialization for generating conversion IR

use smallvec::SmallVec;

use crate::{Location, Operation, Type, Value};

/// Stack-optimized vector for materialization operations.
/// Most materializations produce 0-2 operations; the array is sized
/// to 4 for headroom to avoid heap allocation in edge cases.
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
/// ```
/// use trunk_ir::dialect::core;
/// use trunk_ir::rewrite::{MaterializeResult, TypeConverter};
/// use trunk_ir::DialectType;
///
/// let converter = TypeConverter::new()
///     .add_conversion(|db, ty| {
///         // Convert i64 to i32
///         if core::I64::from_type(db, ty).is_some() {
///             Some(core::I32::new(db).as_type())
///         } else {
///             None
///         }
///     })
///     .add_materialization(|_db, _loc, _value, _from_ty, _to_ty| {
///         // For this example, skip materialization
///         MaterializeResult::Skip
///     });
///
/// let db = salsa::DatabaseImpl::default();
/// let i64_ty = core::I64::new(&db).as_type();
/// let result = converter.convert_type(&db, i64_ty);
/// assert!(result.is_some());
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
    use super::*;
    use crate::DialectType;
    use crate::dialect::core;
    use salsa_test_macros::salsa_test;

    #[salsa_test]
    fn test_empty_converter_returns_none(db: &salsa::DatabaseImpl) {
        let converter = TypeConverter::new();
        let i32_ty = core::I32::new(db).as_type();

        // Empty converter should not convert any types
        assert!(converter.convert_type(db, i32_ty).is_none());
        assert!(converter.is_legal(db, i32_ty));
    }

    #[salsa_test]
    fn test_add_conversion(db: &salsa::DatabaseImpl) {
        let converter = TypeConverter::new().add_conversion(|db, ty| {
            // Convert i32 → i64
            core::I32::from_type(db, ty).map(|_| core::I64::new(db).as_type())
        });

        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        // i32 should convert to i64
        let result = converter.convert_type(db, i32_ty);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), i64_ty);

        // i64 should not be converted (no matching converter)
        assert!(converter.convert_type(db, i64_ty).is_none());
    }

    #[salsa_test]
    fn test_conversion_order(db: &salsa::DatabaseImpl) {
        // First converter wins
        let converter = TypeConverter::new()
            .add_conversion(|db, ty| {
                core::I32::from_type(db, ty).map(|_| core::I64::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                // This should never be reached for i32
                core::I32::from_type(db, ty).map(|_| core::F64::new(db).as_type())
            });

        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        // First converter should win
        let result = converter.convert_type(db, i32_ty);
        assert_eq!(result.unwrap(), i64_ty);
    }

    #[test]
    fn test_materialize_result_helpers() {
        // Test MaterializeResult helper methods without needing a database
        let result = MaterializeResult::<'static>::NoOp;
        assert!(matches!(result, MaterializeResult::NoOp));

        let skip = MaterializeResult::<'static>::Skip;
        assert!(matches!(skip, MaterializeResult::Skip));
    }

    #[salsa_test]
    fn test_multiple_conversions(db: &salsa::DatabaseImpl) {
        // Test chaining multiple conversions
        let converter = TypeConverter::new()
            .add_conversion(|db, ty| {
                // Convert i32 → i64
                core::I32::from_type(db, ty).map(|_| core::I64::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                // Convert f32 → f64
                core::F32::from_type(db, ty).map(|_| core::F64::new(db).as_type())
            });

        let i32_ty = core::I32::new(db).as_type();
        let f32_ty = core::F32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let f64_ty = core::F64::new(db).as_type();

        // Both conversions should work
        assert_eq!(converter.convert_type(db, i32_ty), Some(i64_ty));
        assert_eq!(converter.convert_type(db, f32_ty), Some(f64_ty));

        // Types not registered should return None
        assert!(converter.convert_type(db, i64_ty).is_none());
    }

    #[test]
    fn test_default_converter() {
        // Test Default implementation
        let converter = TypeConverter::default();
        assert!(converter.conversions.is_empty());
        assert!(converter.materializations.is_empty());
    }
}
