//! Dialect operation utilities.

use crate::Operation;

/// Error when converting an Operation to a dialect-specific type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionError {
    /// Operation name doesn't match expected dialect.operation.
    WrongOperation {
        expected: &'static str,
        actual: String,
    },
    /// Missing required attribute.
    MissingAttribute(&'static str),
    /// Attribute has wrong type.
    WrongAttributeType(&'static str),
    /// Missing result type.
    MissingResult,
    /// Missing region.
    MissingRegion,
    /// Wrong number of operands.
    WrongOperandCount { expected: usize, actual: usize },
}

/// Trait for dialect operation wrappers.
pub trait DialectOp<'db>: Sized + Copy {
    /// Try to wrap an Operation as this dialect op type.
    fn from_operation(
        db: &'db dyn salsa::Database,
        op: Operation<'db>,
    ) -> Result<Self, ConversionError>;

    /// Get the underlying Operation.
    fn as_operation(&self) -> Operation<'db>;
}

/// Macro to define a dialect operation wrapper.
///
/// This generates:
/// - A struct wrapping `Operation<'db>`
/// - `DialectOp` trait implementation
///
/// # Validation rules
/// - `single_result` - validates has result AND generates `result()` method
/// - `has_result` - validates has at least one result
/// - `has_region` - validates has at least one region
/// - `has_attr("name")` - validates has attribute with given name
/// - `operand_count(n)` - validates has exactly n operands
///
/// # Example
/// ```ignore
/// define_dialect_op! {
///     /// `arith.const` operation: produces a constant value.
///     pub struct Const("arith", "const") {
///         single_result,
///         has_attr("value"),
///     }
/// }
/// ```
#[macro_export]
macro_rules! define_dialect_op {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident($dialect:literal, $op:literal) {
            $($validation:tt)*
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy)]
        $vis struct $name<'db> {
            op: $crate::Operation<'db>,
        }

        impl<'db> $name<'db> {
            /// Wrap an operation without validation.
            pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                Self { op }
            }

            $crate::define_dialect_op!(@methods $($validation)*);
        }

        impl<'db> $crate::DialectOp<'db> for $name<'db> {
            fn from_operation(
                db: &'db dyn salsa::Database,
                op: $crate::Operation<'db>,
            ) -> Result<Self, $crate::ConversionError> {
                let expected_name = $crate::OpNameId::new(db, $dialect, $op);
                if op.name(db) != expected_name {
                    return Err($crate::ConversionError::WrongOperation {
                        expected: concat!($dialect, ".", $op),
                        actual: op.name(db).to_string(db),
                    });
                }

                $crate::define_dialect_op!(@validate op, db, $($validation)*);

                Ok(Self { op })
            }

            fn as_operation(&self) -> $crate::Operation<'db> {
                self.op
            }
        }
    };

    // Method generation rules
    (@methods ) => {};

    (@methods single_result $(, $($rest:tt)*)?) => {
        /// Get the single result value.
        pub fn result(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.result(db, 0)
        }

        /// Get the result type.
        pub fn result_ty(&self, db: &'db dyn salsa::Database) -> $crate::Type {
            self.op.results(db)[0].clone()
        }
        $($crate::define_dialect_op!(@methods $($rest)*);)?
    };

    (@methods has_result $(, $($rest:tt)*)?) => {
        $($crate::define_dialect_op!(@methods $($rest)*);)?
    };

    (@methods has_region $(, $($rest:tt)*)?) => {
        $($crate::define_dialect_op!(@methods $($rest)*);)?
    };

    (@methods has_attr($attr:literal) $(, $($rest:tt)*)?) => {
        $($crate::define_dialect_op!(@methods $($rest)*);)?
    };

    (@methods operand_count($n:expr) $(, $($rest:tt)*)?) => {
        $($crate::define_dialect_op!(@methods $($rest)*);)?
    };

    // Validation rules
    (@validate $op:ident, $db:ident, ) => {};

    (@validate $op:ident, $db:ident, single_result $(, $($rest:tt)*)?) => {
        if $op.results($db).is_empty() {
            return Err($crate::ConversionError::MissingResult);
        }
        $($crate::define_dialect_op!(@validate $op, $db, $($rest)*);)?
    };

    (@validate $op:ident, $db:ident, has_result $(, $($rest:tt)*)?) => {
        if $op.results($db).is_empty() {
            return Err($crate::ConversionError::MissingResult);
        }
        $($crate::define_dialect_op!(@validate $op, $db, $($rest)*);)?
    };

    (@validate $op:ident, $db:ident, has_region $(, $($rest:tt)*)?) => {
        if $op.regions($db).is_empty() {
            return Err($crate::ConversionError::MissingRegion);
        }
        $($crate::define_dialect_op!(@validate $op, $db, $($rest)*);)?
    };

    (@validate $op:ident, $db:ident, has_attr($attr:literal) $(, $($rest:tt)*)?) => {
        {
            let key = $crate::Symbol::new($db, $attr);
            if !$op.attributes($db).contains_key(&key) {
                return Err($crate::ConversionError::MissingAttribute($attr));
            }
        }
        $($crate::define_dialect_op!(@validate $op, $db, $($rest)*);)?
    };

    (@validate $op:ident, $db:ident, operand_count($n:expr) $(, $($rest:tt)*)?) => {
        if $op.operands($db).len() != $n {
            return Err($crate::ConversionError::WrongOperandCount {
                expected: $n,
                actual: $op.operands($db).len(),
            });
        }
        $($crate::define_dialect_op!(@validate $op, $db, $($rest)*);)?
    };
}
