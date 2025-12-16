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

/// Macro to define multiple operations in a dialect.
///
/// This groups multiple operation definitions under a single dialect name,
/// avoiding repetition.
///
/// # Syntax
/// ```ignore
/// dialect! {
///     dialect_name {
///         /// Doc comment
///         pub op name[attrs](operands) -> result {};
///
///         /// Another op
///         pub op other(a, b) -> result {};
///     }
/// }
/// ```
///
/// # Example
/// ```ignore
/// dialect! {
///     arith {
///         /// Constant value operation.
///         pub op constant[value]() -> result {};
///
///         /// Addition operation.
///         pub op add(lhs, rhs) -> result {};
///     }
/// }
/// ```
#[macro_export]
macro_rules! dialect {
    // Recursively process ops - base case (empty)
    (@parse $dialect:ident) => {};

    // With attrs, result, no region
    (@parse $dialect:ident
        $(#[$meta:meta])*
        $vis:vis op $op:ident[$($attr:ident),+ $(,)?]($($operand:ident),* $(,)?) -> result {};
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            $vis op $dialect.$op[$($attr),+]($($operand),*) -> result {}
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // No attrs, fixed operands, result, no region
    (@parse $dialect:ident
        $(#[$meta:meta])*
        $vis:vis op $op:ident($($operand:ident),+ $(,)?) -> result {};
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            $vis op $dialect.$op($($operand),+) -> result {}
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // No attrs, no operands, result, no region
    (@parse $dialect:ident
        $(#[$meta:meta])*
        $vis:vis op $op:ident() -> result {};
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            $vis op $dialect.$op() -> result {}
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // Variadic, no result, no region
    (@parse $dialect:ident
        $(#[$meta:meta])*
        $vis:vis op $op:ident(..$operands:ident) {};
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            $vis op $dialect.$op(..$operands) {}
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // With attrs, no result, with region
    (@parse $dialect:ident
        $(#[$meta:meta])*
        $vis:vis op $op:ident[$($attr:ident),+ $(,)?]() { $region:ident };
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            $vis op $dialect.$op[$($attr),+]() { $region }
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // Minimal (no attrs, no operands, no result, no region)
    (@parse $dialect:ident
        $(#[$meta:meta])*
        $vis:vis op $op:ident() {};
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            $vis op $dialect.$op() {}
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // Main entry point
    ($dialect:ident { $($body:tt)* }) => {
        $crate::dialect!(@parse $dialect $($body)*);
    };
}

/// Macro to define a dialect operation wrapper.
///
/// This generates:
/// - A struct wrapping `Operation<'db>`
/// - `DialectOp` trait implementation
/// - `new` constructor with appropriate parameters
/// - Accessor methods based on the operation signature
///
/// # Syntax
/// ```ignore
/// define_op! {
///     pub op dialect.name[attr1, attr2](operands...) -> result { region }
/// }
/// ```
///
/// - `[attr1, attr2]` - attributes (generates `Attribute<'db>` parameters)
/// - `(lhs, rhs)` - fixed operands (generates `Value<'db>` parameters)
/// - `(..operands)` - variable operands (generates `Vec<Value<'db>>` parameter)
/// - `-> result` - single result (generates `Type` parameter and `result()` method)
/// - `{ body }` - region (generates `Region<'db>` parameter)
/// - `{}` - no region
///
/// # Examples
/// ```ignore
/// define_op! {
///     /// Constant value operation.
///     pub op arith.constant[value]() -> result {}
/// }
///
/// define_op! {
///     /// Addition operation.
///     pub op arith.add(lhs, rhs) -> result {}
/// }
///
/// define_op! {
///     /// Function definition.
///     pub op func.func[sym_name, r#type]() { body }
/// }
/// ```
#[macro_export]
macro_rules! define_op {
    // With attributes, fixed operands, result, and region
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident[$($attr:ident),+ $(,)?]($($operand:ident),* $(,)?) -> result { $($region:ident)? }
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),+],
            operands: [fixed: $($operand),*],
            result: single,
            region: [$($region)?]
        );
    };

    // With attributes, fixed operands, result, no region
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident[$($attr:ident),+ $(,)?]($($operand:ident),* $(,)?) -> result {}
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),+],
            operands: [fixed: $($operand),*],
            result: single,
            region: []
        );
    };

    // With attributes, variadic operands, no result, no region
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident[$($attr:ident),+ $(,)?](..$operands:ident) {}
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),+],
            operands: [variadic: $operands],
            result: none,
            region: []
        );
    };

    // With attributes, no operands, no result, with region
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident[$($attr:ident),+ $(,)?]() { $region:ident }
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),+],
            operands: [fixed:],
            result: none,
            region: [$region]
        );
    };

    // No attributes, fixed operands, result, no region
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident($($operand:ident),* $(,)?) -> result {}
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [],
            operands: [fixed: $($operand),*],
            result: single,
            region: []
        );
    };

    // No attributes, variadic operands, no result, no region
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident(..$operands:ident) {}
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [],
            operands: [variadic: $operands],
            result: none,
            region: []
        );
    };

    // No attributes, no operands, no result, no region (minimal)
    (
        $(#[$meta:meta])*
        $vis:vis op $dialect:ident.$op:ident() {}
    ) => {
        $crate::define_op!(@impl
            meta: [$(#[$meta])*],
            vis: $vis,
            dialect: $dialect,
            op: $op,
            attrs: [],
            operands: [fixed:],
            result: none,
            region: []
        );
    };

    // ========================================================================
    // Implementation rules - Fixed operands with result
    // ========================================================================

    // Fixed operands (1+), single result, no attrs, no region
    (@impl
        meta: [$(#[$meta:meta])*],
        vis: $vis:vis,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [],
        operands: [fixed: $($operand:ident),+],
        result: single,
        region: []
    ) => {
        $crate::paste::paste! {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            $vis struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    $($operand: $crate::Value<'db>,)+
                    result_ty: $crate::Type,
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        .operands(vec![$($operand),+])
                        .result(result_ty)
                        .build();
                    Self::wrap_unchecked(op)
                }

                $crate::define_op!(@gen_accessors [<$op:camel>], [], $($operand),+);

                pub fn result(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
                    self.op.result(db, 0)
                }

                pub fn result_ty(&self, db: &'db dyn salsa::Database) -> $crate::Type {
                    self.op.results(db)[0].clone()
                }
            }

            impl<'db> $crate::DialectOp<'db> for [<$op:camel>]<'db> {
                fn from_operation(
                    db: &'db dyn salsa::Database,
                    op: $crate::Operation<'db>,
                ) -> Result<Self, $crate::ConversionError> {
                    let expected_name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    if op.name(db) != expected_name {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.name(db).to_string(db),
                        });
                    }
                    if op.results(db).is_empty() {
                        return Err($crate::ConversionError::MissingResult);
                    }
                    // Note: operand count validation is relaxed for generic fixed operands
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // Helper: Generate operand accessors with counter
    // Base case: last operand
    (@gen_accessors $struct:ident, [$($counter:tt)*], $name:ident) => {
        /// Get operand at index.
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.operands(db)[$crate::define_op!(@count $($counter)*)]
        }
    };
    // Recursive case: more operands to process
    (@gen_accessors $struct:ident, [$($counter:tt)*], $name:ident, $($rest:ident),+) => {
        /// Get operand at index.
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.operands(db)[$crate::define_op!(@count $($counter)*)]
        }
        $crate::define_op!(@gen_accessors $struct, [$($counter)* _], $($rest),+);
    };

    // Helper: Count tokens
    (@count) => { 0usize };
    (@count $_:tt $($rest:tt)*) => { 1usize + $crate::define_op!(@count $($rest)*) };

    // No operands, single result, with attrs, no region
    (@impl
        meta: [$(#[$meta:meta])*],
        vis: $vis:vis,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),+],
        operands: [fixed:],
        result: single,
        region: []
    ) => {
        $crate::paste::paste! {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            $vis struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    result_ty: $crate::Type,
                    $($attr: $crate::Attribute<'db>,)+
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        .result(result_ty)
                        $(.attr(stringify!($attr), $attr))+
                        .build();
                    Self::wrap_unchecked(op)
                }

                // Auto-generated attribute accessors
                $(
                    /// Get the attribute.
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
                        let key = $crate::Symbol::new(db, stringify!($attr));
                        self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
                    }
                )+

                pub fn result(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
                    self.op.result(db, 0)
                }

                pub fn result_ty(&self, db: &'db dyn salsa::Database) -> $crate::Type {
                    self.op.results(db)[0].clone()
                }
            }

            impl<'db> $crate::DialectOp<'db> for [<$op:camel>]<'db> {
                fn from_operation(
                    db: &'db dyn salsa::Database,
                    op: $crate::Operation<'db>,
                ) -> Result<Self, $crate::ConversionError> {
                    let expected_name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    if op.name(db) != expected_name {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.name(db).to_string(db),
                        });
                    }
                    if op.results(db).is_empty() {
                        return Err($crate::ConversionError::MissingResult);
                    }
                    $(
                        {
                            let key = $crate::Symbol::new(db, stringify!($attr));
                            if !op.attributes(db).contains_key(&key) {
                                return Err($crate::ConversionError::MissingAttribute(stringify!($attr)));
                            }
                        }
                    )+
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // No operands, single result, no attrs, no region
    (@impl
        meta: [$(#[$meta:meta])*],
        vis: $vis:vis,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [],
        operands: [fixed:],
        result: single,
        region: []
    ) => {
        $crate::paste::paste! {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            $vis struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    result_ty: $crate::Type,
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        .result(result_ty)
                        .build();
                    Self::wrap_unchecked(op)
                }

                pub fn result(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
                    self.op.result(db, 0)
                }

                pub fn result_ty(&self, db: &'db dyn salsa::Database) -> $crate::Type {
                    self.op.results(db)[0].clone()
                }
            }

            impl<'db> $crate::DialectOp<'db> for [<$op:camel>]<'db> {
                fn from_operation(
                    db: &'db dyn salsa::Database,
                    op: $crate::Operation<'db>,
                ) -> Result<Self, $crate::ConversionError> {
                    let expected_name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    if op.name(db) != expected_name {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.name(db).to_string(db),
                        });
                    }
                    if op.results(db).is_empty() {
                        return Err($crate::ConversionError::MissingResult);
                    }
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // ========================================================================
    // Implementation rules - Variadic operands
    // ========================================================================

    // Variadic operands, no result, no attrs, no region
    (@impl
        meta: [$(#[$meta:meta])*],
        vis: $vis:vis,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [],
        operands: [variadic: $operands:ident],
        result: none,
        region: []
    ) => {
        $crate::paste::paste! {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            $vis struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    $operands: Vec<$crate::Value<'db>>,
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        .operands($operands)
                        .build();
                    Self::wrap_unchecked(op)
                }

                pub fn operands(&self, db: &'db dyn salsa::Database) -> &[$crate::Value<'db>] {
                    self.op.operands(db)
                }
            }

            impl<'db> $crate::DialectOp<'db> for [<$op:camel>]<'db> {
                fn from_operation(
                    db: &'db dyn salsa::Database,
                    op: $crate::Operation<'db>,
                ) -> Result<Self, $crate::ConversionError> {
                    let expected_name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    if op.name(db) != expected_name {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.name(db).to_string(db),
                        });
                    }
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // ========================================================================
    // Implementation rules - Region operations
    // ========================================================================

    // No operands, no result, with attrs, with region
    (@impl
        meta: [$(#[$meta:meta])*],
        vis: $vis:vis,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),+],
        operands: [fixed:],
        result: none,
        region: [$region:ident]
    ) => {
        $crate::paste::paste! {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            $vis struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    $($attr: $crate::Attribute<'db>,)+
                    $region: $crate::Region<'db>,
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        $(.attr(stringify!($attr), $attr))+
                        .region($region)
                        .build();
                    Self::wrap_unchecked(op)
                }

                // Auto-generated attribute accessors
                $(
                    /// Get the attribute.
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
                        let key = $crate::Symbol::new(db, stringify!($attr));
                        self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
                    }
                )+

                pub fn $region(&self, db: &'db dyn salsa::Database) -> $crate::Region<'db> {
                    self.op.regions(db)[0]
                }
            }

            impl<'db> $crate::DialectOp<'db> for [<$op:camel>]<'db> {
                fn from_operation(
                    db: &'db dyn salsa::Database,
                    op: $crate::Operation<'db>,
                ) -> Result<Self, $crate::ConversionError> {
                    let expected_name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    if op.name(db) != expected_name {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.name(db).to_string(db),
                        });
                    }
                    $(
                        {
                            let key = $crate::Symbol::new(db, stringify!($attr));
                            if !op.attributes(db).contains_key(&key) {
                                return Err($crate::ConversionError::MissingAttribute(stringify!($attr)));
                            }
                        }
                    )+
                    if op.regions(db).is_empty() {
                        return Err($crate::ConversionError::MissingRegion);
                    }
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // ========================================================================
    // Implementation rules - Minimal (no operands, no result, no attrs, no region)
    // ========================================================================

    (@impl
        meta: [$(#[$meta:meta])*],
        vis: $vis:vis,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [],
        operands: [fixed:],
        result: none,
        region: []
    ) => {
        $crate::paste::paste! {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            $vis struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name).build();
                    Self::wrap_unchecked(op)
                }
            }

            impl<'db> $crate::DialectOp<'db> for [<$op:camel>]<'db> {
                fn from_operation(
                    db: &'db dyn salsa::Database,
                    op: $crate::Operation<'db>,
                ) -> Result<Self, $crate::ConversionError> {
                    let expected_name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    if op.name(db) != expected_name {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.name(db).to_string(db),
                        });
                    }
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };
}
