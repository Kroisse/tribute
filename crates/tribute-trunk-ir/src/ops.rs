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
///         op name[attrs](operands) -> result {};
///
///         /// Another op
///         op other(a, b) -> result {};
///     }
/// }
/// ```
///
/// # Example
/// ```ignore
/// dialect! {
///     arith {
///         /// Constant value operation.
///         op constant[value]() -> result {};
///
///         /// Addition operation.
///         op add(lhs, rhs) -> result {};
///     }
/// }
/// ```
#[macro_export]
macro_rules! dialect {
    // Base case: no more ops to process
    (@parse $dialect:ident) => {};

    // Unified pattern: optional attrs, optional result
    (@parse $dialect:ident
        $(#[$meta:meta])*
        op $op:ident $([$($attrs:tt)*])? ($($operands:tt)*) $(-> $result:ident)? { $($region:tt)* };
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            op $dialect.$op $([$($attrs)*])? ($($operands)*) $(-> $result)? { $($region)* }
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
///     op dialect.name[attr1, attr2](operands...) -> result { region }
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
///     op arith.constant[value]() -> result {}
/// }
///
/// define_op! {
///     /// Addition operation.
///     op arith.add(lhs, rhs) -> result {}
/// }
///
/// define_op! {
///     /// Function definition.
///     op func.func[sym_name, r#type]() { body }
/// }
/// ```
#[macro_export]
macro_rules! define_op {
    // ========================================================================
    // Entry point - start TT munching
    // ========================================================================
    (
        $(#[$meta:meta])*
        op $dialect:ident.$op:ident $($rest:tt)*
    ) => {
        $crate::define_op!(@parse_attrs
            meta: [$(#[$meta])*],
            dialect: $dialect,
            op: $op,
            rest: [$($rest)*]
        );
    };

    // ========================================================================
    // @parse_attrs - Extract [attrs] if present
    // ========================================================================

    // Has attrs: [$attr, ...]
    (@parse_attrs
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        rest: [[$($attr:ident),* $(,)?] $($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_operands
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),*],
            rest: [$($rest)*]
        );
    };

    // No attrs: starts with (
    (@parse_attrs
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        rest: [$($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_operands
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: [],
            rest: [$($rest)*]
        );
    };

    // ========================================================================
    // @parse_operands - Extract (operands)
    // ========================================================================

    // Variadic: (..$name)
    (@parse_operands
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        rest: [(..$operands:ident) $($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_result
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [variadic: $operands],
            rest: [$($rest)*]
        );
    };

    // Fixed: ($a, $b, ...)
    (@parse_operands
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        rest: [($($operand:ident),* $(,)?) $($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_result
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [fixed: $($operand),*],
            rest: [$($rest)*]
        );
    };

    // ========================================================================
    // @parse_result - Extract -> result if present
    // ========================================================================

    // Has result: -> result
    (@parse_result
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        rest: [-> result $($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_region
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            result: [result_ty],
            rest: [$($rest)*]
        );
    };

    // No result: starts with {
    (@parse_result
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        rest: [$($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_region
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            result: [],
            rest: [$($rest)*]
        );
    };

    // ========================================================================
    // @parse_region - Extract { region } or {}
    // ========================================================================

    // Has region: { $name }
    (@parse_region
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        result: $result:tt,
        rest: [{ $region:ident }]
    ) => {
        $crate::define_op!(@impl
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            result: $result,
            region: [$region]
        );
    };

    // No region: {}
    (@parse_region
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        result: $result:tt,
        rest: [{}]
    ) => {
        $crate::define_op!(@impl
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            result: $result,
            region: []
        );
    };

    // ========================================================================
    // Implementation rule - Fixed operands (unified)
    // ========================================================================

    (@impl
        meta: [$($meta:tt)*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),*],
        operands: [fixed: $($operand:ident),*],
        result: [$($result_ty:ident)?],
        region: [$($region:ident)?]
    ) => {
        $crate::paste::paste! {
            $($meta)*
            #[derive(Clone, Copy)]
            pub struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    $($operand: $crate::Value<'db>,)*
                    $($result_ty: $crate::Type,)?
                    $($attr: $crate::Attribute<'db>,)*
                    $($region: $crate::Region<'db>,)?
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        .operands(vec![$($operand),*])
                        $(.result($result_ty))?
                        $(.attr(stringify!($attr), $attr))*
                        $(.region($region))?
                        .build();
                    Self::wrap_unchecked(op)
                }

                // Operand accessors (only if operands exist)
                $crate::define_op!(@maybe_operand_accessors [$($operand),*]);

                // Attribute accessors
                $(
                    #[allow(dead_code)]
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
                        let key = $crate::Symbol::new(db, stringify!($attr));
                        self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
                    }
                )*

                // Result accessors (only if result exists)
                $crate::define_op!(@maybe_result_accessors [$($result_ty)?]);

                // Region accessor (only if region exists)
                $crate::define_op!(@maybe_region_accessor [$($region)?]);
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
                    // Attribute validation
                    $(
                        {
                            let key = $crate::Symbol::new(db, stringify!($attr));
                            if !op.attributes(db).contains_key(&key) {
                                return Err($crate::ConversionError::MissingAttribute(stringify!($attr)));
                            }
                        }
                    )*
                    // Result validation
                    $crate::define_op!(@maybe_result_validation [$($result_ty)?]; op, db);
                    // Region validation
                    $crate::define_op!(@maybe_region_validation [$($region)?]; op, db);
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // ========================================================================
    // Implementation rule - Variadic operands (unified)
    // ========================================================================

    (@impl
        meta: [$($meta:tt)*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),*],
        operands: [variadic: $operands:ident],
        result: [$($result_ty:ident)?],
        region: [$($region:ident)?]
    ) => {
        $crate::paste::paste! {
            $($meta)*
            #[derive(Clone, Copy)]
            pub struct [<$op:camel>]<'db> {
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
                    $($result_ty: $crate::Type,)?
                    $($attr: $crate::Attribute<'db>,)*
                    $($region: $crate::Region<'db>,)?
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    let op = $crate::Operation::of(db, location, name)
                        .operands($operands)
                        $(.result($result_ty))?
                        $(.attr(stringify!($attr), $attr))*
                        $(.region($region))?
                        .build();
                    Self::wrap_unchecked(op)
                }

                #[allow(dead_code)]
                pub fn operands(&self, db: &'db dyn salsa::Database) -> &[$crate::Value<'db>] {
                    self.op.operands(db)
                }

                // Attribute accessors
                $(
                    #[allow(dead_code)]
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
                        let key = $crate::Symbol::new(db, stringify!($attr));
                        self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
                    }
                )*

                // Result accessors (only if result exists)
                $crate::define_op!(@maybe_result_accessors [$($result_ty)?]);

                // Region accessor (only if region exists)
                $crate::define_op!(@maybe_region_accessor [$($region)?]);
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
                    // Attribute validation
                    $(
                        {
                            let key = $crate::Symbol::new(db, stringify!($attr));
                            if !op.attributes(db).contains_key(&key) {
                                return Err($crate::ConversionError::MissingAttribute(stringify!($attr)));
                            }
                        }
                    )*
                    // Result validation
                    $crate::define_op!(@maybe_result_validation [$($result_ty)?]; op, db);
                    // Region validation
                    $crate::define_op!(@maybe_region_validation [$($region)?]; op, db);
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }
        }
    };

    // ========================================================================
    // Helper rules - Conditional code generation
    // ========================================================================

    // Operand accessors: empty case
    (@maybe_operand_accessors []) => {};
    // Operand accessors: non-empty case - delegate to recursive generator
    (@maybe_operand_accessors [$($operand:ident),+]) => {
        $crate::define_op!(@gen_operand_accessors [], $($operand),+);
    };

    // Generate operand accessors with counter - base case
    (@gen_operand_accessors [$($counter:tt)*], $name:ident) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.operands(db)[$crate::define_op!(@count $($counter)*)]
        }
    };
    // Generate operand accessors with counter - recursive case
    (@gen_operand_accessors [$($counter:tt)*], $name:ident, $($rest:ident),+) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.operands(db)[$crate::define_op!(@count $($counter)*)]
        }
        $crate::define_op!(@gen_operand_accessors [$($counter)* _], $($rest),+);
    };

    // Count tokens helper
    (@count) => { 0usize };
    (@count $_:tt $($rest:tt)*) => { 1usize + $crate::define_op!(@count $($rest)*) };

    // Result accessors: empty case (no result)
    (@maybe_result_accessors []) => {};
    // Result accessors: present case
    (@maybe_result_accessors [$_marker:ident]) => {
        #[allow(dead_code)]
        pub fn result(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.result(db, 0)
        }

        #[allow(dead_code)]
        pub fn result_ty(&self, db: &'db dyn salsa::Database) -> $crate::Type {
            self.op.results(db)[0].clone()
        }
    };

    // Region accessor: empty case (no region)
    (@maybe_region_accessor []) => {};
    // Region accessor: present case
    (@maybe_region_accessor [$region:ident]) => {
        #[allow(dead_code)]
        pub fn $region(&self, db: &'db dyn salsa::Database) -> $crate::Region<'db> {
            self.op.regions(db)[0]
        }
    };

    // Result validation: empty case (no result expected)
    (@maybe_result_validation []; $op:ident, $db:ident) => {};
    // Result validation: present case
    (@maybe_result_validation [$_marker:ident]; $op:ident, $db:ident) => {
        if $op.results($db).is_empty() {
            return Err($crate::ConversionError::MissingResult);
        }
    };

    // Region validation: empty case (no region expected)
    (@maybe_region_validation []; $op:ident, $db:ident) => {};
    // Region validation: present case
    (@maybe_region_validation [$_region:ident]; $op:ident, $db:ident) => {
        if $op.regions($db).is_empty() {
            return Err($crate::ConversionError::MissingRegion);
        }
    };
}
