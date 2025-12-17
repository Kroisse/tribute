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
/// Uses Rust-like syntax for better IDE support and rustfmt compatibility.
///
/// # Syntax
/// ```ignore
/// dialect! {
///     mod dialect_name {
///         /// Doc comment
///         #[attr(attr1, attr2)]
///         fn op_name(operand1, operand2) -> result {
///             #[region(body)] {}
///         };
///     }
/// }
/// ```
///
/// # Features
/// - `#[attr(...)]` - attributes
/// - `(a, b)` - fixed operands
/// - `(#[rest] args)` - variadic operands
/// - `(a, #[rest] rest)` - mixed operands
/// - `-> result` - single result
/// - `-> (a, b)` - multiple results (tuple syntax)
/// - `#[region(name)] {}` - regions inside body
///
/// # Example
/// ```ignore
/// dialect! {
///     mod arith {
///         /// Constant value operation.
///         #[attr(value)]
///         fn r#const() -> result;
///
///         /// Addition operation.
///         fn add(lhs, rhs) -> result;
///     }
/// }
/// ```
#[macro_export]
macro_rules! dialect {
    // Entry point
    (mod $dialect:ident { $($body:tt)* }) => {
        $crate::dialect!(@parse $dialect $($body)*);
    };

    // Base case: no more ops
    (@parse $dialect:ident) => {};

    // Operation with doc + attrs
    (@parse $dialect:ident
        $(#[doc = $doc:literal])+
        #[attr($($attr:ident),* $(,)?)]
        fn $op:ident ($($operands:tt)*) $(-> $result:tt)? $({ $($region_body:tt)* })?;
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            doc: [$($doc),+],
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),*],
            operands: ($($operands)*),
            result: [$($result)?],
            regions: [$($($region_body)*)?]
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // Operation with attrs only (no doc)
    (@parse $dialect:ident
        #[attr($($attr:ident),* $(,)?)]
        fn $op:ident ($($operands:tt)*) $(-> $result:tt)? $({ $($region_body:tt)* })?;
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            doc: [],
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),*],
            operands: ($($operands)*),
            result: [$($result)?],
            regions: [$($($region_body)*)?]
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // Operation with doc only (no attrs)
    (@parse $dialect:ident
        $(#[doc = $doc:literal])+
        fn $op:ident ($($operands:tt)*) $(-> $result:tt)? $({ $($region_body:tt)* })?;
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            doc: [$($doc),+],
            dialect: $dialect,
            op: $op,
            attrs: [],
            operands: ($($operands)*),
            result: [$($result)?],
            regions: [$($($region_body)*)?]
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };

    // Operation without doc or attrs
    (@parse $dialect:ident
        fn $op:ident ($($operands:tt)*) $(-> $result:tt)? $({ $($region_body:tt)* })?;
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            doc: [],
            dialect: $dialect,
            op: $op,
            attrs: [],
            operands: ($($operands)*),
            result: [$($result)?],
            regions: [$($($region_body)*)?]
        }
        $crate::dialect!(@parse $dialect $($rest)*);
    };
}

/// Macro to define a dialect operation wrapper.
///
/// This generates:
/// - A struct wrapping `Operation<'db>`
/// - `DialectOp` trait implementation
/// - `new` constructor with appropriate parameters
/// - Accessor methods based on the operation signature
#[macro_export]
macro_rules! define_op {
    // Entry point
    (
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),*],
        operands: ($($operand_tokens:tt)*),
        result: [$($result:tt)?],
        regions: [$($region_tokens:tt)*]
    ) => {
        $crate::define_op!(@parse_operands
            doc: [$($doc),*],
            dialect: $dialect,
            op: $op,
            attrs: [$($attr),*],
            operand_tokens: [$($operand_tokens)*],
            result: [$($result)?],
            region_tokens: [$($region_tokens)*]
        );
    };

    // ========================================================================
    // @parse_operands - Parse operand list into fixed and variadic
    // ========================================================================

    (@parse_operands
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operand_tokens: [$($tokens:tt)*],
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_operands
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            fixed: [],
            tokens: [$($tokens)*],
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Empty operands
    (@munch_operands
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [],
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_result
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [$($fixed),*;],
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Variadic only or at end
    (@munch_operands
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [#[rest] $var:ident $(,)?],
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_result
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [$($fixed),*; $var],
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Fixed operand followed by comma and more
    (@munch_operands
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [$next:ident, $($remaining:tt)+],
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_operands
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            fixed: [$($fixed,)* $next],
            tokens: [$($remaining)+],
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Last fixed operand
    (@munch_operands
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [$last:ident $(,)?],
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_result
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [$($fixed,)* $last;],
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // ========================================================================
    // @parse_result - Parse result type (single, tuple, or none)
    // ========================================================================

    // No result
    (@parse_result
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        result: [],
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_regions
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: [],
            region_tokens: $region_tokens
        );
    };

    // Tuple result: (a, b, ...)
    (@parse_result
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        result: [($($result:ident),+ $(,)?)],
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_regions
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: [$($result),+],
            region_tokens: $region_tokens
        );
    };

    // Single result
    (@parse_result
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        result: [$result:ident],
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_regions
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: [$result],
            region_tokens: $region_tokens
        );
    };

    // ========================================================================
    // @parse_regions - Parse #[region(name)] {} patterns
    // ========================================================================

    (@parse_regions
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        results: $results:tt,
        region_tokens: [$($tokens:tt)*]
    ) => {
        $crate::define_op!(@munch_regions
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: $results,
            regions: [],
            tokens: [$($tokens)*]
        );
    };

    // Empty regions
    (@munch_regions
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        results: $results:tt,
        regions: [$($region:ident),*],
        tokens: []
    ) => {
        $crate::define_op!(@impl
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: $results,
            regions: [$($region),*]
        );
    };

    // Region: #[region(name)] {}
    (@munch_regions
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        results: $results:tt,
        regions: [$($region:ident),*],
        tokens: [#[region($name:ident)] {} $($rest:tt)*]
    ) => {
        $crate::define_op!(@munch_regions
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: $results,
            regions: [$($region,)* $name],
            tokens: [$($rest)*]
        );
    };

    // ========================================================================
    // @impl - Generate the actual code
    // ========================================================================

    (@impl
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),*],
        operands: [$($fixed:ident),*; $($var:ident)?],
        results: [$($result:ident),*],
        regions: [$($region:ident),*]
    ) => {
        $crate::paste::paste! {
            #[doc = concat!($($doc, "\n",)*)]
            #[derive(Clone, Copy, PartialEq, Eq, salsa::Update)]
            pub struct [<$op:camel>]<'db> {
                op: $crate::Operation<'db>,
            }

            impl<'db> [<$op:camel>]<'db> {
                pub(crate) fn wrap_unchecked(op: $crate::Operation<'db>) -> Self {
                    Self { op }
                }

                /// Get the underlying Operation.
                #[allow(dead_code)]
                pub fn operation(self) -> $crate::Operation<'db> {
                    self.op
                }

                // operand accessors
                $crate::define_op!(@gen_operand_accessors { 0 } $($fixed)* $(#[rest] $var)?);

                // Attribute accessors
                $(
                    #[allow(dead_code, clippy::should_implement_trait)]
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
                        let key = $crate::Symbol::new(db, stringify!($attr));
                        self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
                    }
                )*

                // Result accessors
                $crate::define_op!(@gen_result_accessors { 0 } $($result)*);

                // Region accessors
                $crate::define_op!(@gen_region_accessor { 0 } $($region)*);
            }

            impl<'db> std::ops::Deref for [<$op:camel>]<'db> {
                type Target = $crate::Operation<'db>;
                fn deref(&self) -> &Self::Target {
                    &self.op
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
                    $crate::define_op!(@maybe_result_validation [$($result),*]; op, db);
                    // Region validation
                    $crate::define_op!(@maybe_region_validation [$($region),*]; op, db);
                    Ok(Self { op })
                }

                fn as_operation(&self) -> $crate::Operation<'db> {
                    self.op
                }
            }

            // Constructor function for `$dialect.$op`.
            #[doc = concat!($($doc, "\n",)*)]
            #[allow(clippy::too_many_arguments)]
            pub fn $op<'db>(
                db: &'db dyn salsa::Database,
                location: $crate::Location<'db>,
                $($fixed: $crate::Value<'db>,)*
                $($var: impl IntoIterator<Item = $crate::Value<'db>>,)?
                $($result: $crate::Type<'db>,)*
                $($attr: $crate::Attribute<'db>,)*
                $($region: $crate::Region<'db>,)*
            ) -> [<$op:camel>]<'db> {
                let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                #[allow(unused_mut)]
                let mut operands = $crate::smallvec::smallvec![$($fixed),*];
                $(operands.extend($var);)?
                let op = $crate::Operation::of(db, location, name)
                    .operands(operands)
                    $(.result($result))*
                    $(.attr(stringify!($attr), $attr))*
                    $(.region($region))*
                    .build();
                [<$op:camel>]::wrap_unchecked(op)
            }
        }
    };

    // ========================================================================
    // Helper rules
    // ========================================================================

    // Operand accessors - base case
    (@gen_operand_accessors { $idx:expr }) => {};
    // Operand accessors - variadic
    (@gen_operand_accessors { $idx:expr } #[rest] $var:ident) => {
        #[allow(dead_code)]
        pub fn $var(&self, db: &'db dyn salsa::Database) -> &[$crate::Value<'db>] {
            const FIXED_COUNT: usize = $idx;
            &self.op.operands(db)[FIXED_COUNT..]
        }
    };
    // Operand accessors - fixed
    (@gen_operand_accessors { $idx:expr } $name:ident $($rest:tt)*) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.operands(db)[$idx]
        }
        $crate::define_op!(@gen_operand_accessors { $idx + 1 } $($rest)*);
    };

    // Result accessors - base case
    (@gen_result_accessors { $idx:expr }) => {};
    // Result accessors - recursive
    (@gen_result_accessors { $idx:expr } $name:ident $($rest:ident)*) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.result(db, $idx)
        }

        $crate::paste::paste! {
            #[allow(dead_code)]
            pub fn [<$name _ty>](&self, db: &'db dyn salsa::Database) -> $crate::Type<'db> {
                self.op.results(db)[$idx].clone()
            }
        }

        $crate::define_op!(@gen_result_accessors { $idx + 1 } $($rest)*);
    };

    // Region accessor - base case
    (@gen_region_accessor { $idx:expr }) => {};
    // Region accessor - recursive
    (@gen_region_accessor { $idx:expr } $region:ident $($rest:ident)*) => {
        #[allow(dead_code)]
        pub fn $region(&self, db: &'db dyn salsa::Database) -> $crate::Region<'db> {
            self.op.regions(db)[$idx]
        }

        $crate::define_op!(@gen_region_accessor { $idx + 1 } $($rest)*);
    };

    // Result validation - empty
    (@maybe_result_validation []; $op:ident, $db:ident) => {};
    // Result validation - present
    (@maybe_result_validation [$($result:ident),+]; $op:ident, $db:ident) => {
        {
            const EXPECTED_RESULTS: usize = $crate::define_op!(@count $($result)+);
            if $op.results($db).len() < EXPECTED_RESULTS {
                return Err($crate::ConversionError::MissingResult);
            }
        }
    };

    // Region validation - empty
    (@maybe_region_validation []; $op:ident, $db:ident) => {};
    // Region validation - present
    (@maybe_region_validation [$($region:ident),+]; $op:ident, $db:ident) => {
        {
            const EXPECTED_REGIONS: usize = $crate::define_op!(@count $($region)+);
            if $op.regions($db).len() < EXPECTED_REGIONS {
                return Err($crate::ConversionError::MissingRegion);
            }
        }
    };

    // Count tokens
    (@count) => { 0 };
    (@count $first:tt $($rest:tt)*) => { 1 + $crate::define_op!(@count $($rest)*) };
}
