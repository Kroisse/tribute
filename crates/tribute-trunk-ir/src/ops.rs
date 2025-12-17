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

    // Unified pattern: optional attrs, optional results
    (@parse $dialect:ident
        $(#[$meta:meta])*
        op $op:ident $([$($attrs:tt)*])? ($($operands:tt)*) $( -> $($result:ident),+ $(,)? )? $(@ $region:ident {})*;
        $($rest:tt)*
    ) => {
        $crate::define_op! {
            $(#[$meta])*
            op $dialect.$op $([$($attrs)*])? ($($operands)*) $(-> $($result),+)? $(@ $region {})*
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
    // @parse_operands - Delegate to @parse_operand_list for TT munching
    // ========================================================================

    (@parse_operands
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        rest: [($($operand_tokens:tt)*) $($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_operand_list
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            fixed: [],
            tokens: [$($operand_tokens)*],
            rest: [$($rest)*]
        );
    };

    // ========================================================================
    // @parse_operand_list - TT munch operands: (a, b, ..rest) patterns
    // Output format: [$($fixed),*; $($var)?]
    // ========================================================================

    // No operands left - fixed only (or empty)
    (@parse_operand_list
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [],
        rest: $rest:tt
    ) => {
        $crate::define_op!(@parse_result
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [$($fixed),*;],
            rest: $rest
        );
    };

    // Variadic (with or without fixed operands before)
    (@parse_operand_list
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [..$var:ident $(,)?],
        rest: $rest:tt
    ) => {
        $crate::define_op!(@parse_result
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [$($fixed),*; $var],
            rest: $rest
        );
    };

    // Fixed operand followed by comma and more tokens
    (@parse_operand_list
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [$next:ident, $($remaining:tt)+],
        rest: $rest:tt
    ) => {
        $crate::define_op!(@parse_operand_list
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            fixed: [$($fixed,)* $next],
            tokens: [$($remaining)+],
            rest: $rest
        );
    };

    // Last fixed operand (with optional trailing comma)
    (@parse_operand_list
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        fixed: [$($fixed:ident),*],
        tokens: [$last:ident $(,)?],
        rest: $rest:tt
    ) => {
        $crate::define_op!(@parse_result
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: [$($fixed,)* $last;],
            rest: $rest
        );
    };

    // ========================================================================
    // @parse_result - Extract -> result(s) if present
    // ========================================================================

    // Delegate to @parse_result_list for TT munching
    (@parse_result
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        rest: [-> $($rest:tt)*]
    ) => {
        $crate::define_op!(@parse_result_list
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: [],
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
    // @parse_result_list - TT munch result names until { is reached
    // ========================================================================

    // Result followed by comma - continue munching
    (@parse_result_list
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        results: [$($results:ident),*],
        rest: [$next:ident, $($remaining:tt)+]
    ) => {
        $crate::define_op!(@parse_result_list
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            results: [$($results,)* $next],
            rest: [$($remaining)+]
        );
    };

    // Last result (followed by region)
    (@parse_result_list
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        results: [$($results:ident),*],
        rest: [$last:ident $($remaining:tt)*]
    ) => {
        $crate::define_op!(@parse_region
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            result: [$($results,)* $last],
            rest: [$($remaining)*]
        );
    };

    // ========================================================================
    // @parse_region - Extract { region } or {}
    // ========================================================================

    // Has regions: { $name }
    (@parse_region
        meta: $meta:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: $attrs:tt,
        operands: $operands:tt,
        result: $result:tt,
        rest: [$( @ $region:ident { } )*]
    ) => {
        $crate::define_op!(@impl
            meta: $meta,
            dialect: $dialect,
            op: $op,
            attrs: $attrs,
            operands: $operands,
            result: $result,
            region: [$($region),*]
        );
    };

    // ========================================================================
    // Implementation rule - Unified operands handling
    // Format: [$($fixed),*; $($var)?]
    // ========================================================================

    (@impl
        meta: [$($meta:tt)*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr:ident),*],
        operands: [$($fixed:ident),*; $($var:ident)?],
        result: [$($result:ident),*],
        region: [$($region:ident),*]
    ) => {
        $crate::paste::paste! {
            $($meta)*
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

                #[allow(clippy::too_many_arguments)]
                pub fn new(
                    db: &'db dyn salsa::Database,
                    location: $crate::Location<'db>,
                    $($fixed: $crate::Value<'db>,)*
                    $($var: Vec<$crate::Value<'db>>,)?
                    $($result: $crate::Type,)*
                    $($attr: $crate::Attribute<'db>,)*
                    $($region: $crate::Region<'db>,)*
                ) -> Self {
                    let name = $crate::OpNameId::new(db, stringify!($dialect), stringify!($op));
                    #[allow(unused_mut)]
                    let mut operands = vec![$($fixed),*];
                    $(operands.extend($var);)?
                    let op = $crate::Operation::of(db, location, name)
                        .operands(operands)
                        $(.result($result))*
                        $(.attr(stringify!($attr), $attr))*
                        $(.region($region))*
                        .build();
                    Self::wrap_unchecked(op)
                }

                // operand accessors (if any)
                $crate::define_op!(@gen_operand_accessors { 0 } $($fixed)* $(..$var)?);

                // Attribute accessors
                $(
                    #[allow(dead_code, clippy::should_implement_trait)]
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
                        let key = $crate::Symbol::new(db, stringify!($attr));
                        self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
                    }
                )*

                // Result accessors (only if results exist)
                $crate::define_op!(@gen_result_accessors { 0 } $($result)*);

                // Region accessor (only if region exists)
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
        }
    };

    // ========================================================================
    // Helper rules - Conditional code generation
    // ========================================================================

    // Generate operand accessors with index - base case
    (@gen_operand_accessors { $idx:expr }) => {};
    // Generate operand accessors with index - varadic case
    (@gen_operand_accessors { $idx:expr } ..$var:ident) => {
        #[allow(dead_code)]
        pub fn $var(&self, db: &'db dyn salsa::Database) -> &[$crate::Value<'db>] {
            const FIXED_COUNT: usize = $idx;
            &self.op.operands(db)[FIXED_COUNT..]
        }
    };
    // Generate operand accessors with index - recursive case
    (@gen_operand_accessors { $idx:expr } $name:ident $($rest:tt)*) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.operands(db)[$idx]
        }
        $crate::define_op!(@gen_operand_accessors { $idx + 1 } $($rest)*);
    };

    // Generate result accessors with index - base case
    (@gen_result_accessors { $idx:expr }) => {};
    // Generate result accessors with index - recursive case
    (@gen_result_accessors { $idx:expr } $name:ident $($rest:ident)*) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Value<'db> {
            self.op.result(db, $idx)
        }

        $crate::paste::paste! {
            #[allow(dead_code)]
            pub fn [<$name _ty>](&self, db: &'db dyn salsa::Database) -> $crate::Type {
                self.op.results(db)[$idx].clone()
            }
        }

        $crate::define_op!(@gen_result_accessors { $idx + 1 } $($rest)*);
    };

    // Region accessor: empty case (no region)
    (@gen_region_accessor { $idx:expr }) => {};
    // Region accessor: present case
    (@gen_region_accessor { $idx:expr } $region:ident $($rest:ident)*) => {
        #[allow(dead_code)]
        pub fn $region(&self, db: &'db dyn salsa::Database) -> $crate::Region<'db> {
            self.op.regions(db)[$idx]
        }

        $crate::define_op!(@gen_region_accessor { $idx + 1 } $($rest)*);
    };

    // Result validation: empty case (no result expected)
    (@maybe_result_validation []; $op:ident, $db:ident) => {};
    // Result validation: one or more results expected
    (@maybe_result_validation [$($result:ident),+]; $op:ident, $db:ident) => {
        {
            const EXPECTED_RESULTS: usize = $crate::define_op!(@count $($result)+);
            if $op.results($db).len() < EXPECTED_RESULTS {
                return Err($crate::ConversionError::MissingResult);
            }
        }
    };

    // Region validation: empty case (no region expected)
    (@maybe_region_validation []; $op:ident, $db:ident) => {};
    // Region validation: one or more regions expected
    (@maybe_region_validation [$($region:ident),+]; $op:ident, $db:ident) => {
        {
            const EXPECTED_REGIONS: usize = $crate::define_op!(@count $($region)+);
            if $op.regions($db).len() < EXPECTED_REGIONS {
                return Err($crate::ConversionError::MissingRegion);
            }
        }
    };

    // ========================================================================
    // @count - Count tokens (used for fixed operand counts and result counts)
    // ========================================================================
    (@count) => { 0 };
    (@count $first:tt $($rest:tt)*) => { 1 + $crate::define_op!(@count $($rest)*) };
}
