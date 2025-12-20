//! Dialect operation utilities.

use crate::Operation;

/// Strip `r#` prefix from raw identifier names.
///
/// Rust requires the `r#` prefix for reserved keywords like `type` or `yield`
/// when used as identifiers. This function strips that prefix so the IR stores
/// clean names like "type" instead of "r#type".
#[doc(hidden)]
#[inline]
pub fn strip_raw_prefix(s: &str) -> &str {
    s.strip_prefix("r#").unwrap_or(s)
}

/// Helper macro for attribute type mappings.
///
/// Provides conversions between Rust types and `Attribute` variants:
/// - `any` → `Attribute<'db>` (passthrough)
/// - `bool` → `Attribute::Bool`
/// - `Type` → `Attribute::Type`
/// - `String` → `Attribute::String`
/// - `Symbol` → `Attribute::Symbol`
/// - `SymbolRef` → `Attribute::SymbolRef`
#[macro_export]
macro_rules! attr_type_helper {
    // Rust type for parameter
    (@rust_type any) => { $crate::Attribute<'db> };
    (@rust_type bool) => { bool };
    (@rust_type Type) => { $crate::Type<'db> };
    (@rust_type String) => { std::string::String };
    (@rust_type Symbol) => { $crate::Symbol };
    (@rust_type SymbolRef) => { $crate::SymbolVec };

    // Convert Rust value to Attribute
    (@to_attr any, $val:expr) => { $val };
    (@to_attr bool, $val:expr) => { $crate::Attribute::Bool($val) };
    (@to_attr Type, $val:expr) => { $crate::Attribute::Type($val) };
    (@to_attr String, $val:expr) => { $crate::Attribute::String($val) };
    (@to_attr Symbol, $val:expr) => { $crate::Attribute::Symbol($val) };
    (@to_attr SymbolRef, $val:expr) => { $crate::Attribute::SymbolRef($val) };

    // Convert Attribute to Rust value
    (@from_attr any, $attr:expr) => { $attr.clone() };
    (@from_attr bool, $attr:expr) => {
        match $attr {
            $crate::Attribute::Bool(v) => *v,
            _ => panic!("expected Bool attribute"),
        }
    };
    (@from_attr Type, $attr:expr) => {
        match $attr {
            $crate::Attribute::Type(v) => *v,
            _ => panic!("expected Type attribute"),
        }
    };
    (@from_attr String, $attr:expr) => {
        match $attr {
            $crate::Attribute::String(v) => v.clone(),
            _ => panic!("expected String attribute"),
        }
    };
    (@from_attr Symbol, $attr:expr) => {
        match $attr {
            $crate::Attribute::Symbol(v) => *v,
            _ => panic!("expected Symbol attribute"),
        }
    };
    (@from_attr SymbolRef, $attr:expr) => {
        match $attr {
            $crate::Attribute::SymbolRef(v) => v.clone(),
            _ => panic!("expected SymbolRef attribute"),
        }
    };
}

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

/// Macro to define operations and types in a dialect.
///
/// Uses Rust-like syntax for better IDE support and rustfmt compatibility.
///
/// # Syntax
/// ```
/// # use trunk_ir::dialect;
/// dialect! {
///     mod dialect_name {
///         // Operations
///         /// Doc comment
///         #[attr(attr1, attr2)]
///         fn op_name(operand1, operand2) -> result {
///             #[region(body)] {}
///         };
///
///         // Types
///         /// Doc comment
///         type type_name(param1, param2);
///     }
/// }
/// ```
///
/// # Operation Features
/// - `#[attr(...)]` - attributes
/// - `(a, b)` - fixed operands
/// - `(#[rest] args)` - variadic operands
/// - `(a, #[rest] rest)` - mixed operands
/// - `-> result` - single result
/// - `-> (a, b)` - multiple results (tuple syntax)
/// - `#[region(name)] {}` - regions inside body
///
/// # Type Features
/// - `type name;` - type with no parameters
/// - `type name(a, b);` - type with parameters (generates accessors)
/// - `#[attr(name: Type)]` - typed attribute (bool, Type, String)
///
/// # Example
/// ```
/// # use trunk_ir::dialect;
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
///
/// Types generate wrapper structs with `new()` constructors and `DialectType` impl:
/// ```
/// # use trunk_ir::dialect;
/// dialect! {
///     mod core {
///         /// Tuple cons cell.
///         type tuple(head, tail);
///
///         /// Reference type with nullable attribute.
///         #[attr(nullable: bool)]
///         type ref_(pointee);
///     }
/// }
/// // Usage:
/// // core::Tuple::new(db, head_ty, tail_ty)
/// // core::Ref_::new(db, pointee_ty, true)  // nullable = true
/// // ref_.nullable(db)  // -> bool
/// ```
#[macro_export]
macro_rules! dialect {
    // Entry point - generate _NAME static and parse body
    (mod $dialect:ident { $($body:tt)* }) => {
        // Generate _NAME static (cached dialect name, shared by all operations)
        pub static _NAME: std::sync::LazyLock<$crate::Symbol> =
            std::sync::LazyLock::new(|| $crate::Symbol::new($crate::strip_raw_prefix(stringify!($dialect))));

        $crate::dialect!(@parse $dialect [$($body)*]);
    };

    // Base case: no more items
    (@parse $dialect:ident []) => {};

    // Operation with optional doc and optional typed attrs.
    // Note: doc comments must come before #[attr(...)] if both are present.
    // Supports optional attributes: `#[attr(name?: Type)]`
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ($($operands:tt)*) $(-> $result:tt)? $({ $($region_body:tt)* })?;
         $($rest:tt)*]
    ) => {
        // Generate operation-specific static (cached operation name)
        $crate::paste::paste! {
            pub static [<$op:upper>]: std::sync::LazyLock<$crate::Symbol> =
                std::sync::LazyLock::new(|| $crate::Symbol::new($crate::strip_raw_prefix(stringify!($op))));
        }

        $crate::define_op! {
            doc: [$($doc),*],
            dialect: $dialect,
            op: $op,
            attr_tokens: [$($($attr_tokens)*)?],
            operands: ($($operands)*),
            result: [$($result)?],
            regions: [$($($region_body)*)?]
        }
        $crate::dialect!(@parse $dialect [$($rest)*]);
    };

    // Type with optional doc and optional typed attrs (fixed params).
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr:ident : $attr_ty:ident),* $(,)?)])?
         type $ty:ident $(($($params:ident),* $(,)?))?;
         $($rest:tt)*]
    ) => {
        $crate::define_type! {
            doc: [$($doc),*],
            dialect: $dialect,
            ty: $ty,
            attrs: [$($($attr : $attr_ty),*)?],
            params: [$($($params),*)?]
        }
        $crate::dialect!(@parse $dialect [$($rest)*]);
    };

    // Type with variadic params (#[rest] syntax, like operations).
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr:ident : $attr_ty:ident),* $(,)?)])?
         type $ty:ident(#[rest] $variadic:ident);
         $($rest:tt)*]
    ) => {
        $crate::define_type! {
            @variadic
            doc: [$($doc),*],
            dialect: $dialect,
            ty: $ty,
            attrs: [$($($attr : $attr_ty),*)?],
            variadic: $variadic
        }
        $crate::dialect!(@parse $dialect [$($rest)*]);
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
    // Entry point - parse attribute tokens into structured format
    // Supports: `name`, `name: Type`, `name?: Type` (optional)
    (
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        op: $op:ident,
        attr_tokens: [$($attr_tokens:tt)*],
        operands: ($($operand_tokens:tt)*),
        result: [$($result:tt)?],
        regions: [$($region_tokens:tt)*]
    ) => {
        $crate::define_op!(@munch_attrs
            doc: [$($doc),*],
            dialect: $dialect,
            op: $op,
            attrs: [],
            tokens: [$($attr_tokens)*],
            operand_tokens: [$($operand_tokens)*],
            result: [$($result)?],
            region_tokens: [$($region_tokens)*]
        );
    };

    // ========================================================================
    // @munch_attrs - Parse attribute tokens one by one
    // ========================================================================

    // Done parsing attrs
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@parse_operands
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)*],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Optional typed attr: `name?: Type, ...`
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [$attr:ident ?: $attr_ty:ident, $($rest:tt)*],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_attrs
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)* { $attr : ? $attr_ty }],
            tokens: [$($rest)*],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Optional typed attr (last): `name?: Type`
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [$attr:ident ?: $attr_ty:ident $(,)?],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_attrs
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)* { $attr : ? $attr_ty }],
            tokens: [],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Required typed attr: `name: Type, ...`
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [$attr:ident : $attr_ty:ident, $($rest:tt)*],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_attrs
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)* { $attr : $attr_ty }],
            tokens: [$($rest)*],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Required typed attr (last): `name: Type`
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [$attr:ident : $attr_ty:ident $(,)?],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_attrs
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)* { $attr : $attr_ty }],
            tokens: [],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Untyped attr: `name, ...`
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [$attr:ident, $($rest:tt)*],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_attrs
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)* { $attr }],
            tokens: [$($rest)*],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
        );
    };

    // Untyped attr (last): `name`
    (@munch_attrs
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attrs:tt)*],
        tokens: [$attr:ident $(,)?],
        operand_tokens: $operand_tokens:tt,
        result: $result:tt,
        region_tokens: $region_tokens:tt
    ) => {
        $crate::define_op!(@munch_attrs
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attrs: [$($attrs)* { $attr }],
            tokens: [],
            operand_tokens: $operand_tokens,
            result: $result,
            region_tokens: $region_tokens
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

    // @impl with structured attributes: { name }, { name : Type }, { name : ? Type }
    (@impl
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr_block:tt)*],
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
                $crate::define_op!(@gen_attr_accessors $($attr_block)*);

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
                    // Use cached symbols (lazy-initialized, no write lock after first access)
                    if op.dialect(db) != *_NAME || op.name(db) != *[<$op:upper>] {
                        return Err($crate::ConversionError::WrongOperation {
                            expected: concat!(stringify!($dialect), ".", stringify!($op)),
                            actual: op.full_name(db),
                        });
                    }
                    // Attribute validation
                    $crate::define_op!(@validate_attrs op, db, $($attr_block)*);
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

            // Salsa traits - delegate to inner Operation for tracked function parameters
            impl<'db> salsa::plumbing::AsId for [<$op:camel>]<'db> {
                fn as_id(&self) -> salsa::Id {
                    salsa::plumbing::AsId::as_id(&self.op)
                }
            }

            impl<'db> salsa::plumbing::FromId for [<$op:camel>]<'db> {
                fn from_id(id: salsa::Id) -> Self {
                    Self { op: salsa::plumbing::FromId::from_id(id) }
                }
            }

            impl<'db> salsa::plumbing::SalsaStructInDb for [<$op:camel>]<'db> {
                type MemoIngredientMap =
                    <$crate::Operation<'db> as salsa::plumbing::SalsaStructInDb>::MemoIngredientMap;

                fn lookup_ingredient_index(
                    zalsa: &salsa::plumbing::Zalsa,
                ) -> salsa::plumbing::IngredientIndices {
                    <$crate::Operation<'db> as salsa::plumbing::SalsaStructInDb>::lookup_ingredient_index(zalsa)
                }

                fn entries(
                    zalsa: &salsa::plumbing::Zalsa,
                ) -> impl Iterator<Item = salsa::plumbing::DatabaseKeyIndex> + '_ {
                    <$crate::Operation<'db> as salsa::plumbing::SalsaStructInDb>::entries(zalsa)
                }

                fn cast(
                    id: salsa::Id,
                    type_id: std::any::TypeId,
                ) -> Option<Self> {
                    <$crate::Operation<'db> as salsa::plumbing::SalsaStructInDb>::cast(id, type_id)
                        .map(|op| Self { op })
                }

                unsafe fn memo_table(
                    zalsa: &salsa::plumbing::Zalsa,
                    id: salsa::Id,
                    current_revision: salsa::Revision,
                ) -> salsa::plumbing::MemoTableWithTypes<'_> {
                    // SAFETY: delegating to Operation's memo_table with same arguments
                    unsafe {
                        <$crate::Operation<'db> as salsa::plumbing::SalsaStructInDb>::memo_table(
                            zalsa, id, current_revision,
                        )
                    }
                }
            }
        }

        // Generate constructor
        $crate::define_op!(@gen_constructor
            doc: [$($doc),*],
            dialect: $dialect,
            op: $op,
            attrs: [$($attr_block)*],
            fixed: [$($fixed),*],
            var: [$($var)?],
            results: [$($result),*],
            regions: [$($region),*]
        );
    };

    // Generate constructor - dispatch to munch attrs for proper expansion
    (@gen_constructor
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr_block:tt)*],
        fixed: [$($fixed:ident),*],
        var: [$($var:ident)?],
        results: [$($result:ident),*],
        regions: [$($region:ident),*]
    ) => {
        $crate::define_op!(@gen_constructor_munch
            doc: [$($doc),*],
            dialect: $dialect,
            op: $op,
            attr_params: [],
            attr_blocks: [$($attr_block)*],
            fixed: [$($fixed),*],
            var: [$($var)?],
            results: [$($result),*],
            regions: [$($region),*]
        );
    };

    // Munch through attr blocks to collect parameter list
    (@gen_constructor_munch
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attr_params: [$($param:tt)*],
        attr_blocks: [],
        fixed: $fixed:tt,
        var: $var:tt,
        results: $results:tt,
        regions: $regions:tt
    ) => {
        $crate::define_op!(@gen_constructor_final
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attr_params: [$($param)*],
            fixed: $fixed,
            var: $var,
            results: $results,
            regions: $regions
        );
    };

    // Untyped attr - use [untyped] marker
    (@gen_constructor_munch
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attr_params: [$($param:tt)*],
        attr_blocks: [{ $attr:ident } $($rest:tt)*],
        fixed: $fixed:tt,
        var: $var:tt,
        results: $results:tt,
        regions: $regions:tt
    ) => {
        $crate::define_op!(@gen_constructor_munch
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attr_params: [$($param)* [untyped $attr]],
            attr_blocks: [$($rest)*],
            fixed: $fixed,
            var: $var,
            results: $results,
            regions: $regions
        );
    };

    // Required typed attr - use [required] marker
    (@gen_constructor_munch
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attr_params: [$($param:tt)*],
        attr_blocks: [{ $attr:ident : $attr_ty:ident } $($rest:tt)*],
        fixed: $fixed:tt,
        var: $var:tt,
        results: $results:tt,
        regions: $regions:tt
    ) => {
        $crate::define_op!(@gen_constructor_munch
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attr_params: [$($param)* [required $attr $attr_ty]],
            attr_blocks: [$($rest)*],
            fixed: $fixed,
            var: $var,
            results: $results,
            regions: $regions
        );
    };

    // Optional typed attr - use [optional] marker
    (@gen_constructor_munch
        doc: $doc:tt,
        dialect: $dialect:ident,
        op: $op:ident,
        attr_params: [$($param:tt)*],
        attr_blocks: [{ $attr:ident : ? $attr_ty:ident } $($rest:tt)*],
        fixed: $fixed:tt,
        var: $var:tt,
        results: $results:tt,
        regions: $regions:tt
    ) => {
        $crate::define_op!(@gen_constructor_munch
            doc: $doc,
            dialect: $dialect,
            op: $op,
            attr_params: [$($param)* [optional $attr $attr_ty]],
            attr_blocks: [$($rest)*],
            fixed: $fixed,
            var: $var,
            results: $results,
            regions: $regions
        );
    };

    // Final constructor generation with collected params
    (@gen_constructor_final
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        op: $op:ident,
        attr_params: [$([$kind:ident $attr:ident $($attr_ty:ident)?])*],
        fixed: [$($fixed:ident),*],
        var: [$($var:ident)?],
        results: [$($result:ident),*],
        regions: [$($region:ident),*]
    ) => {
        $crate::paste::paste! {
            #[doc = concat!($($doc, "\n",)*)]
            #[allow(clippy::too_many_arguments)]
            pub fn $op<'db>(
                db: &'db dyn salsa::Database,
                location: $crate::Location<'db>,
                $($fixed: $crate::Value<'db>,)*
                $($var: impl IntoIterator<Item = $crate::Value<'db>>,)?
                $($result: $crate::Type<'db>,)*
                $($attr: $crate::define_op!(@attr_param_type $kind $($attr_ty)?),)*
                $($region: $crate::Region<'db>,)*
            ) -> [<$op:camel>]<'db> {
                let dialect = $crate::Symbol::new($crate::strip_raw_prefix(stringify!($dialect)));
                let name = $crate::Symbol::new($crate::strip_raw_prefix(stringify!($op)));
                #[allow(unused_mut)]
                let mut operands = $crate::idvec![$($fixed),*];
                $(operands.extend($var);)?
                #[allow(unused_mut)]
                let mut builder = $crate::Operation::of(db, location, dialect, name)
                    .operands(operands)
                    $(.result($result))*
                    $(.region($region))*;
                // Add attributes
                $(
                    $crate::define_op!(@add_attr_final builder, $attr, $kind $($attr_ty)?);
                )*
                let op = builder.build();
                [<$op:camel>]::wrap_unchecked(op)
            }
        }
    };

    // Attribute parameter type
    (@attr_param_type untyped) => { $crate::Attribute<'db> };
    (@attr_param_type required $attr_ty:ident) => { $crate::attr_type_helper!(@rust_type $attr_ty) };
    (@attr_param_type optional $attr_ty:ident) => { Option<$crate::attr_type_helper!(@rust_type $attr_ty)> };

    // Add attribute to builder (final stage)
    (@add_attr_final $builder:ident, $attr:ident, untyped) => {
        $builder = $builder.attr($crate::strip_raw_prefix(stringify!($attr)), $attr);
    };
    (@add_attr_final $builder:ident, $attr:ident, required $attr_ty:ident) => {
        $builder = $builder.attr($crate::strip_raw_prefix(stringify!($attr)), $crate::attr_type_helper!(@to_attr $attr_ty, $attr));
    };
    (@add_attr_final $builder:ident, $attr:ident, optional $attr_ty:ident) => {
        if let Some(val) = $attr {
            $builder = $builder.attr($crate::strip_raw_prefix(stringify!($attr)), $crate::attr_type_helper!(@to_attr $attr_ty, val));
        }
    };

    // Add attributes to builder from attr blocks
    (@add_attrs $builder:ident,) => {};
    (@add_attrs $builder:ident, { $attr:ident } $($rest:tt)*) => {
        $builder = $builder.attr($crate::strip_raw_prefix(stringify!($attr)), $attr);
        $crate::define_op!(@add_attrs $builder, $($rest)*);
    };
    (@add_attrs $builder:ident, { $attr:ident : $attr_ty:ident } $($rest:tt)*) => {
        $builder = $builder.attr($crate::strip_raw_prefix(stringify!($attr)), $crate::attr_type_helper!(@to_attr $attr_ty, $attr));
        $crate::define_op!(@add_attrs $builder, $($rest)*);
    };
    (@add_attrs $builder:ident, { $attr:ident : ? $attr_ty:ident } $($rest:tt)*) => {
        if let Some(val) = $attr {
            $builder = $builder.attr($crate::strip_raw_prefix(stringify!($attr)), $crate::attr_type_helper!(@to_attr $attr_ty, val));
        }
        $crate::define_op!(@add_attrs $builder, $($rest)*);
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

    // ========================================================================
    // Typed attribute helpers (block-based)
    // ========================================================================

    // Generate attribute accessors from blocks
    (@gen_attr_accessors) => {};
    (@gen_attr_accessors { $attr:ident } $($rest:tt)*) => {
        #[allow(dead_code, clippy::should_implement_trait)]
        pub fn $attr(&self, db: &'db dyn salsa::Database) -> &'db $crate::Attribute<'db> {
            let key = $crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr)));
            self.op.attributes(db).get(&key).expect(concat!("missing attribute: ", stringify!($attr)))
        }
        $crate::define_op!(@gen_attr_accessors $($rest)*);
    };
    (@gen_attr_accessors { $attr:ident : $attr_ty:ident } $($rest:tt)*) => {
        #[allow(dead_code, clippy::should_implement_trait)]
        pub fn $attr(&self, db: &'db dyn salsa::Database) -> $crate::attr_type_helper!(@rust_type $attr_ty) {
            let attr = self.op.attributes(db)
                .get(&$crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr))))
                .expect(concat!("missing attribute: ", stringify!($attr)));
            $crate::attr_type_helper!(@from_attr $attr_ty, attr)
        }
        $crate::define_op!(@gen_attr_accessors $($rest)*);
    };
    (@gen_attr_accessors { $attr:ident : ? $attr_ty:ident } $($rest:tt)*) => {
        #[allow(dead_code, clippy::should_implement_trait)]
        pub fn $attr(&self, db: &'db dyn salsa::Database) -> Option<$crate::attr_type_helper!(@rust_type $attr_ty)> {
            self.op.attributes(db)
                .get(&$crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr))))
                .map(|attr| $crate::attr_type_helper!(@from_attr $attr_ty, attr))
        }
        $crate::define_op!(@gen_attr_accessors $($rest)*);
    };

    // Validate attributes from blocks
    (@validate_attrs $op:ident, $db:ident,) => {};
    (@validate_attrs $op:ident, $db:ident, { $attr:ident } $($rest:tt)*) => {
        {
            let key = $crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr)));
            if !$op.attributes($db).contains_key(&key) {
                return Err($crate::ConversionError::MissingAttribute(stringify!($attr)));
            }
        }
        $crate::define_op!(@validate_attrs $op, $db, $($rest)*);
    };
    (@validate_attrs $op:ident, $db:ident, { $attr:ident : $attr_ty:ident } $($rest:tt)*) => {
        {
            let key = $crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr)));
            if !$op.attributes($db).contains_key(&key) {
                return Err($crate::ConversionError::MissingAttribute(stringify!($attr)));
            }
        }
        $crate::define_op!(@validate_attrs $op, $db, $($rest)*);
    };
    (@validate_attrs $op:ident, $db:ident, { $attr:ident : ? $attr_ty:ident } $($rest:tt)*) => {
        // Optional attribute - no validation needed
        $crate::define_op!(@validate_attrs $op, $db, $($rest)*);
    };

    // Delegate to attr_type_helper for type mappings
    (@rust_type $attr_ty:ident) => { $crate::attr_type_helper!(@rust_type $attr_ty) };
    (@to_attr $attr_ty:ident, $val:expr) => { $crate::attr_type_helper!(@to_attr $attr_ty, $val) };
    (@from_attr $attr_ty:ident, $attr:expr) => { $crate::attr_type_helper!(@from_attr $attr_ty, $attr) };
}

/// Macro to define a dialect type wrapper.
///
/// This generates:
/// - A struct wrapping `Type<'db>`
/// - `new` constructor with type parameters and typed attributes
/// - `DialectType` trait implementation
/// - `Deref<Target = Type<'db>>` implementation
/// - Accessor methods for type parameters and attributes
///
/// # Supported attribute types
/// - `any` - stored as `Attribute` (any variant)
/// - `bool` - stored as `Attribute::Bool`
/// - `Type` - stored as `Attribute::Type`
/// - `String` - stored as `Attribute::String`
#[macro_export]
macro_rules! define_type {
    (
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        ty: $ty:ident,
        attrs: [$($attr:ident : $attr_ty:ident),*],
        params: [$($param:ident),*]
    ) => {
        $crate::paste::paste! {
            #[doc = concat!($($doc, "\n",)*)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
            pub struct [<$ty:camel>]<'db>($crate::Type<'db>);

            impl<'db> [<$ty:camel>]<'db> {
                /// Create a new instance of this type.
                #[allow(clippy::new_without_default)]
                pub fn new(
                    db: &'db dyn salsa::Database,
                    $($param: $crate::Type<'db>,)*
                    $($attr: $crate::define_type!(@rust_type $attr_ty),)*
                ) -> Self {
                    #[allow(unused_mut)]
                    let mut attrs = std::collections::BTreeMap::new();
                    $(
                        attrs.insert(
                            $crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr))),
                            $crate::define_type!(@to_attr $attr_ty, $attr),
                        );
                    )*
                    Self($crate::Type::new(
                        db,
                        $crate::Symbol::new($crate::strip_raw_prefix(stringify!($dialect))),
                        $crate::Symbol::new($crate::strip_raw_prefix(stringify!($ty))),
                        $crate::idvec![$($param),*],
                        attrs,
                    ))
                }

                // Parameter accessors
                $crate::define_type!(@gen_param_accessors { 0 } $($param)*);

                // Attribute accessors
                $(
                    #[allow(dead_code)]
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> $crate::define_type!(@rust_type $attr_ty) {
                        let attr = self.0.get_attr(db, $crate::strip_raw_prefix(stringify!($attr)))
                            .expect(concat!("missing attribute: ", stringify!($attr)));
                        $crate::define_type!(@from_attr $attr_ty, attr)
                    }
                )*
            }

            impl<'db> std::ops::Deref for [<$ty:camel>]<'db> {
                type Target = $crate::Type<'db>;
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<'db> $crate::DialectType<'db> for [<$ty:camel>]<'db> {
                fn as_type(&self) -> $crate::Type<'db> {
                    self.0
                }

                fn from_type(db: &'db dyn salsa::Database, ty: $crate::Type<'db>) -> Option<Self> {
                    if ty.dialect(db) == $crate::Symbol::new($crate::strip_raw_prefix(stringify!($dialect)))
                        && ty.name(db) == $crate::Symbol::new($crate::strip_raw_prefix(stringify!($ty)))
                    {
                        Some(Self(ty))
                    } else {
                        None
                    }
                }
            }
        }
    };

    // Variadic params variant - constructor takes IdVec<Type>
    (@variadic
        doc: [$($doc:literal),*],
        dialect: $dialect:ident,
        ty: $ty:ident,
        attrs: [$($attr:ident : $attr_ty:ident),*],
        variadic: $variadic:ident
    ) => {
        $crate::paste::paste! {
            #[doc = concat!($($doc, "\n",)*)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
            pub struct [<$ty:camel>]<'db>($crate::Type<'db>);

            impl<'db> [<$ty:camel>]<'db> {
                /// Create a new instance of this type with variadic type parameters.
                #[allow(clippy::new_without_default)]
                pub fn new(
                    db: &'db dyn salsa::Database,
                    $variadic: $crate::IdVec<$crate::Type<'db>>,
                    $($attr: $crate::define_type!(@rust_type $attr_ty),)*
                ) -> Self {
                    #[allow(unused_mut)]
                    let mut attrs = std::collections::BTreeMap::new();
                    $(
                        attrs.insert(
                            $crate::Symbol::new($crate::strip_raw_prefix(stringify!($attr))),
                            $crate::define_type!(@to_attr $attr_ty, $attr),
                        );
                    )*
                    Self($crate::Type::new(
                        db,
                        $crate::Symbol::new($crate::strip_raw_prefix(stringify!($dialect))),
                        $crate::Symbol::new($crate::strip_raw_prefix(stringify!($ty))),
                        $variadic,
                        attrs,
                    ))
                }

                /// Get all type parameters.
                #[allow(dead_code)]
                pub fn $variadic(&self, db: &'db dyn salsa::Database) -> &[$crate::Type<'db>] {
                    self.0.params(db)
                }

                // Attribute accessors
                $(
                    #[allow(dead_code)]
                    pub fn $attr(&self, db: &'db dyn salsa::Database) -> $crate::define_type!(@rust_type $attr_ty) {
                        let attr = self.0.get_attr(db, $crate::strip_raw_prefix(stringify!($attr)))
                            .expect(concat!("missing attribute: ", stringify!($attr)));
                        $crate::define_type!(@from_attr $attr_ty, attr)
                    }
                )*
            }

            impl<'db> std::ops::Deref for [<$ty:camel>]<'db> {
                type Target = $crate::Type<'db>;
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<'db> $crate::DialectType<'db> for [<$ty:camel>]<'db> {
                fn as_type(&self) -> $crate::Type<'db> {
                    self.0
                }

                fn from_type(db: &'db dyn salsa::Database, ty: $crate::Type<'db>) -> Option<Self> {
                    if ty.dialect(db) == $crate::Symbol::new($crate::strip_raw_prefix(stringify!($dialect)))
                        && ty.name(db) == $crate::Symbol::new($crate::strip_raw_prefix(stringify!($ty)))
                    {
                        Some(Self(ty))
                    } else {
                        None
                    }
                }
            }
        }
    };

    // Delegate to attr_type_helper for type mappings
    (@rust_type $attr_ty:ident) => { $crate::attr_type_helper!(@rust_type $attr_ty) };
    (@to_attr $attr_ty:ident, $val:expr) => { $crate::attr_type_helper!(@to_attr $attr_ty, $val) };
    (@from_attr $attr_ty:ident, $attr:expr) => { $crate::attr_type_helper!(@from_attr $attr_ty, $attr) };

    // Parameter accessors - base case
    (@gen_param_accessors { $idx:expr }) => {};

    // Parameter accessors - recursive
    (@gen_param_accessors { $idx:expr } $name:ident $($rest:ident)*) => {
        #[allow(dead_code)]
        pub fn $name(&self, db: &'db dyn salsa::Database) -> $crate::Type<'db> {
            self.0.params(db)[$idx]
        }

        $crate::define_type!(@gen_param_accessors { $idx + 1 } $($rest)*);
    };
}
