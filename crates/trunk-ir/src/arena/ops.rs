//! Arena dialect operation utilities.
//!
//! Provides `ArenaDialectOp` trait and `arena_dialect!` macro for defining
//! arena-based dialect operations.

use super::context::IrContext;
use super::refs::OpRef;
use crate::ops::ConversionError;

/// Trait for arena-based dialect operation wrappers.
pub trait ArenaDialectOp: Sized + Copy {
    const DIALECT_NAME: &'static str;
    const OP_NAME: &'static str;

    fn from_op(ctx: &IrContext, op: OpRef) -> Result<Self, ConversionError>;
    fn op_ref(&self) -> OpRef;

    fn matches(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        data.dialect == crate::Symbol::new(Self::DIALECT_NAME)
            && data.name == crate::Symbol::new(Self::OP_NAME)
    }
}

/// Helper macro for arena attribute type mappings.
#[doc(hidden)]
#[macro_export]
macro_rules! arena_attr_type_helper {
    (@rust_type any) => {
        $crate::arena::Attribute
    };
    (@rust_type bool) => {
        bool
    };
    (@rust_type i32) => {
        i32
    };
    (@rust_type i64) => {
        i64
    };
    (@rust_type u32) => {
        u32
    };
    (@rust_type u64) => {
        u64
    };
    (@rust_type f32) => {
        f32
    };
    (@rust_type f64) => {
        f64
    };
    (@rust_type Type) => {
        $crate::arena::TypeRef
    };
    (@rust_type String) => {
        std::string::String
    };
    (@rust_type Symbol) => {
        $crate::Symbol
    };
    (@rust_type QualifiedName) => {
        $crate::Symbol
    };

    (@to_attr any, $val:expr) => {
        $val
    };
    (@to_attr bool, $val:expr) => {
        $crate::arena::Attribute::Bool($val)
    };
    (@to_attr i32, $val:expr) => {
        $crate::arena::Attribute::IntBits($val as u64)
    };
    (@to_attr i64, $val:expr) => {
        $crate::arena::Attribute::IntBits($val as u64)
    };
    (@to_attr u32, $val:expr) => {
        $crate::arena::Attribute::IntBits($val as u64)
    };
    (@to_attr u64, $val:expr) => {
        $crate::arena::Attribute::IntBits($val)
    };
    (@to_attr f32, $val:expr) => {
        $crate::arena::Attribute::FloatBits(($val as f64).to_bits())
    };
    (@to_attr f64, $val:expr) => {
        $crate::arena::Attribute::FloatBits($val.to_bits())
    };
    (@to_attr Type, $val:expr) => {
        $crate::arena::Attribute::Type($val)
    };
    (@to_attr String, $val:expr) => {
        $crate::arena::Attribute::String($val)
    };
    (@to_attr Symbol, $val:expr) => {
        $crate::arena::Attribute::Symbol($val)
    };
    (@to_attr QualifiedName, $val:expr) => {
        $crate::arena::Attribute::Symbol($val)
    };

    (@from_attr any, $attr:expr) => {
        $attr.clone()
    };
    (@from_attr bool, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::Bool(v) => *v,
            _ => panic!("expected Bool attribute"),
        }
    };
    (@from_attr i32, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::IntBits(v) => *v as i32,
            _ => panic!("expected IntBits attribute"),
        }
    };
    (@from_attr i64, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::IntBits(v) => *v as i64,
            _ => panic!("expected IntBits attribute"),
        }
    };
    (@from_attr u32, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::IntBits(v) => *v as u32,
            _ => panic!("expected IntBits attribute"),
        }
    };
    (@from_attr u64, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::IntBits(v) => *v,
            _ => panic!("expected IntBits attribute"),
        }
    };
    (@from_attr f32, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::FloatBits(v) => f64::from_bits(*v) as f32,
            _ => panic!("expected FloatBits attribute"),
        }
    };
    (@from_attr f64, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::FloatBits(v) => f64::from_bits(*v),
            _ => panic!("expected FloatBits attribute"),
        }
    };
    (@from_attr Type, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::Type(v) => *v,
            _ => panic!("expected Type attribute"),
        }
    };
    (@from_attr String, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::String(v) => v.clone(),
            _ => panic!("expected String attribute"),
        }
    };
    (@from_attr Symbol, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::Symbol(v) => *v,
            _ => panic!("expected Symbol attribute"),
        }
    };
    (@from_attr QualifiedName, $attr:expr) => {
        match $attr {
            $crate::arena::Attribute::Symbol(v) => *v,
            _ => panic!("expected Symbol attribute"),
        }
    };
}

/// Main macro for defining arena-based dialect operations and types.
///
/// The syntax is identical to the Salsa-based `dialect!` macro.
/// Generated code uses arena types (OpRef, ValueRef, TypeRef, IrContext)
/// instead of Salsa types.
#[macro_export]
macro_rules! arena_dialect {
    // Entry point
    (mod $dialect:ident { $($body:tt)* }) => {
        #[allow(non_snake_case)]
        #[inline]
        pub fn DIALECT_NAME() -> $crate::Symbol {
            $crate::Symbol::new($crate::raw_ident_str!($dialect))
        }

        $crate::arena_dialect!(@parse $dialect [$($body)*]);
    };

    // Base case
    (@parse $dialect:ident []) => {};

    // ========================================================================
    // Type definitions
    // ========================================================================

    // Type with no params and no attrs
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         type $name:ident;
         $($rest:tt)*]
    ) => {
        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // Type with params (no attrs)
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         type $name:ident($($param:ident),*);
         $($rest:tt)*]
    ) => {
        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // Type with attrs and optional params
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         #[attr($($attr_tokens:tt)*)]
         type $name:ident$(($($param:ident),*))?;
         $($rest:tt)*]
    ) => {
        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: variadic results `-> #[rest] name`
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ($($operands:tt)*) -> #[rest] $result:ident $({ $($region_body:tt)* })?;
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [$($operands)*],
            results: [#[rest] $result],
            regions: [$($($region_body)*)?],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: multi results `-> (a, b)`
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ($($operands:tt)*) -> ($($result:ident),+) $({ $($region_body:tt)* })?;
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [$($operands)*],
            results: [$($result),+],
            regions: [$($($region_body)*)?],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: single result `-> name`
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ($($operands:tt)*) -> $result:ident $({ $($region_body:tt)* })?;
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [$($operands)*],
            results: [$result],
            regions: [$($($region_body)*)?],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: no result, with body
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ($($operands:tt)*) { $($region_body:tt)* };
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [$($operands)*],
            results: [],
            regions: [$($region_body)*],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: no result, no body
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ($($operands:tt)*);
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [$($operands)*],
            results: [],
            regions: [],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: no operands, with body and optional result
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident () $(-> $result:ident)? { $($region_body:tt)* };
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [],
            results: [$($result)?],
            regions: [$($region_body)*],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: no operands, no body, with result
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident () -> $result:ident;
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [],
            results: [$result],
            regions: [],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };

    // ========================================================================
    // Operations: no operands, no body, no result
    // ========================================================================
    (@parse $dialect:ident
        [$(#[doc = $doc:literal])*
         $(#[attr($($attr_tokens:tt)*)])?
         fn $op:ident ();
         $($rest:tt)*]
    ) => {
        $crate::paste::paste! {
            #[allow(non_snake_case)]
            #[inline]
            pub fn [<$op:upper>]() -> $crate::Symbol {
                $crate::Symbol::new($crate::raw_ident_str!($op))
            }
        }

        $crate::arena_define_op! {
            dialect: $dialect,
            op: $op,
            attrs: [$($($attr_tokens)*)?],
            operands: [],
            results: [],
            regions: [],
        }

        $crate::arena_dialect!(@parse $dialect [$($rest)*]);
    };
}

/// Internal macro to define a single arena operation.
///
/// Generates:
/// - Wrapper struct (newtype over OpRef)
/// - ArenaDialectOp trait impl
/// - Operand, result, attribute, region, successor accessors
/// - Constructor function
#[macro_export]
macro_rules! arena_define_op {
    (
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        operands: [$($operand_tokens:tt)*],
        results: [$($result_tokens:tt)*],
        regions: [$($region_tokens:tt)*],
    ) => {
        $crate::paste::paste! {
            /// Arena-based dialect operation wrapper.
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub struct [<$op:camel>]($crate::arena::OpRef);

            impl $crate::arena::ops::ArenaDialectOp for [<$op:camel>] {
                const DIALECT_NAME: &'static str = $crate::raw_ident_str!($dialect);
                const OP_NAME: &'static str = $crate::raw_ident_str!($op);

                fn from_op(
                    ctx: &$crate::arena::IrContext,
                    op: $crate::arena::OpRef,
                ) -> Result<Self, $crate::ops::ConversionError> {
                    if !Self::matches(ctx, op) {
                        return Err($crate::ops::ConversionError::WrongOperation {
                            expected: concat!(
                                $crate::raw_ident_str!($dialect), ".",
                                $crate::raw_ident_str!($op)
                            ),
                            actual: format!("{}.{}",
                                ctx.op(op).dialect,
                                ctx.op(op).name),
                        });
                    }
                    Ok(Self(op))
                }

                fn op_ref(&self) -> $crate::arena::OpRef {
                    self.0
                }
            }

            impl [<$op:camel>] {
                /// Get the underlying OpRef.
                pub fn op_ref(&self) -> $crate::arena::OpRef {
                    self.0
                }

                // Generate operand accessors
                $crate::arena_operand_accessors!($op, [$($operand_tokens)*]);

                // Generate result accessors
                $crate::arena_result_accessors!($op, [$($result_tokens)*]);

                // Generate attribute accessors
                $crate::arena_attr_accessors!([$($attr_tokens)*]);

                // Generate region/successor accessors
                $crate::arena_region_accessors!([$($region_tokens)*]);
            }

            // Generate constructor function
            $crate::arena_constructor!(
                dialect: $dialect,
                op: $op,
                attrs: [$($attr_tokens)*],
                operands: [$($operand_tokens)*],
                results: [$($result_tokens)*],
                regions: [$($region_tokens)*],
            );
        }
    };
}

// ============================================================================
// Operand accessor generation
// ============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! arena_operand_accessors {
    // No operands
    ($op:ident, []) => {};

    // Variadic only: `#[rest] name`
    ($op:ident, [#[rest] $name:ident]) => {
        pub fn $name<'a>(&self, ctx: &'a $crate::arena::IrContext) -> &'a [$crate::arena::ValueRef] {
            ctx.op_operands(self.0)
        }
    };

    // Parse operands recursively to collect fixed + optional rest
    ($op:ident, [$($tokens:tt)*]) => {
        $crate::arena_operand_accessors!(@collect $op, 0, [], [$($tokens)*]);
    };

    // Collect: hit rest
    (@collect $op:ident, $idx:expr, [$($fixed:tt)*], [#[rest] $name:ident]) => {
        $crate::arena_operand_accessors!(@emit_fixed $idx, [$($fixed)*]);
        pub fn $name<'a>(&self, ctx: &'a $crate::arena::IrContext) -> &'a [$crate::arena::ValueRef] {
            &ctx.op_operands(self.0)[$idx..]
        }
    };

    // Collect: more fixed operands (with comma)
    (@collect $op:ident, $idx:expr, [$($fixed:tt)*], [$name:ident, $($rest:tt)*]) => {
        $crate::arena_operand_accessors!(@collect $op, $idx + 1, [$($fixed)* ($name, $idx)], [$($rest)*]);
    };

    // Collect: last fixed operand (no trailing comma, no rest)
    (@collect $op:ident, $idx:expr, [$($fixed:tt)*], [$name:ident]) => {
        $crate::arena_operand_accessors!(@emit_fixed $idx + 1, [$($fixed)* ($name, $idx)]);
    };

    // Emit all fixed operand accessors
    (@emit_fixed $total:expr, [$(($name:ident, $idx:expr))*]) => {
        $(
            pub fn $name(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::ValueRef {
                ctx.op_operands(self.0)[$idx]
            }
        )*
    };
}

// ============================================================================
// Result accessor generation
// ============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! arena_result_accessors {
    // No results
    ($op:ident, []) => {};

    // Variadic results: `#[rest] name`
    ($op:ident, [#[rest] $name:ident]) => {
        pub fn $name<'a>(&self, ctx: &'a $crate::arena::IrContext) -> &'a [$crate::arena::ValueRef] {
            ctx.op_results(self.0)
        }
    };

    // Single result
    ($op:ident, [$result:ident]) => {
        pub fn $result(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::ValueRef {
            ctx.op_result(self.0, 0)
        }

        pub fn result_ty(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::TypeRef {
            ctx.op_result_types(self.0)[0]
        }
    };

    // Multi results: (a, b, ...)
    ($op:ident, [$($result:ident),+]) => {
        $crate::arena_result_accessors!(@multi 0, $($result),+);
    };

    (@multi $idx:expr, $result:ident) => {
        pub fn $result(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::ValueRef {
            ctx.op_result(self.0, $idx)
        }

        $crate::paste::paste! {
            pub fn [<$result _ty>](&self, ctx: &$crate::arena::IrContext) -> $crate::arena::TypeRef {
                ctx.op_result_types(self.0)[$idx as usize]
            }
        }
    };

    (@multi $idx:expr, $result:ident, $($rest:ident),+) => {
        pub fn $result(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::ValueRef {
            ctx.op_result(self.0, $idx)
        }

        $crate::paste::paste! {
            pub fn [<$result _ty>](&self, ctx: &$crate::arena::IrContext) -> $crate::arena::TypeRef {
                ctx.op_result_types(self.0)[$idx as usize]
            }
        }

        $crate::arena_result_accessors!(@multi $idx + 1, $($rest),+);
    };
}

// ============================================================================
// Attribute accessor generation
// ============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! arena_attr_accessors {
    ([]) => {};

    ([$name:ident : $ty:ident $(, $($rest:tt)*)?]) => {
        $crate::paste::paste! {
            pub fn $name(&self, ctx: &$crate::arena::IrContext) -> $crate::arena_attr_type_helper!(@rust_type $ty) {
                let attr = ctx.op(self.0).attributes
                    .get(&$crate::Symbol::new($crate::raw_ident_str!($name)))
                    .expect(concat!("missing attribute: ", $crate::raw_ident_str!($name)));
                $crate::arena_attr_type_helper!(@from_attr $ty, attr)
            }
        }

        $crate::arena_attr_accessors!([$($($rest)*)?]);
    };

    // Handle optional attribute (marked with ?)
    ([$name:ident ?: $ty:ident $(, $($rest:tt)*)?]) => {
        $crate::paste::paste! {
            pub fn $name(&self, ctx: &$crate::arena::IrContext) -> Option<$crate::arena_attr_type_helper!(@rust_type $ty)> {
                ctx.op(self.0).attributes
                    .get(&$crate::Symbol::new($crate::raw_ident_str!($name)))
                    .map(|attr| $crate::arena_attr_type_helper!(@from_attr $ty, attr))
            }
        }

        $crate::arena_attr_accessors!([$($($rest)*)?]);
    };
}

// ============================================================================
// Region/successor accessor generation
// ============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! arena_region_accessors {
    ([]) => {};

    ([$($tokens:tt)*]) => {
        $crate::arena_region_accessors!(@parse 0, 0, [$($tokens)*]);
    };

    // Region
    (@parse $region_idx:expr, $succ_idx:expr,
     [#[region($name:ident)] {} $($rest:tt)*]
    ) => {
        pub fn $name(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::RegionRef {
            ctx.op(self.0).regions[$region_idx]
        }

        $crate::arena_region_accessors!(@parse $region_idx + 1, $succ_idx, [$($rest)*]);
    };

    // Successor
    (@parse $region_idx:expr, $succ_idx:expr,
     [#[successor($name:ident)] $($rest:tt)*]
    ) => {
        pub fn $name(&self, ctx: &$crate::arena::IrContext) -> $crate::arena::BlockRef {
            ctx.op(self.0).successors[$succ_idx]
        }

        $crate::arena_region_accessors!(@parse $region_idx, $succ_idx + 1, [$($rest)*]);
    };

    // Done
    (@parse $region_idx:expr, $succ_idx:expr, []) => {};
}

// ============================================================================
// Constructor function generation (accumulator pattern)
// ============================================================================

/// Generates constructor functions for arena dialect operations.
///
/// Uses an accumulator pattern to collect all parameters into a token list,
/// then emits the complete function in one shot. Body code is generated at
/// emit time from structured descriptors to avoid macro hygiene issues.
#[doc(hidden)]
#[macro_export]
macro_rules! arena_constructor {
    // Entry: start accumulating from operands
    (
        dialect: $dialect:ident,
        op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        operands: [$($operand_tokens:tt)*],
        results: [$($result_tokens:tt)*],
        regions: [$($region_tokens:tt)*],
    ) => {
        $crate::arena_constructor!(@operands
            dialect: $dialect,
            op: $op,
            attrs: [$($attr_tokens)*],
            operand_tokens: [$($operand_tokens)*],
            results: [$($result_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [],
            variadic_op: [],
            params: [],
        );
    };

    // ========================================================================
    // Phase 1: Collect operand params
    // ========================================================================

    // No operands → move to results
    (@operands
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        operand_tokens: [],
        results: [$($result_tokens:tt)*],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@results
            dialect: $dialect, op: $op,
            attrs: [$($attr_tokens)*],
            results: [$($result_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            params: [$($params)*],
        );
    };

    // Variadic operand
    (@operands
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        operand_tokens: [#[rest] $name:ident],
        results: [$($result_tokens:tt)*],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@results
            dialect: $dialect, op: $op,
            attrs: [$($attr_tokens)*],
            results: [$($result_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$name],
            params: [$($params)* $name: impl IntoIterator<Item = $crate::arena::ValueRef>,],
        );
    };

    // Fixed operand with more
    (@operands
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        operand_tokens: [$name:ident, $($rest:tt)*],
        results: [$($result_tokens:tt)*],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@operands
            dialect: $dialect, op: $op,
            attrs: [$($attr_tokens)*],
            operand_tokens: [$($rest)*],
            results: [$($result_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)* $name],
            variadic_op: [$($variadic_op)*],
            params: [$($params)* $name: $crate::arena::ValueRef,],
        );
    };

    // Last fixed operand
    (@operands
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        operand_tokens: [$name:ident],
        results: [$($result_tokens:tt)*],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@results
            dialect: $dialect, op: $op,
            attrs: [$($attr_tokens)*],
            results: [$($result_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)* $name],
            variadic_op: [$($variadic_op)*],
            params: [$($params)* $name: $crate::arena::ValueRef,],
        );
    };

    // ========================================================================
    // Phase 2: Collect result type params
    // ========================================================================

    // No results
    (@results
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        results: [],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@attrs
            dialect: $dialect, op: $op,
            attr_tokens: [$($attr_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [],
            variadic_result_var: [],
            attrs_structured: [],
            params: [$($params)*],
        );
    };

    // Variadic results
    (@results
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        results: [#[rest] $name:ident],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@attrs
            dialect: $dialect, op: $op,
            attr_tokens: [$($attr_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [],
            variadic_result_var: [result_types],
            attrs_structured: [],
            params: [$($params)* result_types: impl IntoIterator<Item = $crate::arena::TypeRef>,],
        );
    };

    // Single result
    (@results
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        results: [$result:ident],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@attrs
            dialect: $dialect, op: $op,
            attr_tokens: [$($attr_tokens)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [result_ty],
            variadic_result_var: [],
            attrs_structured: [],
            params: [$($params)* result_ty: $crate::arena::TypeRef,],
        );
    };

    // Multi results
    (@results
        dialect: $dialect:ident, op: $op:ident,
        attrs: [$($attr_tokens:tt)*],
        results: [$result:ident, $($rest:ident),+],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::paste::paste! {
            $crate::arena_constructor!(@results
                dialect: $dialect, op: $op,
                attrs: [$($attr_tokens)*],
                results: [$($rest),+],
                regions: [$($region_tokens)*],
                fixed_ops: [$($fixed_ops)*],
                variadic_op: [$($variadic_op)*],
                params: [$($params)* [<$result _ty>]: $crate::arena::TypeRef,],
            );
        }
    };

    // ========================================================================
    // Phase 3: Collect attribute params + structured descriptors
    // ========================================================================

    // No attrs
    (@attrs
        dialect: $dialect:ident, op: $op:ident,
        attr_tokens: [],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@regions
            dialect: $dialect, op: $op,
            region_tokens: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)*],
            rs_items: [],
            params: [$($params)*],
        );
    };

    // Required attr with more
    (@attrs
        dialect: $dialect:ident, op: $op:ident,
        attr_tokens: [$name:ident : $ty:ident, $($rest:tt)*],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@attrs
            dialect: $dialect, op: $op,
            attr_tokens: [$($rest)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)* {required $name $ty}],
            params: [$($params)* $name: $crate::arena_attr_type_helper!(@rust_type $ty),],
        );
    };

    // Required attr (last)
    (@attrs
        dialect: $dialect:ident, op: $op:ident,
        attr_tokens: [$name:ident : $ty:ident],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@regions
            dialect: $dialect, op: $op,
            region_tokens: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)* {required $name $ty}],
            rs_items: [],
            params: [$($params)* $name: $crate::arena_attr_type_helper!(@rust_type $ty),],
        );
    };

    // Optional attr with more
    (@attrs
        dialect: $dialect:ident, op: $op:ident,
        attr_tokens: [$name:ident ?: $ty:ident, $($rest:tt)*],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@attrs
            dialect: $dialect, op: $op,
            attr_tokens: [$($rest)*],
            regions: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)* {optional $name $ty}],
            params: [$($params)* $name: Option<$crate::arena_attr_type_helper!(@rust_type $ty)>,],
        );
    };

    // Optional attr (last)
    (@attrs
        dialect: $dialect:ident, op: $op:ident,
        attr_tokens: [$name:ident ?: $ty:ident],
        regions: [$($region_tokens:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@regions
            dialect: $dialect, op: $op,
            region_tokens: [$($region_tokens)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)* {optional $name $ty}],
            rs_items: [],
            params: [$($params)* $name: Option<$crate::arena_attr_type_helper!(@rust_type $ty)>,],
        );
    };

    // ========================================================================
    // Phase 4: Collect region/successor params
    // ========================================================================

    // No regions → emit
    (@regions
        dialect: $dialect:ident, op: $op:ident,
        region_tokens: [],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        rs_items: [$($rs_items:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@emit
            dialect: $dialect, op: $op,
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)*],
            rs_items: [$($rs_items)*],
            params: [$($params)*],
        );
    };

    // Region
    (@regions
        dialect: $dialect:ident, op: $op:ident,
        region_tokens: [#[region($name:ident)] {} $($rest:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        rs_items: [$($rs_items:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@regions
            dialect: $dialect, op: $op,
            region_tokens: [$($rest)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)*],
            rs_items: [$($rs_items)* {region $name}],
            params: [$($params)* $name: $crate::arena::RegionRef,],
        );
    };

    // Successor
    (@regions
        dialect: $dialect:ident, op: $op:ident,
        region_tokens: [#[successor($name:ident)] $($rest:tt)*],
        fixed_ops: [$($fixed_ops:tt)*],
        variadic_op: [$($variadic_op:tt)*],
        result_var: [$($result_var:tt)*],
        variadic_result_var: [$($variadic_result_var:tt)*],
        attrs_structured: [$($attrs_s:tt)*],
        rs_items: [$($rs_items:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::arena_constructor!(@regions
            dialect: $dialect, op: $op,
            region_tokens: [$($rest)*],
            fixed_ops: [$($fixed_ops)*],
            variadic_op: [$($variadic_op)*],
            result_var: [$($result_var)*],
            variadic_result_var: [$($variadic_result_var)*],
            attrs_structured: [$($attrs_s)*],
            rs_items: [$($rs_items)* {successor $name}],
            params: [$($params)* $name: $crate::arena::BlockRef,],
        );
    };

    // ========================================================================
    // Final: Emit. All body code generated here from structured descriptors.
    // result_var/variadic_result_var carry tokens from the same expansion as
    // params, avoiding macro hygiene scope mismatches.
    // ========================================================================
    (@emit
        dialect: $dialect:ident, op: $op:ident,
        fixed_ops: [$($fixed_op:ident)*],
        variadic_op: [$($variadic_op:ident)?],
        result_var: [$($result_var:ident)?],
        variadic_result_var: [$($variadic_result_var:ident)?],
        attrs_structured: [$($as:tt)*],
        rs_items: [$($rs:tt)*],
        params: [$($params:tt)*],
    ) => {
        $crate::paste::paste! {
            #[allow(clippy::too_many_arguments)]
            pub fn $op(
                ctx: &mut $crate::arena::IrContext,
                location: $crate::arena::Location,
                $($params)*
            ) -> [<$op:camel>] {
                #[allow(unused_mut)]
                let mut __builder = $crate::arena::OperationDataBuilder::new(
                    location,
                    $crate::Symbol::new($crate::raw_ident_str!($dialect)),
                    $crate::Symbol::new($crate::raw_ident_str!($op)),
                );
                // Operands
                $( __builder = __builder.operand($fixed_op); )*
                $( __builder = __builder.operands($variadic_op); )?
                // Results (tokens from same expansion as params → same hygiene)
                $( __builder = __builder.result($result_var); )?
                $( __builder = __builder.results($variadic_result_var); )?
                // Attributes
                $( $crate::arena_constructor!(@emit_attr_item __builder, $as); )*
                // Regions/Successors
                $( $crate::arena_constructor!(@emit_rs_item __builder, $rs); )*

                let __data = __builder.build(ctx);
                let __op_ref = ctx.create_op(__data);
                [<$op:camel>](__op_ref)
            }
        }
    };

    // Attribute emit helpers
    (@emit_attr_item $builder:ident, {required $name:ident $ty:ident}) => {
        $builder = $builder.attr(
            $crate::Symbol::new($crate::raw_ident_str!($name)),
            $crate::arena_attr_type_helper!(@to_attr $ty, $name),
        );
    };
    (@emit_attr_item $builder:ident, {optional $name:ident $ty:ident}) => {
        if let ::core::option::Option::Some(__attr_val) = $name {
            $builder = $builder.attr(
                $crate::Symbol::new($crate::raw_ident_str!($name)),
                $crate::arena_attr_type_helper!(@to_attr $ty, __attr_val),
            );
        }
    };

    // Region/successor emit helpers
    (@emit_rs_item $builder:ident, {region $name:ident}) => {
        $builder = $builder.region($name);
    };
    (@emit_rs_item $builder:ident, {successor $name:ident}) => {
        $builder = $builder.successor($name);
    };
}
