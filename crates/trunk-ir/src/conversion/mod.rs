//! Dialect conversion utilities for TrunkIR.
//!
//! This module provides utilities for converting between dialects, including
//! resolution of `unrealized_conversion_cast` operations that may be inserted
//! during dialect conversion.
//!
//! # Overview
//!
//! During dialect conversion (e.g., lowering from high-level to low-level dialects),
//! type mismatches may occur between operations. The conversion infrastructure
//! inserts `core.unrealized_conversion_cast` operations as placeholders for these
//! mismatches. This module provides the resolution pass that eliminates these casts
//! by applying appropriate materialization functions.
//!
//! # Usage
//!
//! ```ignore
//! use trunk_ir::conversion::resolve_unrealized_casts;
//! use trunk_ir::rewrite::TypeConverter;
//!
//! let type_converter = TypeConverter::new()
//!     .add_materialization(|db, loc, value, from_ty, to_ty| {
//!         // Generate actual conversion operations
//!         MaterializeResult::single(my_cast_op)
//!     });
//!
//! let result = resolve_unrealized_casts(db, module, &type_converter)?;
//! println!("Resolved {} casts", result.resolved_count);
//! ```

mod resolve_unrealized_casts;

pub use resolve_unrealized_casts::{
    ResolveResult, UnresolvedCast, UnresolvedCastError, resolve_unrealized_casts,
};
