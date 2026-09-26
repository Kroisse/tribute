//! Dialect conversion utilities for TrunkIR.
//!
//! Dialect conversion inserts `core.unrealized_conversion_cast` operations as
//! placeholders where a value's type and the type its uses declare disagree.
//! This module materializes them into real conversion operations and
//! reconciles the ones that fold away.
//!
//! # Usage
//!
//! ```ignore
//! use trunk_ir::conversion::{convert_unrealized_casts, reconcile_unrealized_casts};
//!
//! // Tail of the target type conversion.
//! convert_unrealized_casts(&mut ctx, module, &target_converter);
//! // Converter-free cleanup; a remaining cast is rejected by the target's
//! // legality boundary.
//! reconcile_unrealized_casts(&mut ctx, module);
//! ```

mod unrealized_casts;

pub use unrealized_casts::{
    UnrealizedCastConversionPattern, convert_unrealized_casts, materialize_unrealized_casts,
    reconcile_unrealized_casts,
};
