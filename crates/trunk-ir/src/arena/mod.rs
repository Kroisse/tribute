//! Arena-based mutable IR.
//!
//! This module provides an alternative IR representation that uses arena
//! allocation (`cranelift-entity`) instead of Salsa interning. It enables
//! efficient in-place mutation, use-chains, and RAUW (replace all uses with).
//!
//! The arena IR coexists with the existing Salsa-based IR during the
//! transition period.

pub mod context;
pub mod dialect;
pub mod ops;
pub mod parser;
pub mod printer;
pub mod refs;
pub mod rewrite;
pub mod transforms;
pub mod types;
pub mod validation;
pub mod walk;

pub use context::{
    BlockArgData, BlockData, IrContext, OperationData, OperationDataBuilder, RegionData, Use,
    ValueData,
};
pub use refs::{BlockRef, OpRef, PathRef, RegionRef, TypeRef, ValueDef, ValueRef};
pub use rewrite::ArenaModule;
pub use types::{Attribute, Location, PathInterner, TypeData, TypeDataBuilder, TypeInterner};
pub use walk::WalkAction;
