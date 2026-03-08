//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate provides IR definitions for the multi-level dialect system.

#![recursion_limit = "512"]

// === Salsa-independent primitives ===
pub mod symbol;

// === ADT layout computation ===
pub mod adt_layout;

// === Source location types ===
pub mod location;

// === Operation interface (purity, isolation) ===
pub mod op_interface;

// === Operation and type utilities ===
pub mod ops;

// === IR core structures ===
pub mod context;
pub mod refs;
pub mod types;

// === Dialect definitions ===
pub mod dialect;

// === IR infrastructure ===
pub mod printer;
pub mod rewrite;
pub mod transforms;
pub mod validation;
pub mod walk;

// === Dialect conversion utilities ===
pub mod conversion;

// === IR text format parser ===
pub mod parser;

// Re-export proc macro for arena dialect definitions
pub use trunk_ir_macros::arena_dialect;

// Re-export paste for use in macros
#[doc(hidden)]
pub use paste;

// Re-export smallvec for use in macros and external crates
pub use smallvec;

pub use location::{Span, Spanned};
pub use ops::ConversionError;
pub use symbol::{BlockId, IdVec, Symbol, SymbolVec};

pub use smallvec::smallvec as idvec;

// Re-exports from submodules (previously from arena/mod.rs)
pub use context::{
    BlockArgData, BlockData, IrContext, OperationData, OperationDataBuilder, RegionData, Use,
    ValueData,
};
pub use refs::{BlockRef, OpRef, PathRef, RegionRef, TypeRef, ValueDef, ValueRef};
pub use rewrite::Module;
pub use types::{Attribute, Location, PathInterner, TypeData, TypeDataBuilder, TypeInterner};
pub use walk::WalkAction;

/// Backward-compatible `arena` module.
///
/// All items have been moved to the crate root. This module re-exports
/// them so that existing `trunk_ir::arena::*` paths continue to work
/// during the transition.
pub mod arena {
    pub use crate::context;
    pub use crate::dialect;
    pub use crate::ops;
    pub use crate::printer;
    pub use crate::refs;
    pub use crate::rewrite;
    pub use crate::transforms;
    pub use crate::types;
    pub use crate::validation;
    pub use crate::walk;

    pub use crate::context::{
        BlockArgData, BlockData, IrContext, OperationData, OperationDataBuilder, RegionData, Use,
        ValueData,
    };
    pub use crate::refs::{BlockRef, OpRef, PathRef, RegionRef, TypeRef, ValueDef, ValueRef};
    pub use crate::rewrite::Module;
    pub use crate::types::{
        Attribute, Location, PathInterner, TypeData, TypeDataBuilder, TypeInterner,
    };
    pub use crate::walk::WalkAction;

    /// Re-export parser module for backward compatibility.
    pub mod parser {
        pub use crate::parser::builder::*;
    }
}
