//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate provides IR definitions for the multi-level dialect system.

#![recursion_limit = "512"]

// Self-alias so proc-macro–generated `::trunk_ir::...` paths resolve
// inside this crate the same way they do for downstream consumers.
extern crate self as trunk_ir;

// === Salsa-independent primitives ===
pub mod symbol;

// === ADT layout computation ===
pub mod adt_layout;

// === Diagnostic types ===
pub mod diagnostic;

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

// === IR mapping for clone/transform ===
pub mod ir_mapping;

// === IR infrastructure ===
pub mod analysis;
pub mod printer;
pub mod rewrite;
pub mod transforms;
pub mod validation;
pub mod walk;

// === Dialect conversion utilities ===
pub mod conversion;

// === IR text format parser ===
pub mod parser;

// Re-export proc macro for dialect definitions
pub use trunk_ir_macros::dialect;

/// Deprecated alias for [`dialect`].
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
pub use ir_mapping::IrMapping;
pub use refs::{BlockRef, OpRef, PathRef, RegionRef, TypeRef, ValueDef, ValueRef};
pub use rewrite::Module;
pub use types::{Attribute, Location, PathInterner, TypeData, TypeDataBuilder, TypeInterner};
pub use walk::WalkAction;
