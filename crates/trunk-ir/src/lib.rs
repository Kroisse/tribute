//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate provides IR definitions for the multi-level dialect system.

#![recursion_limit = "512"]

// === ADT layout computation ===
pub mod adt_layout;

// === Arena-based IR ===
pub mod arena;

// === Dialect modules ===
pub mod dialect;

// === IR infrastructure ===
pub mod conversion;
pub mod ir;
pub mod location;
pub mod op_interface;
pub mod ops;
pub mod parser;
pub mod printer;
pub mod rewrite;
pub mod transforms;
pub mod type_interface;
pub mod types;
pub mod validation;
pub mod walk;

// Re-export proc macro for arena dialect definitions
pub use trunk_ir_macros::arena_dialect;

/// Internal macro for defining arena dialects within the trunk-ir crate.
#[doc(hidden)]
macro_rules! arena_dialect_internal {
    ($($tt:tt)*) => {
        crate::arena_dialect! {
            #[crate = crate]
            $($tt)*
        }
    };
}
pub(crate) use arena_dialect_internal;

// Re-export paste for use in macros (still used by Salsa dialect! macro)
#[doc(hidden)]
pub use paste;

// Re-export smallvec for use in macros and external crates
pub use smallvec;

pub use ir::{Block, BlockArg, BlockBuilder, BlockId, Operation, Region, Symbol, Value, ValueDef};
pub use location::{Location, PathId, Span, Spanned};
pub use ops::{ConversionError, DialectOp};
pub use types::{Attribute, Attrs, DialectType, Type};
pub use walk::{OperationWalk, WalkAction};

/// Small vector for values tracked by Salsa framework.
pub type IdVec<T> = smallvec::SmallVec<[T; 2]>;
pub use smallvec::smallvec as idvec;

/// Small vector for symbols.
pub type SymbolVec = smallvec::SmallVec<[Symbol; 4]>;
