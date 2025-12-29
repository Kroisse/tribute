//! Tribute language-specific IR dialects.
//!
//! This crate provides dialects specific to the Tribute programming language,
//! built on top of the trunk-ir infrastructure.

pub mod dialect;

// Re-export common trunk-ir types for convenience
pub use trunk_ir::{
    Attribute, Attrs, Block, BlockBuilder, BlockId, ConversionError, DialectOp, DialectType, IdVec,
    Location, Operation, PathId, QualifiedName, Region, Span, Symbol, Type, Value, ValueDef, idvec,
};

// Re-export trunk_ir::register_pure_op for convenience
// Users can use:
//   use tribute_ir::register_pure_op;
//   register_pure_op!(crate::dialect::src::Var<'_>);
pub use trunk_ir::register_pure_op;
