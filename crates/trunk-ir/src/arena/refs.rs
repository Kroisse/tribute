//! Entity references for arena-based IR.
//!
//! Each ref type is a thin `u32` wrapper providing type-safe indexing
//! into `PrimaryMap` storage in `IrContext`.

use cranelift_entity::entity_impl;
use std::fmt;

/// Reference to an operation in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpRef(u32);
entity_impl!(OpRef, "op");

/// Reference to an SSA value in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueRef(u32);
entity_impl!(ValueRef, "v");

/// Reference to a basic block in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockRef(u32);
entity_impl!(BlockRef, "block");

/// Reference to a region (list of blocks) in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionRef(u32);
entity_impl!(RegionRef, "region");

/// Reference to an interned type in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeRef(u32);
entity_impl!(TypeRef, "ty");

/// Reference to an interned path string in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PathRef(u32);
entity_impl!(PathRef, "path");

/// Where a value is defined: either an operation result or a block argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueDef {
    /// Result of an operation at the given index.
    OpResult(OpRef, u32),
    /// Block argument at the given index.
    BlockArg(BlockRef, u32),
}

impl fmt::Display for ValueDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueDef::OpResult(op, idx) => write!(f, "{}#{}", op, idx),
            ValueDef::BlockArg(block, idx) => write!(f, "{}#{}", block, idx),
        }
    }
}
