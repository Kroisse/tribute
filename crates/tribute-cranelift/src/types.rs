//! Type system for Tribute → Cranelift compilation
//!
//! This module defines how Tribute's dynamic types are represented
//! in Cranelift IR. We use a tagged union approach for runtime values.

use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::ir::types::{I8, I32, I64, Type};
use tribute_core::{Db, TargetInfo};

/// Cranelift types used in Tribute compilation
pub struct TributeTypes;

impl TributeTypes {
    /// Get the pointer type for the target platform
    pub fn pointer_type(db: &dyn Db, target: &TargetInfo) -> Type {
        match target.pointer_size(db) {
            4 => I32,
            8 => I64,
            _ => I64, // Default to 64-bit
        }
    }

    /// Get the type for value tags (discriminant)
    pub fn tag_type() -> Type {
        I8 // Single byte for type tags
    }

    /// Get the type for numbers (f64 internally)
    pub fn number_type() -> Type {
        cranelift_codegen::ir::types::F64
    }

    /// Get the type for string length/capacity
    pub fn size_type(db: &dyn Db, target: &TargetInfo) -> Type {
        // Use native integer size for string lengths
        match target.native_int_size(db) {
            4 => I32,
            8 => I64,
            _ => I64, // Default to 64-bit
        }
    }

    /// Get the ABI parameter for a Tribute value handle (TrHandle)
    pub fn value_param(db: &dyn Db, target: &TargetInfo) -> AbiParam {
        AbiParam::new(Self::pointer_type(db, target))
    }

    /// Get the ABI parameter for a raw number
    pub fn number_param() -> AbiParam {
        AbiParam::new(Self::number_type())
    }
}

/// Runtime value tags
#[repr(u8)]
pub enum ValueTag {
    Number = 0,
    String = 1,
    Unit = 2,
}

/// Runtime value layout (matches what the runtime library expects)
///
/// The TrValue enum is defined in the runtime crate with this layout:
/// - Number variant: discriminant + 8 bytes (f64)
/// - String variant: discriminant + 12 bytes (TrString enum)
/// - Unit variant: discriminant only
///
/// Total size: 24 bytes with 8-byte alignment
pub struct ValueLayout;

impl ValueLayout {
    /// Offset of the discriminant field (enum tag)
    pub const DISCRIMINANT_OFFSET: i32 = 0;

    /// Offset of the data field (after discriminant and padding)
    pub const DATA_OFFSET: i32 = 8; // After discriminant + padding for alignment

    /// Total size of a TrValue enum
    pub const VALUE_SIZE: i32 = 24; // discriminant(1) + padding + largest_variant(12) with 8-byte alignment
}
