//! Type system for Tribute → Cranelift compilation
//!
//! This module defines how Tribute's dynamic types are represented
//! in Cranelift IR. We use a tagged union approach for runtime values.

use cranelift_codegen::ir::types::{Type, I64, I8};
use cranelift_codegen::ir::AbiParam;

/// Cranelift types used in Tribute compilation
pub struct TributeTypes;

impl TributeTypes {
    /// Get the pointer type for the current platform
    pub fn pointer_type() -> Type {
        I64  // 64-bit pointers for now, should be platform-specific
    }
    
    /// Get the type for value tags (discriminant)
    pub fn tag_type() -> Type {
        I8  // Single byte for type tags
    }
    
    /// Get the type for numbers (f64 internally)
    pub fn number_type() -> Type {
        cranelift_codegen::ir::types::F64
    }
    
    /// Get the type for string length/capacity
    pub fn size_type() -> Type {
        I64
    }
    
    /// Get the ABI parameter for a Tribute value handle (TrHandle)
    pub fn value_param() -> AbiParam {
        AbiParam::new(Self::pointer_type())
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
/// Now uses Rust enum layout:
/// ```rust
/// #[repr(C)]
/// enum TrValue {
///     Number(f64),      // discriminant + 8 bytes
///     String(TrString), // discriminant + 12 bytes
///     Unit,             // discriminant only
/// }
/// ```
pub struct ValueLayout;

impl ValueLayout {
    /// Offset of the discriminant field (enum tag)
    pub const DISCRIMINANT_OFFSET: i32 = 0;
    
    /// Offset of the data field (after discriminant and padding)
    pub const DATA_OFFSET: i32 = 8;  // After discriminant + padding for alignment
    
    /// Total size of a TrValue enum
    pub const VALUE_SIZE: i32 = 24;  // discriminant(1) + padding + largest_variant(12) with 8-byte alignment
}