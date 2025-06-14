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
/// ```c
/// struct TrValue {
///     uint8_t tag;
///     uint8_t _padding[7];
///     union {
///         double number;
///         struct {
///             char* data;
///             size_t len;
///             size_t capacity;
///         } string;
///     } data;
/// };
/// ```
pub struct ValueLayout;

impl ValueLayout {
    /// Offset of the tag field
    pub const TAG_OFFSET: i32 = 0;
    
    /// Offset of the data union
    pub const DATA_OFFSET: i32 = 8;  // After tag + padding
    
    /// Offset of string data pointer (within data union)
    pub const STRING_DATA_OFFSET: i32 = Self::DATA_OFFSET;
    
    /// Offset of string length (within data union)
    pub const STRING_LEN_OFFSET: i32 = Self::DATA_OFFSET + 8;
    
    /// Offset of string capacity (within data union)
    pub const STRING_CAPACITY_OFFSET: i32 = Self::DATA_OFFSET + 16;
    
    /// Total size of a TrValue
    pub const VALUE_SIZE: i32 = 32;  // tag(1) + padding(7) + union(24)
}