//! Tribute value representation and type system
//!
//! This module defines the runtime representation of Tribute values,
//! which must match the layout expected by the Cranelift compiler.
//! Uses Handle-based API for future GC compatibility.

use std::{boxed::Box, string::String, fmt, mem::ManuallyDrop};

/// Handle type for GC-compatible value references
/// In a GC system, handles can be updated when objects move
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrHandle {
    /// For now, this is just a raw pointer, but in a GC system
    /// this would be an index into a handle table
    pub raw: *mut TrValue,
}

impl TrHandle {
    pub fn null() -> Self {
        TrHandle { raw: std::ptr::null_mut() }
    }
    
    pub fn is_null(&self) -> bool {
        self.raw.is_null()
    }
    
    pub fn from_raw(ptr: *mut TrValue) -> Self {
        TrHandle { raw: ptr }
    }
    
    /// SAFETY: Caller must ensure the handle is valid
    pub unsafe fn deref(&self) -> &TrValue {
        &*self.raw
    }
}

/// Value type tags - must match ValueTag in tribute-cranelift/src/types.rs
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTag {
    Number = 0,
    String = 1,
    Unit = 2,
}

/// String representation for dynamic values
#[repr(C)]
#[derive(Debug)]
pub struct TrString {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl TrString {
    pub fn new(s: String) -> Self {
        let mut s = s.into_boxed_str();
        let data = s.as_mut_ptr();
        let len = s.len();
        let capacity = len;
        
        // Prevent the Box from being dropped
        Box::leak(s);
        
        TrString { data, len, capacity }
    }
    
    pub fn from_static(s: &'static str) -> Self {
        TrString {
            data: s.as_ptr() as *mut u8,
            len: s.len(),
            capacity: s.len(),
        }
    }
    
    /// Convert back to Rust String (takes ownership)
    pub unsafe fn to_string(&self) -> String {
        if self.data.is_null() || self.len == 0 {
            return String::new();
        }
        
        let slice = std::slice::from_raw_parts(self.data, self.len);
        String::from_utf8_lossy(slice).into_owned()
    }
    
    /// Get a string slice (borrowing)
    pub unsafe fn as_str(&self) -> &str {
        if self.data.is_null() || self.len == 0 {
            return "";
        }
        
        let slice = std::slice::from_raw_parts(self.data, self.len);
        std::str::from_utf8_unchecked(slice)
    }
}

impl Drop for TrString {
    fn drop(&mut self) {
        if !self.data.is_null() && self.capacity > 0 {
            // Only drop if this was allocated (not static)
            unsafe {
                let _ = Box::from_raw(std::slice::from_raw_parts_mut(self.data, self.capacity));
            }
        }
    }
}

/// Main Tribute value type - must match the layout in tribute-cranelift/src/types.rs
#[repr(C)]
pub struct TrValue {
    pub tag: ValueTag,
    pub _padding: [u8; 7], // Padding to align data to 8 bytes
    pub data: TrValueData,
}

/// Value data union
#[repr(C)]
pub union TrValueData {
    pub number: f64,
    pub string: ManuallyDrop<TrString>,
    pub unit: (), // Zero-sized for unit values
}

impl TrValue {
    /// Create a new number value
    pub fn number(n: f64) -> Self {
        TrValue {
            tag: ValueTag::Number,
            _padding: [0; 7],
            data: TrValueData { number: n },
        }
    }
    
    /// Create a new string value
    pub fn string(s: String) -> Self {
        TrValue {
            tag: ValueTag::String,
            _padding: [0; 7],
            data: TrValueData { 
                string: ManuallyDrop::new(TrString::new(s))
            },
        }
    }
    
    /// Create a new string value from static str
    pub fn string_static(s: &'static str) -> Self {
        TrValue {
            tag: ValueTag::String,
            _padding: [0; 7],
            data: TrValueData { 
                string: ManuallyDrop::new(TrString::from_static(s))
            },
        }
    }
    
    /// Create a unit value
    pub fn unit() -> Self {
        TrValue {
            tag: ValueTag::Unit,
            _padding: [0; 7],
            data: TrValueData { unit: () },
        }
    }
    
    /// Get the value as a number (returns 0.0 if not a number)
    pub fn as_number(&self) -> f64 {
        match self.tag {
            ValueTag::Number => unsafe { self.data.number },
            _ => 0.0,
        }
    }
    
    /// Get the value as a string reference
    pub fn as_string(&self) -> Option<&str> {
        match self.tag {
            ValueTag::String => unsafe { 
                Some(self.data.string.as_str())
            },
            _ => None,
        }
    }
    
    /// Check if this is a unit value
    pub fn is_unit(&self) -> bool {
        matches!(self.tag, ValueTag::Unit)
    }
    
    /// Clone the value (deep copy for strings)
    pub fn clone_value(&self) -> Self {
        match self.tag {
            ValueTag::Number => TrValue::number(unsafe { self.data.number }),
            ValueTag::String => {
                let s = unsafe { self.data.string.to_string() };
                TrValue::string(s)
            },
            ValueTag::Unit => TrValue::unit(),
        }
    }
}

impl fmt::Debug for TrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.tag {
            ValueTag::Number => write!(f, "Number({})", unsafe { self.data.number }),
            ValueTag::String => {
                let s = unsafe { self.data.string.as_str() };
                write!(f, "String({:?})", s)
            },
            ValueTag::Unit => write!(f, "Unit"),
        }
    }
}

impl fmt::Display for TrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.tag {
            ValueTag::Number => write!(f, "{}", unsafe { self.data.number }),
            ValueTag::String => {
                let s = unsafe { self.data.string.as_str() };
                write!(f, "{}", s)
            },
            ValueTag::Unit => write!(f, "()"),
        }
    }
}

// Ensure the layout matches what Cranelift expects
const _: () = {
    assert!(std::mem::size_of::<TrValue>() == 32); // tag(1) + padding(7) + data(24)
    assert!(std::mem::align_of::<TrValue>() == 8);
};