//! Tribute value representation and type system
//!
//! This module defines the runtime representation of Tribute values,
//! which must match the layout expected by the Cranelift compiler.
//! Uses Handle-based API with allocation table for true GC compatibility.

use std::{boxed::Box, string::String, fmt, mem::ManuallyDrop, sync::{Mutex, LazyLock}, collections::HashMap};

/// Handle type for GC-compatible value references
/// Uses index-based approach with allocation table for true GC compatibility
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrHandle {
    /// Index into the global allocation table
    /// 0 is reserved as null handle
    pub index: u32,
}

impl TrHandle {
    /// Create a null handle (index 0)
    pub fn null() -> Self {
        TrHandle { index: 0 }
    }
    
    /// Check if this is a null handle
    pub fn is_null(&self) -> bool {
        self.index == 0
    }
    
    /// Create a handle from an index (internal use)
    pub(crate) fn from_index(index: u32) -> Self {
        TrHandle { index }
    }
    
    /// Get a reference to the value this handle points to
    /// SAFETY: Caller must ensure the handle is valid and the allocation table is initialized
    pub unsafe fn deref(&self) -> &TrValue {
        if self.is_null() {
            panic!("Attempted to dereference null handle");
        }
        ALLOCATION_TABLE.deref_handle(*self)
    }
}

/// Global allocation table for managing TrValue instances and string memory
/// This provides the indirection needed for garbage collection
pub struct AllocationTable {
    /// Map from handle index to allocated TrValue
    /// Using Box to keep values on heap and allow for future relocation
    table: Mutex<HashMap<u32, Box<TrValue>>>,
    /// Map from string handle index to allocated string data
    /// All string memory is managed centrally for GC compatibility
    string_table: Mutex<HashMap<u32, Box<[u8]>>>,
    /// Counter for generating unique handle indices
    next_index: Mutex<u32>,
    /// Counter for generating unique string handle indices
    next_string_index: Mutex<u32>,
}

impl AllocationTable {
    /// Create a new allocation table
    fn new() -> Self {
        Self {
            table: Mutex::new(HashMap::new()),
            string_table: Mutex::new(HashMap::new()),
            next_index: Mutex::new(1), // Start at 1, reserve 0 for null
            next_string_index: Mutex::new(1), // Start at 1, reserve 0 for null
        }
    }
    
    /// Allocate a new value and return its handle
    pub fn allocate(&self, value: TrValue) -> TrHandle {
        let mut table = self.table.lock().unwrap();
        let mut next_index = self.next_index.lock().unwrap();
        
        let index = *next_index;
        *next_index += 1;
        
        table.insert(index, Box::new(value));
        TrHandle::from_index(index)
    }
    
    /// Free a handle and its associated value
    pub fn free(&self, handle: TrHandle) {
        if handle.is_null() {
            return;
        }
        
        let mut table = self.table.lock().unwrap();
        table.remove(&handle.index);
    }
    
    /// Get a reference to the value for a handle
    /// SAFETY: Caller must ensure handle is valid
    unsafe fn deref_handle(&self, handle: TrHandle) -> &TrValue {
        let table = self.table.lock().unwrap();
        let value_box = table.get(&handle.index)
            .expect("Invalid handle: value not found in allocation table");
        
        // SAFETY: We're extending the lifetime of the reference here.
        // This is safe as long as:
        // 1. The caller doesn't hold the reference longer than the value's lifetime
        // 2. The allocation table is not modified while the reference is held
        // 3. This is used only for temporary access within C functions
        std::mem::transmute::<&TrValue, &TrValue>(value_box.as_ref())
    }
    
    /// Clone a value (deep copy)
    pub fn clone_value(&self, handle: TrHandle) -> TrHandle {
        if handle.is_null() {
            return TrHandle::null();
        }
        
        let cloned_value = {
            let table = self.table.lock().unwrap();
            if let Some(value_box) = table.get(&handle.index) {
                value_box.clone_value()
            } else {
                return TrHandle::null();
            }
        };
        self.allocate(cloned_value)
    }
    
    /// Get the tag of a value
    pub fn get_tag(&self, handle: TrHandle) -> u8 {
        if handle.is_null() {
            return ValueTag::Unit as u8;
        }
        
        let table = self.table.lock().unwrap();
        if let Some(value_box) = table.get(&handle.index) {
            value_box.tag as u8
        } else {
            ValueTag::Unit as u8
        }
    }
    
    /// Check if two values are equal
    pub fn values_equal(&self, left: TrHandle, right: TrHandle) -> bool {
        if left.is_null() && right.is_null() {
            return true;
        }
        if left.is_null() || right.is_null() {
            return false;
        }
        
        let table = self.table.lock().unwrap();
        let left_val = table.get(&left.index);
        let right_val = table.get(&right.index);
        
        match (left_val, right_val) {
            (Some(l), Some(r)) => {
                if l.tag != r.tag {
                    return false;
                }
                
                match l.tag {
                    ValueTag::Number => unsafe { l.data.number == r.data.number },
                    ValueTag::String => unsafe {
                        let left_str = l.data.string.as_str();
                        let right_str = r.data.string.as_str();
                        left_str == right_str
                    },
                    ValueTag::Unit => true,
                }
            },
            _ => false,
        }
    }
    
    /// Extract a number from a value
    pub fn to_number(&self, handle: TrHandle) -> f64 {
        if handle.is_null() {
            return 0.0;
        }
        
        let table = self.table.lock().unwrap();
        if let Some(value_box) = table.get(&handle.index) {
            value_box.as_number()
        } else {
            0.0
        }
    }
    
    /// Allocate string data and return its handle index
    pub fn allocate_string(&self, data: Vec<u8>) -> u32 {
        let mut string_table = self.string_table.lock().unwrap();
        let mut next_string_index = self.next_string_index.lock().unwrap();
        
        let index = *next_string_index;
        *next_string_index += 1;
        
        string_table.insert(index, data.into_boxed_slice());
        index
    }
    
    /// Free string data by handle index
    pub fn free_string(&self, string_index: u32) {
        if string_index == 0 {
            return;
        }
        
        let mut string_table = self.string_table.lock().unwrap();
        string_table.remove(&string_index);
    }
    
    /// Get string data by handle index
    pub fn get_string_data(&self, string_index: u32) -> Option<&[u8]> {
        if string_index == 0 {
            return None;
        }
        
        let string_table = self.string_table.lock().unwrap();
        let data = string_table.get(&string_index)?;
        
        // SAFETY: We're extending the lifetime of the reference here.
        // This is safe as long as:
        // 1. The caller doesn't hold the reference longer than the string's lifetime
        // 2. The string table is not modified while the reference is held
        // 3. This is used only for temporary access within C functions
        Some(unsafe { std::mem::transmute::<&[u8], &[u8]>(data.as_ref()) })
    }
    
    /// Clear all allocations (for cleanup)
    pub fn clear(&self) {
        let mut table = self.table.lock().unwrap();
        table.clear();
        let mut string_table = self.string_table.lock().unwrap();
        string_table.clear();
        let mut next_index = self.next_index.lock().unwrap();
        *next_index = 1;
        let mut next_string_index = self.next_string_index.lock().unwrap();
        *next_string_index = 1;
    }
    
    /// Get the current number of allocated values (for debugging)
    #[allow(dead_code)]
    fn allocation_count(&self) -> usize {
        let table = self.table.lock().unwrap();
        table.len()
    }
}

/// Global allocation table instance
static ALLOCATION_TABLE: LazyLock<AllocationTable> = LazyLock::new(AllocationTable::new);

/// Value type tags - must match ValueTag in tribute-cranelift/src/types.rs
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTag {
    Number = 0,
    String = 1,
    Unit = 2,
}

/// String representation for dynamic values
/// Uses allocation table index instead of direct pointers for GC compatibility
#[repr(C)]
#[derive(Debug)]
pub struct TrString {
    /// Index into the global string allocation table (0 = null/empty)
    pub data_index: u32,
    /// Length of the string in bytes
    pub len: usize,
    /// Whether this is a static string (doesn't need deallocation)
    pub is_static: bool,
    /// Padding to maintain C ABI compatibility
    pub _padding: [u8; 3],
}

impl TrString {
    pub fn new(s: String) -> Self {
        let len = s.len();
        if len == 0 {
            return TrString {
                data_index: 0,
                len: 0,
                is_static: false,
                _padding: [0; 3],
            };
        }
        
        let data_index = allocation_table().allocate_string(s.into_bytes());
        TrString {
            data_index,
            len,
            is_static: false,
            _padding: [0; 3],
        }
    }
    
    pub fn from_static(s: &'static str) -> Self {
        let len = s.len();
        if len == 0 {
            return TrString {
                data_index: 0,
                len: 0,
                is_static: true,
                _padding: [0; 3],
            };
        }
        
        // For static strings, we still allocate in the table for consistency
        // but mark them as static so they can be handled specially during GC
        let data_index = allocation_table().allocate_string(s.as_bytes().to_vec());
        TrString {
            data_index,
            len,
            is_static: true,
            _padding: [0; 3],
        }
    }
    
    /// Convert back to Rust String (takes ownership)
    pub unsafe fn to_string(&self) -> String {
        if self.data_index == 0 || self.len == 0 {
            return String::new();
        }
        
        if let Some(data) = allocation_table().get_string_data(self.data_index) {
            String::from_utf8_lossy(&data[..self.len]).into_owned()
        } else {
            String::new()
        }
    }
    
    /// Get a string slice (borrowing)
    pub unsafe fn as_str(&self) -> &str {
        if self.data_index == 0 || self.len == 0 {
            return "";
        }
        
        if let Some(data) = allocation_table().get_string_data(self.data_index) {
            std::str::from_utf8_unchecked(&data[..self.len])
        } else {
            ""
        }
    }
}

impl Drop for TrString {
    fn drop(&mut self) {
        // Free string data from allocation table when TrString is dropped
        // Static strings are left in the table for consistency but could be freed during GC
        if self.data_index != 0 && !self.is_static {
            allocation_table().free_string(self.data_index);
        }
    }
}

// SAFETY: TrString now uses indices instead of pointers
// All memory access goes through the thread-safe allocation table
unsafe impl Send for TrString {}
unsafe impl Sync for TrString {}

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

/// Access to the global allocation table (for internal use)
pub(crate) fn allocation_table() -> &'static AllocationTable {
    &ALLOCATION_TABLE
}