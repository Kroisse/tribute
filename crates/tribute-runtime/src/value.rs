//! Tribute value representation and type system
//!
//! This module defines the runtime representation of Tribute values,
//! which must match the layout expected by the Cranelift compiler.
//! Uses Handle-based API with allocation table for true GC compatibility.

use dashmap::DashMap;
use std::{
    boxed::Box,
    fmt,
    string::String,
    sync::{
        LazyLock,
        atomic::{AtomicU32, Ordering},
    },
};

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

    /// Execute a function with access to the value this handle points to
    /// Returns None if the handle is null or invalid
    pub fn with_value<T>(&self, f: impl FnOnce(&TrValue) -> T) -> Option<T> {
        allocation_table().with_value(*self, f)
    }

    /// Get a reference to the value this handle points to
    /// SAFETY: Caller must ensure the handle is valid and the allocation table is initialized
    /// DEPRECATED: Use with_value() instead for safer access
    /// This method now redirects to with_value() to avoid unsafe transmute
    pub unsafe fn deref(&self) -> &TrValue {
        panic!("deref() is deprecated and unsafe. Use with_value() instead for safe access.");
    }
}

/// Global allocation table for managing TrValue instances and string memory
/// This provides the indirection needed for garbage collection
pub struct AllocationTable {
    /// Concurrent map from handle index to allocated TrValue
    /// Using Box to keep values on heap and allow for future relocation
    table: DashMap<u32, Box<TrValue>>,
    /// Concurrent map from string handle index to allocated string data
    /// All string memory is managed centrally for GC compatibility
    string_table: DashMap<u32, Box<[u8]>>,
    /// Atomic counter for generating unique handle indices
    next_index: AtomicU32,
    /// Atomic counter for generating unique string handle indices
    next_string_index: AtomicU32,
}

impl AllocationTable {
    /// Create a new allocation table
    fn new() -> Self {
        Self {
            table: DashMap::new(),
            string_table: DashMap::new(),
            next_index: AtomicU32::new(1), // Start at 1, reserve 0 for null
            next_string_index: AtomicU32::new(1), // Start at 1, reserve 0 for null
        }
    }

    /// Allocate a new value and return its handle
    pub fn allocate(&self, value: TrValue) -> TrHandle {
        let index = self.next_index.fetch_add(1, Ordering::Relaxed);
        self.table.insert(index, Box::new(value));
        TrHandle::from_index(index)
    }

    /// Free a handle and its associated value
    pub fn free(&self, handle: TrHandle) {
        if handle.is_null() {
            return;
        }

        self.table.remove(&handle.index);
    }

    /// Clone a value (deep copy)
    pub fn clone_value(&self, handle: TrHandle) -> TrHandle {
        if handle.is_null() {
            return TrHandle::null();
        }

        if let Some(value_ref) = self.table.get(&handle.index) {
            let cloned_value = value_ref.value().clone_value();
            self.allocate(cloned_value)
        } else {
            TrHandle::null()
        }
    }

    /// Get the tag of a value
    pub fn get_tag(&self, handle: TrHandle) -> u8 {
        if handle.is_null() {
            return ValueTag::Unit as u8;
        }

        if let Some(value_ref) = self.table.get(&handle.index) {
            value_ref.value().tag() as u8
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

        let left_val = self.table.get(&left.index);
        let right_val = self.table.get(&right.index);

        match (left_val, right_val) {
            (Some(l), Some(r)) => {
                let l_val = l.value().as_ref();
                let r_val = r.value().as_ref();

                match (l_val, r_val) {
                    (TrValue::Number(ln), TrValue::Number(rn)) => ln == rn,
                    (TrValue::String(ls), TrValue::String(rs)) => {
                        ls.with_string(|left_str| rs.with_string(|right_str| left_str == right_str))
                    }
                    (TrValue::Unit, TrValue::Unit) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Extract a number from a value
    pub fn to_number(&self, handle: TrHandle) -> f64 {
        if handle.is_null() {
            return 0.0;
        }

        if let Some(value_ref) = self.table.get(&handle.index) {
            value_ref.value().as_number()
        } else {
            0.0
        }
    }

    /// Allocate string data and return its handle index
    pub fn allocate_string(&self, data: Vec<u8>) -> u32 {
        let index = self.next_string_index.fetch_add(1, Ordering::Relaxed);
        self.string_table.insert(index, data.into_boxed_slice());
        index
    }

    /// Free string data by handle index
    pub fn free_string(&self, string_index: u32) {
        if string_index == 0 {
            return;
        }

        self.string_table.remove(&string_index);
    }

    /// Get string data by handle index with a closure to avoid lifetime issues
    pub fn with_string_data<T>(&self, string_index: u32, f: impl FnOnce(&[u8]) -> T) -> Option<T> {
        if string_index == 0 {
            return None;
        }

        let data_ref = self.string_table.get(&string_index)?;
        Some(f(data_ref.value()))
    }

    /// Access value data safely with a closure to avoid lifetime issues
    pub fn with_value<T>(&self, handle: TrHandle, f: impl FnOnce(&TrValue) -> T) -> Option<T> {
        if handle.is_null() {
            return None;
        }

        let value_ref = self.table.get(&handle.index)?;
        Some(f(value_ref.value().as_ref()))
    }

    /// Clear all allocations (for cleanup)
    pub fn clear(&self) {
        self.table.clear();
        self.string_table.clear();
        self.next_index.store(1, Ordering::Relaxed);
        self.next_string_index.store(1, Ordering::Relaxed);
    }

    /// Get the current number of allocated values (for debugging)
    #[allow(dead_code)]
    fn allocation_count(&self) -> usize {
        self.table.len()
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
/// Three-mode enum for optimal memory usage and performance
///
/// Note: TrString allows null bytes (\0) in string content, unlike C strings,
/// but requires valid UTF-8 encoding. This enables storing UTF-8 text with
/// embedded null characters while maintaining string safety.
#[repr(C)]
#[derive(Debug)]
pub enum TrString {
    /// Mode 1: Inline - strings ≤ 7 bytes stored directly
    Inline { data: [u8; 7], len: u8 },
    /// Mode 2: Static - compile-time strings in object file
    Static {
        offset: u32, // Offset in .rodata section
        len: u32,    // Use u32 instead of usize to keep size consistent
    },
    /// Mode 3: Heap - runtime strings in AllocationTable
    Heap {
        data_index: u32, // AllocationTable index
        len: u32,        // Use u32 instead of usize to keep size consistent
    },
}

impl TrString {
    /// Create an appropriate TrString from a regular String
    /// Supports null bytes (\0) within valid UTF-8 text
    pub fn new(s: String) -> Self {
        let bytes = s.as_bytes();
        let len = bytes.len();

        if len <= 7 {
            // Use inline mode for short strings
            let mut data = [0u8; 7];
            data[..len].copy_from_slice(bytes);
            TrString::Inline {
                data,
                len: len as u8,
            }
        } else {
            // Use heap mode for longer strings
            let data_index = allocation_table().allocate_string(s.into_bytes());
            TrString::Heap {
                data_index,
                len: len as u32,
            }
        }
    }

    /// Create an inline TrString from bytes (for strings ≤ 7 bytes)
    /// Supports null bytes (\0) within valid UTF-8 text
    pub fn new_inline(bytes: &[u8]) -> Self {
        assert!(bytes.len() <= 7, "Inline strings must be ≤ 7 bytes");
        let mut data = [0u8; 7];
        data[..bytes.len()].copy_from_slice(bytes);
        TrString::Inline {
            data,
            len: bytes.len() as u8,
        }
    }

    /// Create a static TrString with offset into .rodata section
    pub fn new_static(offset: u32, len: u32) -> Self {
        TrString::Static { offset, len }
    }

    /// Create a heap TrString from bytes
    /// Supports null bytes (\0) within valid UTF-8 text
    pub fn new_heap(bytes: &[u8]) -> Self {
        let len = bytes.len();
        let data_index = allocation_table().allocate_string(bytes.to_vec());
        TrString::Heap {
            data_index,
            len: len as u32,
        }
    }

    /// Create from a static str (currently uses heap allocation)
    /// Supports null bytes (\0) within valid UTF-8 text
    /// TODO: This will be optimized to use Static mode when compiler support is ready
    pub fn from_static(s: &'static str) -> Self {
        TrString::new(s.to_string())
    }

    /// Get the length of the string
    pub fn len(&self) -> usize {
        match self {
            TrString::Inline { len, .. } => *len as usize,
            TrString::Static { len, .. } => *len as usize,
            TrString::Heap { len, .. } => *len as usize,
        }
    }

    /// Check if string is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert back to Rust String (takes ownership)
    pub unsafe fn to_string(&self) -> String {
        match self {
            TrString::Inline { data, len } => {
                String::from_utf8_lossy(&data[..*len as usize]).into_owned()
            }
            TrString::Static { offset, len } => {
                if *len == 0 {
                    String::new()
                } else {
                    #[cfg(test)]
                    {
                        // For testing: access mock string table
                        let (ptr, len) = get_mock_static_string(*offset, *len as usize);
                        unsafe {
                            String::from_utf8_unchecked(
                                std::slice::from_raw_parts(ptr, len).to_vec(),
                            )
                        }
                    }
                    #[cfg(not(test))]
                    {
                        // In a real implementation, this would read from .rodata section
                        panic!(
                            "Static string conversion requires compiler integration (offset: {}, len: {})",
                            offset, len
                        );
                    }
                }
            }
            TrString::Heap { data_index, len } => allocation_table()
                .with_string_data(*data_index, |data| {
                    String::from_utf8_lossy(&data[..*len as usize]).into_owned()
                })
                .unwrap_or_default(),
        }
    }

    /// Execute a function with access to the string content without memory leaks
    /// This is the safe alternative to as_str() that doesn't leak memory
    pub fn with_string<T>(&self, f: impl FnOnce(&str) -> T) -> T {
        match self {
            TrString::Inline { data, len } => {
                let s = unsafe { std::str::from_utf8_unchecked(&data[..*len as usize]) };
                f(s)
            }
            TrString::Static { offset, len } => {
                if *len == 0 {
                    f("")
                } else {
                    #[cfg(test)]
                    {
                        let (ptr, len) = get_mock_static_string(*offset, *len as usize);
                        let s = unsafe {
                            std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len))
                        };
                        f(s)
                    }
                    #[cfg(not(test))]
                    {
                        panic!(
                            "Static string access requires compiler integration (offset: {}, len: {})",
                            offset, len
                        );
                    }
                }
            }
            TrString::Heap { data_index, len } => {
                if *data_index == 0 || *len == 0 {
                    f("")
                } else {
                    allocation_table()
                        .with_string_data(*data_index, |data| {
                            let s =
                                unsafe { std::str::from_utf8_unchecked(&data[..*len as usize]) };
                            f(s)
                        })
                        .expect("Invalid string data access")
                }
            }
        }
    }

    /// Get pointer and length for C FFI
    pub fn as_ptr_len(&self) -> (*const u8, usize) {
        match self {
            TrString::Inline { data, len } => (data.as_ptr(), *len as usize),
            TrString::Static { offset, len } => {
                // For testing: use a mock string table
                // In real compilation, this would access .rodata section
                if *len == 0 {
                    (b"".as_ptr(), 0)
                } else {
                    // For testing purposes, mock the static string access
                    #[cfg(test)]
                    {
                        let (ptr, actual_len) = get_mock_static_string(*offset, *len as usize);
                        if ptr.is_null() || actual_len == 0 {
                            (b"".as_ptr(), 0)
                        } else {
                            (ptr, actual_len)
                        }
                    }
                    #[cfg(not(test))]
                    {
                        // In a real implementation, this would return:
                        // let base_addr = get_rodata_base_address(); // from compiler
                        // ((base_addr + *offset) as *const u8, *len as usize)
                        panic!(
                            "Static string pointer access requires compiler integration (offset: {}, len: {})",
                            offset, len
                        );
                    }
                }
            }
            TrString::Heap { data_index, len } => {
                // WARNING: This is inherently unsafe for heap strings
                // The pointer returned here is only valid during the with_string_data closure
                // For proper C FFI, consider using a different API that copies the string data
                if *data_index == 0 || *len == 0 {
                    (b"".as_ptr(), 0)
                } else {
                    // UNSAFE: This creates a temporary pointer that may become invalid
                    // Only use this in controlled situations where the lifetime is managed
                    allocation_table()
                        .with_string_data(*data_index, |data| (data.as_ptr(), *len as usize))
                        .unwrap_or((b"".as_ptr(), 0))
                }
            }
        }
    }
}

impl Drop for TrString {
    fn drop(&mut self) {
        // Only heap strings need cleanup
        if let TrString::Heap { data_index, .. } = self {
            if *data_index != 0 {
                allocation_table().free_string(*data_index);
            }
        }
        // Inline and Static strings don't need cleanup
    }
}

// SAFETY: TrString is safe to send/sync as it uses indices or inline data
unsafe impl Send for TrString {}
unsafe impl Sync for TrString {}

// Ensure the layout is 12 bytes with the optimized structure
const _: () = {
    assert!(std::mem::size_of::<TrString>() == 12);
    assert!(std::mem::align_of::<TrString>() == 4);
};

/// Main Tribute value type - must match the layout in tribute-cranelift/src/types.rs
///
/// String values support null bytes (\0) within valid UTF-8 text, enabling
/// storage of UTF-8 strings with embedded null characters.
#[repr(C)]
pub enum TrValue {
    Number(f64),
    String(TrString),
    Unit,
}

impl TrValue {
    /// Create a new number value
    pub fn number(n: f64) -> Self {
        TrValue::Number(n)
    }

    /// Create a new string value
    pub fn string(s: String) -> Self {
        TrValue::String(TrString::new(s))
    }

    /// Create a new string value from static str
    pub fn string_static(s: &'static str) -> Self {
        TrValue::String(TrString::from_static(s))
    }

    /// Create a unit value
    pub fn unit() -> Self {
        TrValue::Unit
    }

    /// Get the value as a number (returns 0.0 if not a number)
    pub fn as_number(&self) -> f64 {
        match self {
            TrValue::Number(n) => *n,
            _ => 0.0,
        }
    }

    /// Execute a function with access to the string value if this is a string
    /// Returns None if this is not a string value
    pub fn with_string<T>(&self, f: impl FnOnce(&str) -> T) -> Option<T> {
        match self {
            TrValue::String(s) => Some(s.with_string(f)),
            _ => None,
        }
    }

    /// Check if this is a unit value
    pub fn is_unit(&self) -> bool {
        matches!(self, TrValue::Unit)
    }

    /// Clone the value (deep copy for strings)
    pub fn clone_value(&self) -> Self {
        match self {
            TrValue::Number(n) => TrValue::Number(*n),
            TrValue::String(s) => {
                let s_str = unsafe { s.to_string() };
                TrValue::String(TrString::new(s_str))
            }
            TrValue::Unit => TrValue::Unit,
        }
    }

    /// Get the tag of the value
    pub fn tag(&self) -> ValueTag {
        match self {
            TrValue::Number(_) => ValueTag::Number,
            TrValue::String(_) => ValueTag::String,
            TrValue::Unit => ValueTag::Unit,
        }
    }
}

impl fmt::Debug for TrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrValue::Number(n) => write!(f, "Number({})", n),
            TrValue::String(s) => s.with_string(|s_str| write!(f, "String({:?})", s_str)),
            TrValue::Unit => write!(f, "Unit"),
        }
    }
}

impl fmt::Display for TrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrValue::Number(n) => write!(f, "{}", n),
            TrValue::String(s) => s.with_string(|s_str| write!(f, "{}", s_str)),
            TrValue::Unit => write!(f, "()"),
        }
    }
}

// Ensure the layout matches what Cranelift expects
const _: () = {
    // For enums, Rust adds a discriminant tag (1 byte) + padding for alignment
    // Largest variant is TrString (12 bytes) + discriminant (1 byte) + padding
    // With 8-byte alignment, total size will be 24 bytes
    assert!(std::mem::size_of::<TrValue>() == 24);
    assert!(std::mem::align_of::<TrValue>() == 8);
};

/// Access to the global allocation table (for internal use)
pub(crate) fn allocation_table() -> &'static AllocationTable {
    &ALLOCATION_TABLE
}

#[cfg(test)]
/// Mock static string data for testing
static MOCK_STRING_TABLE: &[u8] = b"hello\0world\0test string\0static data\0";

#[cfg(test)]
/// Mock function to simulate .rodata string access for testing
fn get_mock_static_string(offset: u32, len: usize) -> (*const u8, usize) {
    let offset_usize = offset as usize;
    if offset_usize < MOCK_STRING_TABLE.len()
        && len > 0
        && offset_usize.saturating_add(len) <= MOCK_STRING_TABLE.len()
    {
        let ptr = unsafe { MOCK_STRING_TABLE.as_ptr().add(offset_usize) };
        // Extra safety check - make sure we don't read beyond our mock data
        let actual_len = std::cmp::min(len, MOCK_STRING_TABLE.len() - offset_usize);
        (ptr, actual_len)
    } else {
        (b"".as_ptr(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_value_creation() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        let num_val = TrValue::number(42.0);
        assert!(matches!(num_val, TrValue::Number(_)));
        assert_eq!(num_val.as_number(), 42.0);

        let str_val = TrValue::string("hello".to_string());
        assert!(matches!(str_val, TrValue::String(_)));
        assert_eq!(
            str_val.with_string(|s| s.to_string()),
            Some("hello".to_string())
        );

        let unit_val = TrValue::unit();
        assert!(matches!(unit_val, TrValue::Unit));
        assert!(unit_val.is_unit());
    }

    #[test]
    #[serial]
    fn test_value_equality() {
        use crate::memory::*;
        // Clear allocation table for test isolation
        allocation_table().clear();
        let num1 = tr_value_from_number(42.0);
        let num2 = tr_value_from_number(42.0);
        let num3 = tr_value_from_number(43.0);

        assert_eq!(tr_value_equals(num1, num2), 1);
        assert_eq!(tr_value_equals(num1, num3), 0);

        tr_value_free(num1);
        tr_value_free(num2);
        tr_value_free(num3);
    }

    #[test]
    #[serial]
    fn test_trstring_three_modes() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        // Test inline mode (≤ 7 bytes)
        let short_str = TrString::new("hello".to_string());
        assert!(matches!(short_str, TrString::Inline { .. }));
        assert_eq!(short_str.len(), 5);

        // Test heap mode (> 7 bytes)
        let long_str = TrString::new("this is a very long string".to_string());
        assert!(matches!(long_str, TrString::Heap { .. }));
        assert_eq!(long_str.len(), 26);

        // Test static mode
        let static_str = TrString::new_static(0, 5); // "hello"
        assert!(matches!(static_str, TrString::Static { .. }));
        assert_eq!(static_str.len(), 5);
    }

    #[test]
    #[serial]
    fn test_static_string_runtime_functions() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        // Clear allocation table before test
        allocation_table().clear();

        // Test creating static string value
        let handle = crate::memory::tr_value_from_static_string(0, 5); // "hello"
        assert!(!handle.is_null());

        // Test extracting string data
        let mut len = 0usize;
        let ptr = crate::memory::tr_string_as_ptr(handle, &mut len as *mut usize);
        assert!(!ptr.is_null());
        assert_eq!(len, 5);

        // Check the actual string content
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, len);
            assert_eq!(slice, b"hello");
        }

        // Cleanup
        crate::memory::tr_value_free(handle);
    }

    #[test]
    #[serial]
    fn test_static_string_different_offsets() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        // Clear allocation table before test
        allocation_table().clear();

        // Test different strings in mock table
        // "hello" at offset 0, length 5
        let handle1 = crate::memory::tr_value_from_static_string(0, 5);
        let mut len1 = 0usize;
        let ptr1 = crate::memory::tr_string_as_ptr(handle1, &mut len1 as *mut usize);
        unsafe {
            let slice1 = std::slice::from_raw_parts(ptr1, len1);
            assert_eq!(slice1, b"hello");
        }

        // "world" at offset 6, length 5 (after "hello\0")
        let handle2 = crate::memory::tr_value_from_static_string(6, 5);
        let mut len2 = 0usize;
        let ptr2 = crate::memory::tr_string_as_ptr(handle2, &mut len2 as *mut usize);
        unsafe {
            let slice2 = std::slice::from_raw_parts(ptr2, len2);
            assert_eq!(slice2, b"world");
        }

        // "test string" at offset 12, length 11 (after "hello\0world\0")
        let handle3 = crate::memory::tr_value_from_static_string(12, 11);
        let mut len3 = 0usize;
        let ptr3 = crate::memory::tr_string_as_ptr(handle3, &mut len3 as *mut usize);
        unsafe {
            let slice3 = std::slice::from_raw_parts(ptr3, len3);
            assert_eq!(slice3, b"test string");
        }

        // Cleanup
        crate::memory::tr_value_free(handle1);
        crate::memory::tr_value_free(handle2);
        crate::memory::tr_value_free(handle3);
    }

    #[test]
    #[serial]
    fn test_string_mode_automatic_selection() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        // Clear allocation table before test
        allocation_table().clear();

        // Test automatic mode selection for runtime strings
        // Short string should use inline mode
        let short_data = b"short";
        let handle_short =
            crate::memory::tr_value_from_string(short_data.as_ptr(), short_data.len());

        // Verify it's handled correctly
        let mut len = 0usize;
        let ptr = crate::memory::tr_string_as_ptr(handle_short, &mut len as *mut usize);
        assert_eq!(len, 5);
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, len);
            assert_eq!(slice, b"short");
        }

        // Long string should use heap mode
        let long_data = b"this is a very long string that exceeds seven characters";
        let handle_long = crate::memory::tr_value_from_string(long_data.as_ptr(), long_data.len());

        let mut len_long = 0usize;
        let ptr_long = crate::memory::tr_string_as_ptr(handle_long, &mut len_long as *mut usize);
        assert_eq!(len_long, long_data.len());
        unsafe {
            let slice_long = std::slice::from_raw_parts(ptr_long, len_long);
            assert_eq!(slice_long, long_data);
        }

        // Cleanup
        crate::memory::tr_value_free(handle_short);
        crate::memory::tr_value_free(handle_long);
    }

    #[test]
    #[serial]
    fn test_empty_static_string() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        // Test edge case: empty static string
        let handle = crate::memory::tr_value_from_static_string(0, 0);
        assert!(!handle.is_null());

        let mut len = 0usize;
        let ptr = crate::memory::tr_string_as_ptr(handle, &mut len as *mut usize);
        assert_eq!(len, 0);
        assert!(!ptr.is_null()); // Should point to empty string, not null

        // Cleanup
        crate::memory::tr_value_free(handle);
    }

    #[test]
    #[serial]
    fn test_static_string_value_comparison() {
        // Clear allocation table for test isolation
        allocation_table().clear();

        // Test that static strings work with value comparison
        let handle1 = crate::memory::tr_value_from_static_string(0, 5); // "hello"
        let handle2 = crate::memory::tr_value_from_static_string(0, 5); // same "hello"
        let handle3 = crate::memory::tr_value_from_static_string(6, 5); // "world"

        // Verify all handles are valid
        assert!(!handle1.is_null());
        assert!(!handle2.is_null());
        assert!(!handle3.is_null());

        // For now, just verify they're different handle values but same content
        assert_ne!(handle1.index, handle2.index); // Different handles

        // Verify content is the same
        let mut len1 = 0usize;
        let ptr1 = crate::memory::tr_string_as_ptr(handle1, &mut len1 as *mut usize);
        let mut len2 = 0usize;
        let ptr2 = crate::memory::tr_string_as_ptr(handle2, &mut len2 as *mut usize);

        // Debug: print values if assertion fails
        if len1 == 0 {
            println!("DEBUG: handle1 returned length 0, handle: {:?}", handle1);
        }
        if len2 == 0 {
            println!("DEBUG: handle2 returned length 0, handle: {:?}", handle2);
        }

        assert_eq!(len1, 5, "First handle should return length 5");
        assert_eq!(len2, 5, "Second handle should return length 5");
        unsafe {
            let slice1 = std::slice::from_raw_parts(ptr1, len1);
            let slice2 = std::slice::from_raw_parts(ptr2, len2);
            assert_eq!(slice1, slice2);
            assert_eq!(slice1, b"hello");
        }

        // Cleanup
        crate::memory::tr_value_free(handle1);
        crate::memory::tr_value_free(handle2);
        crate::memory::tr_value_free(handle3);
    }
}
