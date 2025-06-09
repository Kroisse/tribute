//! Handle-based memory management for the Tribute runtime
//!
//! This module provides an indirection layer between C FFI and TributeBoxed values,
//! making it easier to migrate to mark-and-sweep GC in the future.
//!
//! ## Value Interning
//!
//! The following values are interned (always return the same handle):
//! - `true` → Handle(1)
//! - `false` → Handle(2)
//! - `nil` → Handle(3)
//! - `""` (empty string) → Handle(4)
//!
//! Interned values have special properties:
//! - They are allocated once at first use and never deallocated
//! - Reference counting is bypassed (always reports ref count of 1)
//! - They survive `tribute_handle_clear_all()` operations
//! - Multiple requests for the same value return the same handle

use crate::value::{TributeBoxed, TributeValue};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

/// An opaque handle that indirectly references a TributeBoxed value
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TributeHandle(u64);

/// Invalid handle constant
pub const TRIBUTE_HANDLE_INVALID: TributeHandle = TributeHandle(0);

/// Global handle table that maps handles to TributeBoxed values
static HANDLE_TABLE: LazyLock<DashMap<u64, Box<TributeBoxed>>> = LazyLock::new(|| {
    use crate::interned_string::TributeString;
    
    // Pre-insert interned values
    [
        (INTERNED_TRUE.0, Box::new(TributeBoxed::new(TributeValue::Boolean(true)))),
        (INTERNED_FALSE.0, Box::new(TributeBoxed::new(TributeValue::Boolean(false)))),
        (INTERNED_NIL.0, Box::new(TributeBoxed::new(TributeValue::Nil))),
        (INTERNED_EMPTY_STRING.0, Box::new(TributeBoxed::new(TributeValue::String(TributeString::from_str(""))))),
    ]
    .into_iter()
    .collect()
});

/// Global handle counter for generating unique handles
static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(5); // Start at 5 to reserve 1-4 for interned values

/// Interned handle constants
pub const INTERNED_TRUE: TributeHandle = TributeHandle(1);
pub const INTERNED_FALSE: TributeHandle = TributeHandle(2);
pub const INTERNED_NIL: TributeHandle = TributeHandle(3);
pub const INTERNED_EMPTY_STRING: TributeHandle = TributeHandle(4);

/// Statistics for handle management
static HANDLE_STATS: LazyLock<Mutex<HandleStats>> = LazyLock::new(Mutex::default);

#[derive(Debug)]
struct HandleStats {
    allocated: u64,
    deallocated: u64,
    peak_count: u64,
}

impl HandleStats {
    const fn new() -> Self {
        Self {
            allocated: 0,
            deallocated: 0,
            peak_count: 0,
        }
    }
}

impl Default for HandleStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a new handle for a TributeBoxed value (public function for other modules)
pub fn create_handle(boxed: TributeBoxed) -> TributeHandle {
    TributeHandle::new(boxed)
}

impl TributeHandle {
    /// Create a new handle for a TributeBoxed value
    pub fn new(boxed: TributeBoxed) -> Self {
        let handle_id = HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let handle = TributeHandle(handle_id);

        // Store the boxed value in the handle table
        HANDLE_TABLE.insert(handle_id, Box::new(boxed));

        // Update statistics
        {
            let mut stats = HANDLE_STATS.lock().unwrap();
            stats.allocated += 1;
            let current_count = stats.allocated - stats.deallocated;
            if current_count > stats.peak_count {
                stats.peak_count = current_count;
            }
        }

        handle
    }

    /// Check if this handle is valid
    pub fn is_valid(&self) -> bool {
        if self.0 == 0 {
            return false;
        }

        HANDLE_TABLE.contains_key(&self.0)
    }

    /// Execute a closure with access to the TributeBoxed value
    pub fn with_value<T, F>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&TributeBoxed) -> T,
    {
        if self.0 == 0 {
            return None;
        }

        HANDLE_TABLE.get(&self.0).map(|boxed| f(&boxed))
    }

    /// Execute a closure with mutable access to the TributeBoxed value
    pub fn with_value_mut<T, F>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&mut TributeBoxed) -> T,
    {
        if self.0 == 0 {
            return None;
        }

        HANDLE_TABLE.get_mut(&self.0).map(|mut boxed| f(&mut boxed))
    }

    /// Release this handle and deallocate the associated value
    pub fn release(self) {
        if self.0 == 0 {
            return;
        }

        // Never deallocate interned values
        if self.0 >= 1 && self.0 <= 4 {
            return;
        }

        if HANDLE_TABLE.remove(&self.0).is_some() {
            let mut stats = HANDLE_STATS.lock().unwrap();
            stats.deallocated += 1;
        }
    }
}

// C FFI functions for handle management

/// Create a new handle for a number value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_new_number(value: i64) -> TributeHandle {
    let boxed = TributeBoxed::new(TributeValue::Number(value));
    TributeHandle::new(boxed)
}

/// Create a new handle for a boolean value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_new_boolean(value: bool) -> TributeHandle {
    // Use interned handles for true/false
    if value { INTERNED_TRUE } else { INTERNED_FALSE }
}

/// Create a new handle for a nil value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_new_nil() -> TributeHandle {
    // Use interned handle for nil
    INTERNED_NIL
}

/// Create a new handle for a string value from raw data
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_new_string(data: *const u8, length: usize) -> TributeHandle {
    use crate::interned_string::TributeString;
    
    // Check for empty string
    if length == 0 {
        return INTERNED_EMPTY_STRING;
    }
    
    // Create string from byte slice
    let bytes = if data.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, length) }
    };
    
    let tribute_string = TributeString::from_bytes(bytes);
    
    // Check if this resulted in an empty string (should be rare due to above check)
    if tribute_string.is_empty() {
        return INTERNED_EMPTY_STRING;
    }
    
    let boxed = TributeBoxed::new(TributeValue::String(tribute_string));
    TributeHandle::new(boxed)
}

/// Create a new handle for a string value from a Rust string slice
pub fn tribute_handle_new_string_from_str(s: &str) -> TributeHandle {
    use crate::interned_string::TributeString;
    
    if s.is_empty() {
        return INTERNED_EMPTY_STRING;
    }
    
    let tribute_string = TributeString::from_str(s);
    let boxed = TributeBoxed::new(TributeValue::String(tribute_string));
    TributeHandle::new(boxed)
}

/// Check if a handle is valid
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_is_valid(handle: TributeHandle) -> bool {
    handle.is_valid()
}

/// Get the type of the value referenced by a handle
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_get_type(handle: TributeHandle) -> u8 {
    handle
        .with_value(|boxed| boxed.value.type_id())
        .unwrap_or(TributeValue::TYPE_NIL)
}

/// Unbox a number value from a handle
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_unbox_number(handle: TributeHandle) -> i64 {
    handle
        .with_value(|boxed| {
            match &boxed.value {
                TributeValue::Number(n) => *n,
                _ => 0, // Type error
            }
        })
        .unwrap_or(0)
}

/// Unbox a boolean value from a handle
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_unbox_boolean(handle: TributeHandle) -> bool {
    handle
        .with_value(|boxed| {
            match &boxed.value {
                TributeValue::Boolean(b) => *b,
                _ => false, // Type error
            }
        })
        .unwrap_or(false)
}

/// Get string data from a handle (returns length, caller must copy data)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_get_string_length(handle: TributeHandle) -> usize {
    handle
        .with_value(|boxed| {
            match &boxed.value {
                TributeValue::String(tribute_string) => tribute_string.len(),
                _ => 0, // Type error
            }
        })
        .unwrap_or(0)
}

/// Copy string data from a handle to a buffer
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_copy_string_data(
    handle: TributeHandle,
    buffer: *mut u8,
    buffer_size: usize,
) -> usize {
    handle
        .with_value(|boxed| {
            match &boxed.value {
                TributeValue::String(tribute_string) => {
                    let bytes = tribute_string.as_bytes();
                    let copy_len = bytes.len().min(buffer_size);
                    
                    if !buffer.is_null() && copy_len > 0 {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                buffer,
                                copy_len
                            );
                        }
                    }
                    
                    bytes.len() // Return actual string length
                }
                _ => 0, // Type error
            }
        })
        .unwrap_or(0)
}

/// Add two number handles
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_add_numbers(
    lhs: TributeHandle,
    rhs: TributeHandle,
) -> TributeHandle {
    let lhs_val = lhs
        .with_value(|boxed| match &boxed.value {
            TributeValue::Number(n) => Some(*n),
            _ => None,
        })
        .flatten();

    let rhs_val = rhs
        .with_value(|boxed| match &boxed.value {
            TributeValue::Number(n) => Some(*n),
            _ => None,
        })
        .flatten();

    match (lhs_val, rhs_val) {
        (Some(a), Some(b)) => tribute_handle_new_number(a + b),
        _ => tribute_handle_new_number(0), // Type error
    }
}

/// Retain a handle (increment reference count)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_retain(handle: TributeHandle) -> TributeHandle {
    // Interned values don't need reference counting
    if handle.0 >= 1 && handle.0 <= 4 {
        return handle;
    }

    if handle.is_valid() {
        handle.with_value(|boxed| boxed.retain());
    }
    handle
}

/// Release a handle (decrement reference count and potentially deallocate)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_release(handle: TributeHandle) {
    // Never release interned values
    if handle.0 >= 1 && handle.0 <= 4 {
        return;
    }

    // Check if we should deallocate
    let should_deallocate = handle
        .with_value(|boxed| boxed.release() == 0)
        .unwrap_or(false);

    if should_deallocate {
        handle.release();
    }
}

/// Get the reference count for a handle
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_get_ref_count(handle: TributeHandle) -> usize {
    // Interned values have infinite reference count (represented as 1)
    if handle.0 >= 1 && handle.0 <= 4 {
        return 1;
    }

    handle.with_value(|boxed| boxed.ref_count()).unwrap_or(0)
}

/// Get handle management statistics
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_get_stats(
    allocated: *mut u64,
    deallocated: *mut u64,
    peak_count: *mut u64,
) {
    if allocated.is_null() || deallocated.is_null() || peak_count.is_null() {
        return;
    }

    unsafe {
        let stats = HANDLE_STATS.lock().unwrap();
        *allocated = stats.allocated;
        *deallocated = stats.deallocated;
        *peak_count = stats.peak_count;
    }
}

/// Clear all handles (for testing/cleanup)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_clear_all() {
    // Remove all handles except interned ones
    let interned_keys = [1, 2, 3, 4];
    HANDLE_TABLE.retain(|k, _| interned_keys.contains(k));

    let mut stats = HANDLE_STATS.lock().unwrap();
    stats.deallocated = stats.allocated.saturating_sub(4); // Keep 4 interned values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_creation_and_validity() {
        let handle = tribute_handle_new_number(42);
        assert_ne!(handle, TRIBUTE_HANDLE_INVALID);
        assert!(tribute_handle_is_valid(handle));

        tribute_handle_release(handle);
    }

    #[test]
    fn test_handle_number_operations() {
        let h1 = tribute_handle_new_number(10);
        let h2 = tribute_handle_new_number(5);

        assert_eq!(tribute_handle_unbox_number(h1), 10);
        assert_eq!(tribute_handle_unbox_number(h2), 5);

        let sum_handle = tribute_handle_add_numbers(h1, h2);
        assert_eq!(tribute_handle_unbox_number(sum_handle), 15);

        tribute_handle_release(h1);
        tribute_handle_release(h2);
        tribute_handle_release(sum_handle);
    }

    #[test]
    fn test_handle_boolean_operations() {
        let h_true = tribute_handle_new_boolean(true);
        let h_false = tribute_handle_new_boolean(false);

        assert!(tribute_handle_unbox_boolean(h_true));
        assert!(!tribute_handle_unbox_boolean(h_false));

        // Interned values should be valid after release
        tribute_handle_release(h_true);
        tribute_handle_release(h_false);

        assert!(h_true.is_valid());
        assert!(h_false.is_valid());
    }

    #[test]
    fn test_handle_type_checking() {
        let num_handle = tribute_handle_new_number(123);
        let bool_handle = tribute_handle_new_boolean(true);
        let nil_handle = tribute_handle_new_nil();

        assert_eq!(
            tribute_handle_get_type(num_handle),
            TributeValue::TYPE_NUMBER
        );
        assert_eq!(
            tribute_handle_get_type(bool_handle),
            TributeValue::TYPE_BOOLEAN
        );
        assert_eq!(tribute_handle_get_type(nil_handle), TributeValue::TYPE_NIL);

        tribute_handle_release(num_handle);
        tribute_handle_release(bool_handle);
        tribute_handle_release(nil_handle);

        // Boolean and nil handles should remain valid after release (interned)
        assert!(bool_handle.is_valid());
        assert!(nil_handle.is_valid());
    }

    #[test]
    fn test_invalid_handle() {
        assert!(!tribute_handle_is_valid(TRIBUTE_HANDLE_INVALID));
        assert_eq!(tribute_handle_unbox_number(TRIBUTE_HANDLE_INVALID), 0);
        assert!(!tribute_handle_unbox_boolean(TRIBUTE_HANDLE_INVALID));
        assert_eq!(
            tribute_handle_get_type(TRIBUTE_HANDLE_INVALID),
            TributeValue::TYPE_NIL
        );
    }

    #[test]
    fn test_reference_counting() {
        let handle = tribute_handle_new_number(42);
        assert_eq!(tribute_handle_get_ref_count(handle), 1);

        tribute_handle_retain(handle);
        assert_eq!(tribute_handle_get_ref_count(handle), 2);

        tribute_handle_release(handle);
        assert_eq!(tribute_handle_get_ref_count(handle), 1);

        tribute_handle_release(handle);
        assert!(!tribute_handle_is_valid(handle));
    }

    #[test]
    fn test_interned_values() {
        // Test that boolean values return the same handle
        let h_true1 = tribute_handle_new_boolean(true);
        let h_true2 = tribute_handle_new_boolean(true);
        let h_false1 = tribute_handle_new_boolean(false);
        let h_false2 = tribute_handle_new_boolean(false);

        assert_eq!(h_true1.0, h_true2.0);
        assert_eq!(h_false1.0, h_false2.0);
        assert_ne!(h_true1.0, h_false1.0);

        // Test that nil values return the same handle
        let h_nil1 = tribute_handle_new_nil();
        let h_nil2 = tribute_handle_new_nil();

        assert_eq!(h_nil1.0, h_nil2.0);

        // Test that interned values persist after clear_all
        tribute_handle_clear_all();

        assert!(h_true1.is_valid());
        assert!(h_false1.is_valid());
        assert!(h_nil1.is_valid());
    }
    
    #[test]
    fn test_string_interning() {
        use crate::interned_string::TributeString;
        
        // Clear interned strings for clean test
        TributeString::clear_interned();
        
        // Test empty string interning
        let empty1 = tribute_handle_new_string(std::ptr::null(), 0);
        let empty2 = tribute_handle_new_string_from_str("");
        let empty3 = tribute_handle_new_string("hello".as_ptr(), 0); // Zero length
        
        assert_eq!(empty1.0, empty2.0);
        assert_eq!(empty2.0, empty3.0);
        assert_eq!(empty1.0, INTERNED_EMPTY_STRING.0);
        
        // Test inline strings (should create new handles but use efficient storage)
        let short1 = tribute_handle_new_string_from_str("hello");
        let short2 = tribute_handle_new_string_from_str("hello");
        let world = tribute_handle_new_string_from_str("world");
        
        // Each string gets its own handle even if content is the same
        assert_ne!(short1.0, short2.0);
        assert_ne!(short1.0, world.0);
        assert_ne!(short1.0, empty1.0);
        
        // Test longer strings (these get interned based on content)
        let long1 = tribute_handle_new_string_from_str("this is a very long string that will be interned");
        let long2 = tribute_handle_new_string_from_str("this is a very long string that will be interned");
        
        // Longer strings still get separate handles (handle != content equality)
        assert_ne!(long1.0, long2.0);
        
        // Test that empty string persists after clear_all
        tribute_handle_clear_all();
        
        assert!(empty1.is_valid());
        assert!(!short1.is_valid()); // Non-interned handles should be cleared
        
        // Test string data retrieval
        let test_str = "Hello, Rust!";
        let str_handle = tribute_handle_new_string_from_str(test_str);
        
        let length = tribute_handle_get_string_length(str_handle);
        assert_eq!(length, test_str.len());
        
        // Test copying string data
        let mut buffer = vec![0u8; length + 10]; // Extra space
        let copied_len = tribute_handle_copy_string_data(
            str_handle,
            buffer.as_mut_ptr(),
            buffer.len()
        );
        
        assert_eq!(copied_len, test_str.len());
        
        let retrieved_str = std::str::from_utf8(&buffer[..length]).unwrap();
        assert_eq!(retrieved_str, test_str);
        
        tribute_handle_release(str_handle);
        
        // Test interned string statistics
        assert!(TributeString::interned_count() > 0);
    }
}
