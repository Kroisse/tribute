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
use std::sync::Mutex;

/// An opaque handle that indirectly references a TributeBoxed value
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TributeHandle(u64);

/// Invalid handle constant
pub const TRIBUTE_HANDLE_INVALID: TributeHandle = TributeHandle(0);

/// Handle table that manages handles to TributeBoxed values
pub struct HandleTable {
    table: DashMap<u64, Box<TributeBoxed>>,
    counter: AtomicU64,
    stats: Mutex<HandleStats>,
}

impl HandleTable {
    /// Create a new handle table with pre-populated interned values
    pub fn new() -> Self {
        use crate::interned_string::TributeString;

        let table = DashMap::new();

        // Pre-insert interned values
        table.insert(INTERNED_TRUE.0, Box::new(TributeBoxed::new(TributeValue::Boolean(true))));
        table.insert(INTERNED_FALSE.0, Box::new(TributeBoxed::new(TributeValue::Boolean(false))));
        table.insert(INTERNED_NIL.0, Box::new(TributeBoxed::new(TributeValue::Nil)));
        // Note: We'll create the empty string using inline storage, not interned
        table.insert(INTERNED_EMPTY_STRING.0, Box::new(TributeBoxed::new(TributeValue::String(TributeString::Empty))));

        Self {
            table,
            counter: AtomicU64::new(5), // Start at 5 to reserve 1-4 for interned values
            stats: Mutex::new(HandleStats::new()),
        }
    }

    /// Create a new handle for a TributeBoxed value
    pub fn create_handle(&self, boxed: TributeBoxed) -> TributeHandle {
        let handle_id = self.counter.fetch_add(1, Ordering::Relaxed);
        let handle = TributeHandle(handle_id);

        // Store the boxed value in the handle table
        self.table.insert(handle_id, Box::new(boxed));

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocated += 1;
            let current_count = stats.allocated - stats.deallocated;
            if current_count > stats.peak_count {
                stats.peak_count = current_count;
            }
        }

        handle
    }

    /// Check if a handle is valid
    pub fn is_valid(&self, handle: TributeHandle) -> bool {
        if handle.0 == 0 {
            return false;
        }

        self.table.contains_key(&handle.0)
    }

    /// Execute a closure with access to the TributeBoxed value
    pub fn with_value<T, F>(&self, handle: TributeHandle, f: F) -> Option<T>
    where
        F: FnOnce(&TributeBoxed) -> T,
    {
        if handle.0 == 0 {
            return None;
        }

        self.table.get(&handle.0).map(|boxed| f(&boxed))
    }

    /// Execute a closure with mutable access to the TributeBoxed value
    pub fn with_value_mut<T, F>(&self, handle: TributeHandle, f: F) -> Option<T>
    where
        F: FnOnce(&mut TributeBoxed) -> T,
    {
        if handle.0 == 0 {
            return None;
        }

        self.table.get_mut(&handle.0).map(|mut boxed| f(&mut boxed))
    }

    /// Release a handle and deallocate the associated value
    pub fn release(&self, handle: TributeHandle) {
        if handle.0 == 0 {
            return;
        }

        // Never deallocate interned values
        if handle.0 >= 1 && handle.0 <= 4 {
            return;
        }

        if self.table.remove(&handle.0).is_some() {
            let mut stats = self.stats.lock().unwrap();
            stats.deallocated += 1;
        }
    }

    /// Clear all handles except interned ones (for testing/cleanup)
    pub fn clear_all(&self) {
        // Remove all handles except interned ones
        let interned_keys = [1, 2, 3, 4];
        self.table.retain(|k, _| interned_keys.contains(k));

        let mut stats = self.stats.lock().unwrap();
        stats.deallocated = stats.allocated.saturating_sub(4); // Keep 4 interned values
    }

    /// Get handle management statistics
    pub fn get_stats(&self) -> (u64, u64, u64) {
        let stats = self.stats.lock().unwrap();
        (stats.allocated, stats.deallocated, stats.peak_count)
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime context that contains a handle table and other runtime state
pub struct TributeRuntime {
    handle_table: HandleTable,
    pub(crate) interned_strings: crate::interned_string::InternedStringTable,
}

impl TributeRuntime {
    /// Create a new runtime instance
    pub fn new() -> Self {
        Self {
            handle_table: HandleTable::new(),
            interned_strings: crate::interned_string::InternedStringTable::new(),
        }
    }
    
    /// Execute a closure with access to the TributeBoxed value
    pub fn with_value<T, F>(&self, handle: TributeHandle, f: F) -> Option<T>
    where
        F: FnOnce(&TributeBoxed) -> T,
    {
        self.handle_table.with_value(handle, f)
    }
    
    /// Execute a closure with mutable access to the TributeBoxed value
    pub fn with_value_mut<T, F>(&self, handle: TributeHandle, f: F) -> Option<T>
    where
        F: FnOnce(&mut TributeBoxed) -> T,
    {
        self.handle_table.with_value_mut(handle, f)
    }
    
    /// Check if a handle is valid
    pub fn is_valid(&self, handle: TributeHandle) -> bool {
        self.handle_table.is_valid(handle)
    }
    
    /// Create a new handle for a TributeBoxed value
    pub fn create_handle(&self, boxed: TributeBoxed) -> TributeHandle {
        self.handle_table.create_handle(boxed)
    }
    
    /// Release a handle and deallocate the associated value
    pub fn release(&self, handle: TributeHandle) {
        self.handle_table.release(handle)
    }
    
    /// Clear all handles except interned ones (for testing/cleanup)
    pub fn clear_all(&self) {
        self.handle_table.clear_all()
    }
    
    /// Get handle management statistics
    pub fn get_stats(&self) -> (u64, u64, u64) {
        self.handle_table.get_stats()
    }
}

impl Default for TributeRuntime {
    fn default() -> Self {
        Self::new()
    }
}



/// Interned handle constants
pub const INTERNED_TRUE: TributeHandle = TributeHandle(1);
pub const INTERNED_FALSE: TributeHandle = TributeHandle(2);
pub const INTERNED_NIL: TributeHandle = TributeHandle(3);
pub const INTERNED_EMPTY_STRING: TributeHandle = TributeHandle(4);


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


// TributeHandle is now just a simple wrapper - all operations require a runtime context



// Context-aware C FFI functions

/// Create a new runtime instance
#[unsafe(no_mangle)]
pub extern "C" fn tribute_runtime_new() -> *mut TributeRuntime {
    Box::into_raw(Box::new(TributeRuntime::new()))
}

/// Destroy a runtime instance
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_runtime_destroy(runtime: *mut TributeRuntime) {
    if !runtime.is_null() {
        unsafe {
            drop(Box::from_raw(runtime));
        }
    }
}

/// Create a new handle for a number value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_new_number(
    runtime: *mut TributeRuntime,
    value: i64,
) -> TributeHandle {
    if runtime.is_null() {
        return TRIBUTE_HANDLE_INVALID;
    }

    unsafe {
        let runtime_ref = &*runtime;
        let boxed = TributeBoxed::new(TributeValue::Number(value));
        runtime_ref.create_handle(boxed)
    }
}

/// Create a new handle for a boolean value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_new_boolean(
    runtime: *mut TributeRuntime,
    value: bool,
) -> TributeHandle {
    if runtime.is_null() {
        return TRIBUTE_HANDLE_INVALID;
    }

    // Use interned handles for true/false
    if value {
        INTERNED_TRUE
    } else {
        INTERNED_FALSE
    }
}

/// Create a new handle for a nil value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_new_nil(
    _runtime: *mut TributeRuntime,
) -> TributeHandle {
    // Use interned handle for nil
    INTERNED_NIL
}

/// Create a new handle for a string value#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_new_string(
    runtime: *mut TributeRuntime,
    data: *const u8,
    length: usize,
) -> TributeHandle {
    use crate::interned_string::TributeString;

    if runtime.is_null() {
        return TRIBUTE_HANDLE_INVALID;
    }

    // Check for empty string
    if length == 0 {
        return INTERNED_EMPTY_STRING;
    }

    unsafe {
        let runtime_ref = &*runtime;

        // Create string from byte slice
        let bytes = if data.is_null() {
            &[]
        } else {
            std::slice::from_raw_parts(data, length)
        };

        let tribute_string = TributeString::from_bytes_with_table(bytes, &runtime_ref.interned_strings);

        // Check if this resulted in an empty string (should be rare due to above check)
        if tribute_string.is_empty() {
            return INTERNED_EMPTY_STRING;
        }

        let boxed = TributeBoxed::new(TributeValue::String(tribute_string));
        runtime_ref.create_handle(boxed)
    }
}

/// Check if a handle is valid#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_is_valid(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> bool {
    if runtime.is_null() {
        return false;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref.is_valid(handle)
    }
}

/// Get the type of the value referenced by a handle#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_get_type(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> u8 {
    if runtime.is_null() {
        return TributeValue::TYPE_NIL;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref
            .with_value(handle, |boxed| boxed.value.type_id())
            .unwrap_or(TributeValue::TYPE_NIL)
    }
}

/// Unbox a number value from a handle#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_unbox_number(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> i64 {
    if runtime.is_null() {
        return 0;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::Number(n) => *n,
                    _ => 0, // Type error
                }
            })
            .unwrap_or(0)
    }
}

/// Unbox a boolean value from a handle#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_unbox_boolean(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> bool {
    if runtime.is_null() {
        return false;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::Boolean(b) => *b,
                    _ => false, // Type error
                }
            })
            .unwrap_or(false)
    }
}

/// Get string data from a handle (returns length, caller must copy data)#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_get_string_length(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> usize {
    if runtime.is_null() {
        return 0;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::String(tribute_string) => tribute_string.len(),
                    _ => 0, // Type error
                }
            })
            .unwrap_or(0)
    }
}

/// Copy string data from a handle to a buffer#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_copy_string_data(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
    buffer: *mut u8,
    buffer_size: usize,
) -> usize {
    if runtime.is_null() {
        return 0;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::String(tribute_string) => {
                        #[allow(deprecated)]
                        let bytes = tribute_string.as_bytes();
                        let copy_len = bytes.len().min(buffer_size);

                        if !buffer.is_null() && copy_len > 0 {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                buffer,
                                copy_len
                            );
                        }

                        bytes.len() // Return actual string length
                    }
                    _ => 0, // Type error
                }
            })
            .unwrap_or(0)
    }
}

/// Add two number handles#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_add_numbers(
    runtime: *mut TributeRuntime,
    lhs: TributeHandle,
    rhs: TributeHandle,
) -> TributeHandle {
    if runtime.is_null() {
        return TRIBUTE_HANDLE_INVALID;
    }

    unsafe {
        let runtime_ref = &*runtime;

        let lhs_val = runtime_ref
            .with_value(lhs, |boxed| match &boxed.value {
                TributeValue::Number(n) => Some(*n),
                _ => None,
            })
            .flatten();

        let rhs_val = runtime_ref
            .with_value(rhs, |boxed| match &boxed.value {
                TributeValue::Number(n) => Some(*n),
                _ => None,
            })
            .flatten();

        match (lhs_val, rhs_val) {
            (Some(a), Some(b)) => {
                let boxed = TributeBoxed::new(TributeValue::Number(a + b));
                runtime_ref.create_handle(boxed)
            },
            _ => {
                let boxed = TributeBoxed::new(TributeValue::Number(0));
                runtime_ref.create_handle(boxed)
            }
        }
    }
}

/// Retain a handle (increment reference count)#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_retain(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> TributeHandle {
    if runtime.is_null() {
        return handle;
    }

    // Interned values don't need reference counting
    if handle.0 >= 1 && handle.0 <= 4 {
        return handle;
    }

    unsafe {
        let runtime_ref = &*runtime;
        if runtime_ref.is_valid(handle) {
            runtime_ref.with_value(handle, |boxed| boxed.retain());
        }
        handle
    }
}

/// Release a handle (decrement reference count and potentially deallocate)#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_release(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) {
    if runtime.is_null() {
        return;
    }

    // Never release interned values
    if handle.0 >= 1 && handle.0 <= 4 {
        return;
    }

    unsafe {
        let runtime_ref = &*runtime;

        // Check if we should deallocate
        let should_deallocate = runtime_ref
            .with_value(handle, |boxed| boxed.release() == 0)
            .unwrap_or(false);

        if should_deallocate {
            runtime_ref.release(handle);
        }
    }
}

/// Get the reference count for a handle#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_get_ref_count(
    runtime: *mut TributeRuntime,
    handle: TributeHandle,
) -> usize {
    if runtime.is_null() {
        return 0;
    }

    // Interned values have infinite reference count (represented as 1)
    if handle.0 >= 1 && handle.0 <= 4 {
        return 1;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref.with_value(handle, |boxed| boxed.ref_count()).unwrap_or(0)
    }
}

/// Get handle management statistics#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_get_stats(
    runtime: *mut TributeRuntime,
    allocated: *mut u64,
    deallocated: *mut u64,
    peak_count: *mut u64,
) {
    if runtime.is_null() || allocated.is_null() || deallocated.is_null() || peak_count.is_null() {
        return;
    }

    unsafe {
        let runtime_ref = &*runtime;
        let (allocated_count, deallocated_count, peak_count_val) = runtime_ref.get_stats();
        *allocated = allocated_count;
        *deallocated = deallocated_count;
        *peak_count = peak_count_val;
    }
}

/// Clear all handles (for testing/cleanup)#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_handle_clear_all(runtime: *mut TributeRuntime) {
    if runtime.is_null() {
        return;
    }

    unsafe {
        let runtime_ref = &*runtime;
        runtime_ref.clear_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thread_local::ThreadLocal;
    use std::sync::LazyLock;

    // Test runtime for isolated testing (thread local)
    static TEST_RUNTIME: LazyLock<ThreadLocal<TributeRuntime>> = LazyLock::new(ThreadLocal::new);

    // Test wrapper functions using test runtime
    fn tribute_handle_new_number(value: i64) -> TributeHandle {
        let runtime = TEST_RUNTIME.get_or_default();
        let boxed = TributeBoxed::new(TributeValue::Number(value));
        runtime.handle_table.create_handle(boxed)
    }

    fn tribute_handle_new_boolean(value: bool) -> TributeHandle {
        if value { INTERNED_TRUE } else { INTERNED_FALSE }
    }

    fn tribute_handle_new_nil() -> TributeHandle {
        INTERNED_NIL
    }

    fn tribute_handle_new_string(data: *const u8, length: usize) -> TributeHandle {
        use crate::interned_string::TributeString;

        if length == 0 {
            return INTERNED_EMPTY_STRING;
        }

        let bytes = if data.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(data, length) }
        };

        let runtime = TEST_RUNTIME.get_or_default();
        let tribute_string = TributeString::from_bytes_with_table(bytes, &runtime.interned_strings);

        if tribute_string.is_empty() {
            return INTERNED_EMPTY_STRING;
        }

        let boxed = TributeBoxed::new(TributeValue::String(tribute_string));
        runtime.handle_table.create_handle(boxed)
    }

    fn tribute_handle_new_string_from_str(s: &str) -> TributeHandle {
        use crate::interned_string::TributeString;

        if s.is_empty() {
            return INTERNED_EMPTY_STRING;
        }

        let runtime = TEST_RUNTIME.get_or_default();
        let tribute_string = TributeString::from_str_with_table(s, &runtime.interned_strings);
        let boxed = TributeBoxed::new(TributeValue::String(tribute_string));
        runtime.handle_table.create_handle(boxed)
    }

    fn tribute_handle_is_valid(handle: TributeHandle) -> bool {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table.is_valid(handle)
    }

    fn tribute_handle_get_type(handle: TributeHandle) -> u8 {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table
            .with_value(handle, |boxed| boxed.value.type_id())
            .unwrap_or(TributeValue::TYPE_NIL)
    }

    fn tribute_handle_unbox_number(handle: TributeHandle) -> i64 {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::Number(n) => *n,
                    _ => 0,
                }
            })
            .unwrap_or(0)
    }

    fn tribute_handle_unbox_boolean(handle: TributeHandle) -> bool {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::Boolean(b) => *b,
                    _ => false,
                }
            })
            .unwrap_or(false)
    }

    fn tribute_handle_get_string_length(handle: TributeHandle) -> usize {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::String(tribute_string) => tribute_string.len(),
                    _ => 0,
                }
            })
            .unwrap_or(0)
    }

    fn tribute_handle_copy_string_data(
        handle: TributeHandle,
        buffer: *mut u8,
        buffer_size: usize,
    ) -> usize {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::String(tribute_string) => {
                        #[allow(deprecated)]
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

                        bytes.len()
                    }
                    _ => 0,
                }
            })
            .unwrap_or(0)
    }

    fn tribute_handle_add_numbers(
        lhs: TributeHandle,
        rhs: TributeHandle,
    ) -> TributeHandle {
        let runtime = TEST_RUNTIME.get_or_default();
        
        let lhs_val = runtime.handle_table
            .with_value(lhs, |boxed| match &boxed.value {
                TributeValue::Number(n) => Some(*n),
                _ => None,
            })
            .flatten();

        let rhs_val = runtime.handle_table
            .with_value(rhs, |boxed| match &boxed.value {
                TributeValue::Number(n) => Some(*n),
                _ => None,
            })
            .flatten();

        match (lhs_val, rhs_val) {
            (Some(a), Some(b)) => {
                let boxed = TributeBoxed::new(TributeValue::Number(a + b));
                runtime.handle_table.create_handle(boxed)
            },
            _ => {
                let boxed = TributeBoxed::new(TributeValue::Number(0));
                runtime.handle_table.create_handle(boxed)
            }
        }
    }

    fn tribute_handle_retain(handle: TributeHandle) -> TributeHandle {
        if handle.0 >= 1 && handle.0 <= 4 {
            return handle;
        }

        let runtime = TEST_RUNTIME.get_or_default();
        if runtime.handle_table.is_valid(handle) {
            runtime.handle_table.with_value(handle, |boxed| boxed.retain());
        }
        handle
    }

    fn tribute_handle_release(handle: TributeHandle) {
        if handle.0 >= 1 && handle.0 <= 4 {
            return;
        }

        let runtime = TEST_RUNTIME.get_or_default();
        let should_deallocate = runtime.handle_table
            .with_value(handle, |boxed| boxed.release() == 0)
            .unwrap_or(false);

        if should_deallocate {
            runtime.handle_table.release(handle);
        }
    }

    fn tribute_handle_get_ref_count(handle: TributeHandle) -> usize {
        if handle.0 >= 1 && handle.0 <= 4 {
            return 1;
        }

        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table.with_value(handle, |boxed| boxed.ref_count()).unwrap_or(0)
    }

    fn tribute_handle_clear_all() {
        let runtime = TEST_RUNTIME.get_or_default();
        runtime.handle_table.clear_all();
    }

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

        assert!(tribute_handle_is_valid(h_true));
        assert!(tribute_handle_is_valid(h_false));
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
        assert!(tribute_handle_is_valid(bool_handle));
        assert!(tribute_handle_is_valid(nil_handle));
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

        assert!(tribute_handle_is_valid(h_true1));
        assert!(tribute_handle_is_valid(h_false1));
        assert!(tribute_handle_is_valid(h_nil1));
    }

    #[test]
    fn test_string_interning() {
        use crate::interned_string::TributeString;

        // Clear interned strings for clean test
        let runtime = TEST_RUNTIME.get_or_default();
        TributeString::clear_interned_from_table(&runtime.interned_strings);

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

        assert!(tribute_handle_is_valid(empty1));
        assert!(!tribute_handle_is_valid(short1)); // Non-interned handles should be cleared

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
        let runtime = TEST_RUNTIME.get_or_default();
        assert!(TributeString::interned_count_from_table(&runtime.interned_strings) > 0);
    }

    #[test]
    fn test_context_aware_api() {
        unsafe {
            // Create a new runtime
            let runtime = tribute_runtime_new();
            assert!(!runtime.is_null());

            // Create handles using runtime-aware API
            let num_handle = super::tribute_handle_new_number(runtime, 42);
            assert_ne!(num_handle, TRIBUTE_HANDLE_INVALID);

            // Test that the handle is valid in this runtime context
            let runtime_ref = &*runtime;
            let value = runtime_ref.with_value(num_handle, |boxed| {
                match &boxed.value {
                    TributeValue::Number(n) => *n,
                    _ => 0,
                }
            }).unwrap_or(0);
            assert_eq!(value, 42);

            // Test boolean handles (should return interned values)
            let true_handle = super::tribute_handle_new_boolean(runtime, true);
            let false_handle = super::tribute_handle_new_boolean(runtime, false);
            assert_eq!(true_handle, INTERNED_TRUE);
            assert_eq!(false_handle, INTERNED_FALSE);

            // Test nil handle (should return interned value)
            let nil_handle = super::tribute_handle_new_nil(runtime);
            assert_eq!(nil_handle, INTERNED_NIL);

            // Test string handle
            let hello = "Hello, Context!";
            let str_handle = super::tribute_handle_new_string(
                runtime,
                hello.as_ptr(),
                hello.len()
            );
            assert_ne!(str_handle, TRIBUTE_HANDLE_INVALID);

            // Test string length using context-aware runtime
            let str_len = runtime_ref.with_value(str_handle, |boxed| {
                match &boxed.value {
                    TributeValue::String(s) => s.len(),
                    _ => 0,
                }
            }).unwrap_or(0);
            assert_eq!(str_len, hello.len());

            // Clean up
            tribute_runtime_destroy(runtime);
        }
    }

    #[test]
    fn test_runtime_basic_functionality() {
        unsafe {
            // Test that we can create and destroy runtimes
            let runtime1 = tribute_runtime_new();
            let runtime2 = tribute_runtime_new();
            assert!(!runtime1.is_null());
            assert!(!runtime2.is_null());

            // Test that each runtime can create handles
            let handle1 = super::tribute_handle_new_number(runtime1, 100);
            let handle2 = super::tribute_handle_new_number(runtime2, 200);

            assert_ne!(handle1, TRIBUTE_HANDLE_INVALID);
            assert_ne!(handle2, TRIBUTE_HANDLE_INVALID);

            // Test that each runtime can access its own handles
            let runtime1_ref = &*runtime1;
            let runtime2_ref = &*runtime2;

            let val1 = runtime1_ref.handle_table.with_value(handle1, |boxed| {
                match &boxed.value {
                    TributeValue::Number(n) => *n,
                    _ => 0,
                }
            }).unwrap_or(-1);

            let val2 = runtime2_ref.handle_table.with_value(handle2, |boxed| {
                match &boxed.value {
                    TributeValue::Number(n) => *n,
                    _ => 0,
                }
            }).unwrap_or(-1);

            assert_eq!(val1, 100);
            assert_eq!(val2, 200);

            // Clean up
            tribute_runtime_destroy(runtime1);
            tribute_runtime_destroy(runtime2);
        }
    }
}

/// Create a new handle for a string value from a Rust string slice
pub unsafe fn tribute_handle_new_string_from_str(
    runtime: *mut TributeRuntime,
    s: &str,
) -> TributeHandle {
    use crate::interned_string::TributeString;

    if runtime.is_null() {
        return TRIBUTE_HANDLE_INVALID;
    }

    if s.is_empty() {
        return INTERNED_EMPTY_STRING;
    }

    unsafe {
        let runtime_ref = &*runtime;
        let tribute_string = TributeString::from_str_with_table(s, &runtime_ref.interned_strings);
        let boxed = TributeBoxed::new(TributeValue::String(tribute_string));
        runtime_ref.create_handle(boxed)
    }
}
