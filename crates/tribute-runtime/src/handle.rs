//! Handle-based memory management for the Tribute runtime
//!
//! This module provides an indirection layer between C FFI and TributeBoxed values,
//! making it easier to migrate to mark-and-sweep GC in the future.

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
static HANDLE_TABLE: LazyLock<DashMap<u64, Box<TributeBoxed>>> = LazyLock::new(DashMap::new);

/// Global handle counter for generating unique handles
static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

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
    let boxed = TributeBoxed::new(TributeValue::Boolean(value));
    TributeHandle::new(boxed)
}

/// Create a new handle for a nil value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_new_nil() -> TributeHandle {
    let boxed = TributeBoxed::new(TributeValue::Nil);
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
    if handle.is_valid() {
        handle.with_value(|boxed| boxed.retain());
    }
    handle
}

/// Release a handle (decrement reference count and potentially deallocate)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_release(handle: TributeHandle) {
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
    HANDLE_TABLE.clear();

    let mut stats = HANDLE_STATS.lock().unwrap();
    stats.deallocated = stats.allocated;
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

        tribute_handle_release(h_true);
        tribute_handle_release(h_false);
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
}
