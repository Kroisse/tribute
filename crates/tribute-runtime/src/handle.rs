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
static HANDLE_TABLE: LazyLock<DashMap<u64, Box<TributeBoxed>>> = LazyLock::new(||
    // Pre-insert interned values
    [
        (INTERNED_TRUE.0, Box::new(TributeBoxed::new(TributeValue::Boolean(true)))),
        (INTERNED_FALSE.0, Box::new(TributeBoxed::new(TributeValue::Boolean(false)))),
        (INTERNED_NIL.0, Box::new(TributeBoxed::new(TributeValue::Nil))),
    ]
    .into_iter()
    .collect());

/// Global handle counter for generating unique handles
static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(4); // Start at 4 to reserve 1-3 for interned values

/// Interned handle constants
pub const INTERNED_TRUE: TributeHandle = TributeHandle(1);
pub const INTERNED_FALSE: TributeHandle = TributeHandle(2);
pub const INTERNED_NIL: TributeHandle = TributeHandle(3);

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
        if self.0 >= 1 && self.0 <= 3 {
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
    // Interned values don't need reference counting
    if handle.0 >= 1 && handle.0 <= 3 {
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
    if handle.0 >= 1 && handle.0 <= 3 {
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
    if handle.0 >= 1 && handle.0 <= 3 {
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
    let interned_keys = [1, 2, 3];
    HANDLE_TABLE.retain(|k, _| interned_keys.contains(k));

    let mut stats = HANDLE_STATS.lock().unwrap();
    stats.deallocated = stats.allocated.saturating_sub(3); // Keep 3 interned values
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
}
