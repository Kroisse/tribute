//! Core value system for the Tribute runtime
//!
//! Provides reference-counted boxed values and memory management.

use crate::{array::TributeArray, handle::TributeHandle};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A reference-counted boxed value in the Tribute runtime
#[repr(C)]
pub struct TributeBoxed {
    pub value: TributeValue,
    ref_count: AtomicUsize,
}

/// The actual value stored in a TributeBoxed
#[derive(Debug)]
pub enum TributeValue {
    Number(i64),
    String(TributeArray<u8>),
    Boolean(bool),
    List(TributeArray<TributeHandle>),
    Nil,
}

impl TributeValue {
    pub const TYPE_NUMBER: u8 = 0;
    pub const TYPE_STRING: u8 = 1;
    pub const TYPE_BOOLEAN: u8 = 2;
    pub const TYPE_LIST: u8 = 3;
    pub const TYPE_NIL: u8 = 4;

    pub fn type_id(&self) -> u8 {
        match self {
            TributeValue::Number(_) => Self::TYPE_NUMBER,
            TributeValue::String(_) => Self::TYPE_STRING,
            TributeValue::Boolean(_) => Self::TYPE_BOOLEAN,
            TributeValue::List(_) => Self::TYPE_LIST,
            TributeValue::Nil => Self::TYPE_NIL,
        }
    }
}

// Safety: TributeBoxed uses atomic reference counting and proper cleanup,
// making it safe to send between threads and share via RwLock
unsafe impl Send for TributeBoxed {}
unsafe impl Sync for TributeBoxed {}

impl TributeBoxed {
    /// Create a new boxed value
    pub fn new(value: TributeValue) -> Self {
        Self {
            value,
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Get a pointer to this boxed value for C FFI
    pub fn as_ptr(self) -> *mut Self {
        Box::into_raw(Box::new(self))
    }

    /// Increment the reference count
    pub fn retain(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the reference count and return the new count
    pub fn release(&self) -> usize {
        let old_count = self.ref_count.fetch_sub(1, Ordering::Relaxed);
        let new_count = old_count.saturating_sub(1);

        if old_count == 1 {
            // This was the last reference, deallocate
            unsafe {
                match &self.value {
                    TributeValue::String(string_data) => {
                        string_data.deallocate();
                    }
                    TributeValue::List(list_data) => {
                        list_data.release_all_handles();
                        list_data.deallocate();
                    }
                    _ => {} // Other types don't need special cleanup
                }
            }
        }

        new_count
    }

    /// Get the current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Relaxed)
    }
}

/// Increment the reference count of a boxed value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_retain(boxed: *mut TributeBoxed) -> *mut TributeBoxed {
    if boxed.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let boxed_ref = &*boxed;
        boxed_ref.retain();
        boxed
    }
}

/// Decrement the reference count of a boxed value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_release(boxed: *mut TributeBoxed) {
    if !boxed.is_null() {
        unsafe {
            let boxed_ref = &*boxed;
            let old_count = boxed_ref.ref_count.fetch_sub(1, Ordering::Relaxed);
            if old_count == 1 {
                // Last reference, deallocate
                drop(Box::from_raw(boxed));
            }
        }
    }
}

/// Get the reference count of a boxed value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_get_ref_count(boxed: *const TributeBoxed) -> usize {
    if boxed.is_null() {
        return 0;
    }

    unsafe {
        let boxed_ref = &*boxed;
        boxed_ref.ref_count()
    }
}

/// Get the type of a boxed value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_get_type(boxed: *const TributeBoxed) -> u8 {
    if boxed.is_null() {
        return TributeValue::TYPE_NIL;
    }

    unsafe {
        let boxed_ref = &*boxed;
        boxed_ref.value.type_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxed_value_creation() {
        let value = TributeBoxed::new(TributeValue::Number(42));
        assert_eq!(value.ref_count(), 1);
        match value.value {
            TributeValue::Number(n) => assert_eq!(n, 42),
            _ => panic!("Expected number"),
        }
    }

    #[test]
    fn test_reference_counting() {
        unsafe {
            let value = Box::new(TributeBoxed::new(TributeValue::Boolean(true)));
            let ptr = Box::into_raw(value);

            assert_eq!(tribute_get_ref_count(ptr), 1);

            let retained_ptr = tribute_retain(ptr);
            assert_eq!(tribute_get_ref_count(ptr), 2);
            assert_eq!(retained_ptr, ptr);

            tribute_release(ptr);
            assert_eq!(tribute_get_ref_count(ptr), 1);

            tribute_release(ptr);
        }
    }

    #[test]
    fn test_type_checking() {
        unsafe {
            let num_ptr = TributeBoxed::new(TributeValue::Number(123)).as_ptr();
            let bool_ptr = TributeBoxed::new(TributeValue::Boolean(false)).as_ptr();

            assert_eq!(tribute_get_type(num_ptr), TributeValue::TYPE_NUMBER);
            assert_eq!(tribute_get_type(bool_ptr), TributeValue::TYPE_BOOLEAN);

            tribute_release(num_ptr);
            tribute_release(bool_ptr);
        }
    }
}
