//! Core value system for the Tribute runtime
//!
//! Provides reference-counted boxed values and memory management.


use crate::{array::TributeArray, handle::TributeHandle, interned_string::TributeString};
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
    String(TributeString),
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
                    TributeValue::String(_) => {
                        // Interned strings manage their own memory
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{
        tribute_runtime_new, tribute_runtime_destroy,
        tribute_handle_new_number, tribute_handle_new_boolean,
        tribute_handle_get_type, tribute_handle_get_ref_count,
        tribute_handle_retain, tribute_handle_release,
    };

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
    fn test_handle_reference_counting() {
        unsafe {
            let runtime = tribute_runtime_new();
            
            // Use a non-interned value (numbers > 4 are not interned)
            let handle = tribute_handle_new_number(runtime, 100);
            assert_eq!(tribute_handle_get_ref_count(runtime, handle), 1);

            let retained_handle = tribute_handle_retain(runtime, handle);
            assert_eq!(tribute_handle_get_ref_count(runtime, handle), 2);
            assert_eq!(retained_handle, handle);

            tribute_handle_release(runtime, handle);
            assert_eq!(tribute_handle_get_ref_count(runtime, handle), 1);

            tribute_handle_release(runtime, retained_handle);
            
            tribute_runtime_destroy(runtime);
        }
    }

    #[test]
    fn test_handle_type_checking() {
        unsafe {
            let runtime = tribute_runtime_new();
            
            let num_handle = tribute_handle_new_number(runtime, 123);
            let bool_handle = tribute_handle_new_boolean(runtime, false);

            assert_eq!(tribute_handle_get_type(runtime, num_handle), TributeValue::TYPE_NUMBER);
            assert_eq!(tribute_handle_get_type(runtime, bool_handle), TributeValue::TYPE_BOOLEAN);

            tribute_handle_release(runtime, num_handle);
            tribute_handle_release(runtime, bool_handle);
            
            tribute_runtime_destroy(runtime);
        }
    }
}
