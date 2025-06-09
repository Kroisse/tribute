#![allow(deprecated)] // Internal implementation can use deprecated functions

use crate::value::{TributeBoxed, TributeValue};

/// Box a boolean value
#[deprecated(
    since = "0.1.0",
    note = "Use handle-based API instead. See tribute_handle_new_boolean() for safer alternatives."
)]
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_boolean(value: bool) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeValue::Boolean(value));
    boxed.as_ptr()
}

/// Unbox a boolean
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[deprecated(
    since = "0.1.0",
    note = "Use handle-based API instead. See tribute_handle_unbox_boolean() for safer alternatives."
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_boolean(boxed: *mut TributeBoxed) -> bool {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        match &(*boxed).value {
            TributeValue::Boolean(value) => *value,
            _ => panic!("Type error: expected Boolean, got different type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{tribute_get_type, tribute_release};

    #[test]
    fn test_box_unbox_boolean() {
        unsafe {
            // Test true
            let boxed_true = tribute_box_boolean(true);
            assert!(!boxed_true.is_null());
            assert_eq!(tribute_get_type(boxed_true), TributeValue::TYPE_BOOLEAN);
            assert_eq!(tribute_unbox_boolean(boxed_true), true);
            tribute_release(boxed_true);

            // Test false
            let boxed_false = tribute_box_boolean(false);
            assert!(!boxed_false.is_null());
            assert_eq!(tribute_get_type(boxed_false), TributeValue::TYPE_BOOLEAN);
            assert_eq!(tribute_unbox_boolean(boxed_false), false);
            tribute_release(boxed_false);
        }
    }
}
