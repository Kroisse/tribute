use crate::{array::TributeArray, TributeBoxed, TributeValue};
use std::{alloc::{alloc, Layout}, ptr};

pub type TributeString = TributeArray<u8>;

/// Box a string value (takes ownership of the string data)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_string(data: *mut u8, length: usize) -> *mut TributeBoxed {
    let string = TributeString {
        data,
        length,
        capacity: length,
    };

    let boxed = TributeBoxed::new(TributeValue::String(string));
    boxed.as_ptr()
}

/// Unbox a string (returns pointer to string data and length)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_string(
    boxed: *mut TributeBoxed,
    length_out: *mut usize,
) -> *mut u8 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        match &(*boxed).value {
            TributeValue::String(string) => {
                if !length_out.is_null() {
                    *length_out = string.length;
                }
                string.data
            }
            _ => panic!("Type error: expected String, got different type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tribute_get_type, tribute_release};

    #[test]
    fn test_box_unbox_string() {
        unsafe {
            // Create a test string
            let test_str = "Hello, World!";
            let length = test_str.len();
            
            // Allocate memory and copy the string data
            let layout = Layout::from_size_align(length, 1).unwrap();
            let data = alloc(layout);
            ptr::copy_nonoverlapping(test_str.as_ptr(), data, length);

            // Box the string
            let boxed = tribute_box_string(data, length);
            assert!(!boxed.is_null());

            // Check type
            assert_eq!(tribute_get_type(boxed), TributeValue::TYPE_STRING);

            // Unbox the string
            let mut out_length = 0;
            let unboxed_data = tribute_unbox_string(boxed, &mut out_length);
            assert_eq!(out_length, length);
            assert!(!unboxed_data.is_null());

            // Verify content
            let slice = std::slice::from_raw_parts(unboxed_data, out_length);
            let unboxed_str = std::str::from_utf8(slice).unwrap();
            assert_eq!(unboxed_str, test_str);

            // Clean up
            tribute_release(boxed);
        }
    }
}