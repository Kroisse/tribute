use crate::{array::TributeArray, TributeBoxed, TributeValue};

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