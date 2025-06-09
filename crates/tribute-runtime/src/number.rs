use crate::value::{TributeBoxed, TributeValue};

/// Box a number value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_number(value: i64) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeValue::Number(value));
    boxed.as_ptr()
}

/// Unbox a number (with type checking)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_number(boxed: *mut TributeBoxed) -> i64 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        match &(*boxed).value {
            TributeValue::Number(value) => *value,
            _ => panic!("Type error: expected Number, got different type"),
        }
    }
}

/// Arithmetic operations on boxed values
/// Add two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_add_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val + right_val;

        // Release input arguments (consumed by the operation)
        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Subtract two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_sub_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val - right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Multiply two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_mul_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val * right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Divide two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_div_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);

        if right_val == 0 {
            panic!("Division by zero");
        }

        let result = left_val / right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Compare two boxed numbers for equality
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_eq_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val == right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        crate::boolean::tribute_box_boolean(result)
    }
}

/// Compare two boxed numbers for less than
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_lt_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val < right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        crate::boolean::tribute_box_boolean(result)
    }
}

/// Compare two boxed numbers for greater than
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_gt_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val > right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        crate::boolean::tribute_box_boolean(result)
    }
}

/// Compare two boxed numbers for less than or equal
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_le_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val <= right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        crate::boolean::tribute_box_boolean(result)
    }
}

/// Compare two boxed numbers for greater than or equal
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_ge_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val >= right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        crate::boolean::tribute_box_boolean(result)
    }
}

/// Modulo operation on two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_mod_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);

        if right_val == 0 {
            panic!("Modulo by zero");
        }

        let result = left_val % right_val;

        crate::tribute_release(lhs);
        crate::tribute_release(rhs);

        tribute_box_number(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tribute_get_type, tribute_release};

    #[test]
    fn test_box_unbox_number() {
        unsafe {
            let value = 42i64;

            // Box the number
            let boxed = tribute_box_number(value);
            assert!(!boxed.is_null());

            // Check type
            assert_eq!(tribute_get_type(boxed), TributeValue::TYPE_NUMBER);

            // Unbox the number
            let unboxed = tribute_unbox_number(boxed);
            assert_eq!(unboxed, value);

            // Clean up
            tribute_release(boxed);
        }
    }

    #[test]
    fn test_arithmetic_operations() {
        unsafe {
            // Test addition
            let a = tribute_box_number(10);
            let b = tribute_box_number(20);
            let result = tribute_add_boxed(a, b);
            assert_eq!(tribute_unbox_number(result), 30);
            tribute_release(result);

            // Test subtraction
            let a = tribute_box_number(30);
            let b = tribute_box_number(10);
            let result = tribute_sub_boxed(a, b);
            assert_eq!(tribute_unbox_number(result), 20);
            tribute_release(result);

            // Test multiplication
            let a = tribute_box_number(5);
            let b = tribute_box_number(6);
            let result = tribute_mul_boxed(a, b);
            assert_eq!(tribute_unbox_number(result), 30);
            tribute_release(result);

            // Test division
            let a = tribute_box_number(20);
            let b = tribute_box_number(4);
            let result = tribute_div_boxed(a, b);
            assert_eq!(tribute_unbox_number(result), 5);
            tribute_release(result);

            // Test modulo
            let a = tribute_box_number(17);
            let b = tribute_box_number(5);
            let result = tribute_mod_boxed(a, b);
            assert_eq!(tribute_unbox_number(result), 2);
            tribute_release(result);
        }
    }

    #[test]
    fn test_comparison_operations() {
        unsafe {
            // Test equality
            let a = tribute_box_number(10);
            let b = tribute_box_number(10);
            let result = tribute_eq_boxed(a, b);
            assert_eq!(crate::boolean::tribute_unbox_boolean(result), true);
            tribute_release(result);

            // Test less than
            let a = tribute_box_number(5);
            let b = tribute_box_number(10);
            let result = tribute_lt_boxed(a, b);
            assert_eq!(crate::boolean::tribute_unbox_boolean(result), true);
            tribute_release(result);

            // Test greater than
            let a = tribute_box_number(15);
            let b = tribute_box_number(10);
            let result = tribute_gt_boxed(a, b);
            assert_eq!(crate::boolean::tribute_unbox_boolean(result), true);
            tribute_release(result);

            // Test less than or equal
            let a = tribute_box_number(10);
            let b = tribute_box_number(10);
            let result = tribute_le_boxed(a, b);
            assert_eq!(crate::boolean::tribute_unbox_boolean(result), true);
            tribute_release(result);

            // Test greater than or equal
            let a = tribute_box_number(15);
            let b = tribute_box_number(10);
            let result = tribute_ge_boxed(a, b);
            assert_eq!(crate::boolean::tribute_unbox_boolean(result), true);
            tribute_release(result);
        }
    }
}