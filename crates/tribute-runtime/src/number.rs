//! Number value operations for the Tribute runtime

use crate::value::{TributeBoxed, TributeValue};

/// Box a number value
/// 
/// # Safety
/// This function is safe to call with any i64 value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_box_number(value: i64) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeValue::Number(value));
    boxed.as_ptr()
}

/// Unbox a number value
/// 
/// # Safety
/// The caller must ensure that `boxed` is either null or points to a valid TributeBoxed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_number(boxed: *const TributeBoxed) -> i64 {
    if boxed.is_null() {
        return 0;
    }
    
    unsafe {
        let boxed_ref = &*boxed;
        match &boxed_ref.value {
            TributeValue::Number(n) => *n,
            _ => 0, // Type error, return default
        }
    }
}

/// Add two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_add_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_number(0);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_number(a + b)
            }
            _ => tribute_box_number(0), // Type error
        }
    }
}

/// Subtract two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_sub_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_number(0);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_number(a - b)
            }
            _ => tribute_box_number(0), // Type error
        }
    }
}

/// Multiply two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_mul_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_number(0);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_number(a * b)
            }
            _ => tribute_box_number(0), // Type error
        }
    }
}

/// Divide two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_div_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_number(0);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                if *b == 0 {
                    tribute_box_number(0) // Division by zero
                } else {
                    tribute_box_number(a / b)
                }
            }
            _ => tribute_box_number(0), // Type error
        }
    }
}

/// Modulo two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_mod_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_number(0);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                if *b == 0 {
                    tribute_box_number(0) // Modulo by zero
                } else {
                    tribute_box_number(a % b)
                }
            }
            _ => tribute_box_number(0), // Type error
        }
    }
}

// Comparison operations return boolean boxed values
use crate::boolean::tribute_box_boolean;

/// Check equality of two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_eq_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_boolean(false);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_boolean(a == b)
            }
            _ => tribute_box_boolean(false), // Type error
        }
    }
}

/// Check inequality of two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_neq_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_boolean(false);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_boolean(a != b)
            }
            _ => tribute_box_boolean(false), // Type error
        }
    }
}

/// Check if left < right for two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_lt_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_boolean(false);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_boolean(a < b)
            }
            _ => tribute_box_boolean(false), // Type error
        }
    }
}

/// Check if left <= right for two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_lte_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_boolean(false);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_boolean(a <= b)
            }
            _ => tribute_box_boolean(false), // Type error
        }
    }
}

/// Check if left > right for two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_gt_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_boolean(false);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_boolean(a > b)
            }
            _ => tribute_box_boolean(false), // Type error
        }
    }
}

/// Check if left >= right for two boxed numbers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_gte_boxed(lhs: *const TributeBoxed, rhs: *const TributeBoxed) -> *mut TributeBoxed {
    if lhs.is_null() || rhs.is_null() {
        return tribute_box_boolean(false);
    }
    
    unsafe {
        let lhs_ref = &*lhs;
        let rhs_ref = &*rhs;
        
        match (&lhs_ref.value, &rhs_ref.value) {
            (TributeValue::Number(a), TributeValue::Number(b)) => {
                tribute_box_boolean(a >= b)
            }
            _ => tribute_box_boolean(false), // Type error
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::tribute_release;

    #[test]
    fn test_number_boxing() {
        unsafe {
            let boxed = tribute_box_number(42);
            let unboxed = tribute_unbox_number(boxed);
            assert_eq!(unboxed, 42);
            tribute_release(boxed);
        }
    }

    #[test]
    fn test_arithmetic_operations() {
        unsafe {
            let a = tribute_box_number(10);
            let b = tribute_box_number(5);
            
            let sum = tribute_add_boxed(a, b);
            assert_eq!(tribute_unbox_number(sum), 15);
            
            let diff = tribute_sub_boxed(a, b);
            assert_eq!(tribute_unbox_number(diff), 5);
            
            let prod = tribute_mul_boxed(a, b);
            assert_eq!(tribute_unbox_number(prod), 50);
            
            let quot = tribute_div_boxed(a, b);
            assert_eq!(tribute_unbox_number(quot), 2);
            
            let rem = tribute_mod_boxed(a, b);
            assert_eq!(tribute_unbox_number(rem), 0);
            
            tribute_release(a);
            tribute_release(b);
            tribute_release(sum);
            tribute_release(diff);
            tribute_release(prod);
            tribute_release(quot);
            tribute_release(rem);
        }
    }
}