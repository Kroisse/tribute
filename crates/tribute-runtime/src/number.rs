//! Number value operations for the Tribute runtime
//!
//! This module is now deprecated. Use the handle-based API in the `handle` module instead.

#[cfg(test)]
mod tests {
    use crate::handle::{
        tribute_runtime_new, tribute_runtime_destroy,
        tribute_new_number, tribute_unbox_number,
        tribute_add_numbers, tribute_release,
    };

    #[test]
    fn test_handle_number_operations() {
        unsafe {
            let runtime = tribute_runtime_new();
            
            let handle = tribute_new_number(runtime, 42);
            let unboxed = tribute_unbox_number(runtime, handle);
            assert_eq!(unboxed, 42);
            tribute_release(runtime, handle);
            
            tribute_runtime_destroy(runtime);
        }
    }

    #[test]
    fn test_handle_arithmetic_operations() {
        unsafe {
            let runtime = tribute_runtime_new();
            
            let a = tribute_new_number(runtime, 10);
            let b = tribute_new_number(runtime, 5);
            
            let sum = tribute_add_numbers(runtime, a, b);
            assert_eq!(tribute_unbox_number(runtime, sum), 15);
            
            tribute_release(runtime, a);
            tribute_release(runtime, b);
            tribute_release(runtime, sum);
            
            tribute_runtime_destroy(runtime);
        }
    }
}