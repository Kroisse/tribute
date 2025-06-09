//! String operations for the Tribute runtime
//!
//! This module is now deprecated. Use the handle-based API in the `handle` module instead.

#[cfg(test)]
mod tests {
    use crate::handle::{
        tribute_runtime_new, tribute_runtime_destroy,
        tribute_handle_new_string_from_str, tribute_handle_get_string_length,
        tribute_handle_get_type, tribute_handle_release,
    };
    use crate::value::TributeValue;

    #[test]
    fn test_handle_string_operations() {
        unsafe {
            let runtime = tribute_runtime_new();
            
            // Create a test string
            let test_str = "Hello, World!";
            let expected_length = test_str.len();

            // Create string handle
            let handle = tribute_handle_new_string_from_str(runtime, test_str);
            
            // Check type
            assert_eq!(tribute_handle_get_type(runtime, handle), TributeValue::TYPE_STRING);
            
            // Check length
            assert_eq!(tribute_handle_get_string_length(runtime, handle), expected_length);
            
            // Clean up
            tribute_handle_release(runtime, handle);
            tribute_runtime_destroy(runtime);
        }
    }
}
