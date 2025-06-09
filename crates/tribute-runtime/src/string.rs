//! String operations for the Tribute runtime
//!
//! This module is now deprecated. Use the handle-based API in the `handle` module instead.

#[cfg(test)]
mod tests {
    use crate::handle::{
        tribute_runtime_new, tribute_runtime_destroy,
        tribute_new_string_from_str, tribute_get_string_length,
        tribute_get_type, tribute_release,
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
            let handle = tribute_new_string_from_str(runtime, test_str);
            
            // Check type
            assert_eq!(tribute_get_type(runtime, handle), TributeValue::TYPE_STRING);
            
            // Check length
            assert_eq!(tribute_get_string_length(runtime, handle), expected_length);
            
            // Clean up
            tribute_release(runtime, handle);
            tribute_runtime_destroy(runtime);
        }
    }
}
