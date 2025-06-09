//! Boolean operations for the Tribute runtime
//!
//! This module is now deprecated. Use the handle-based API in the `handle` module instead.

#[cfg(test)]
mod tests {
    use crate::handle::{
        tribute_runtime_new, tribute_runtime_destroy,
        tribute_new_boolean, tribute_unbox_boolean,
        tribute_get_type, tribute_release,
    };
    use crate::value::TributeValue;

    #[test]
    fn test_handle_boolean_operations() {
        unsafe {
            let runtime = tribute_runtime_new();
            
            // Test true
            let handle_true = tribute_new_boolean(runtime, true);
            assert_eq!(tribute_get_type(runtime, handle_true), TributeValue::TYPE_BOOLEAN);
            assert_eq!(tribute_unbox_boolean(runtime, handle_true), true);
            tribute_release(runtime, handle_true);

            // Test false
            let handle_false = tribute_new_boolean(runtime, false);
            assert_eq!(tribute_get_type(runtime, handle_false), TributeValue::TYPE_BOOLEAN);
            assert_eq!(tribute_unbox_boolean(runtime, handle_false), false);
            tribute_release(runtime, handle_false);
            
            tribute_runtime_destroy(runtime);
        }
    }
}
