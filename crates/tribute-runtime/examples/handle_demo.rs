//! Demonstration of the handle-based API for Tribute runtime
//! 
//! This example shows how handles provide better safety and GC compatibility
//! compared to raw pointer-based APIs.

use tribute_runtime::{
    tribute_runtime_new, tribute_runtime_destroy,
    tribute_handle_new_number, tribute_handle_new_boolean,
    tribute_handle_unbox_number, tribute_handle_unbox_boolean,
    tribute_handle_add_numbers, tribute_handle_is_valid,
    tribute_handle_get_type, tribute_handle_get_ref_count,
    tribute_handle_retain, tribute_handle_release,
    tribute_handle_get_stats,
    TributeValue, TRIBUTE_HANDLE_INVALID,
};

fn main() {
    println!("=== Tribute Runtime Handle Demo ===\n");
    
    unsafe {
        // Create a runtime
        let runtime = tribute_runtime_new();
        
        // Create some values using handles
        println!("1. Creating values with handles:");
        let num1 = tribute_handle_new_number(runtime, 42);
        let num2 = tribute_handle_new_number(runtime, 58);
        let bool_val = tribute_handle_new_boolean(runtime, true);
        
        println!("  Number 1 handle: {:?}", num1);
        println!("  Number 2 handle: {:?}", num2);
        println!("  Boolean handle: {:?}", bool_val);
        
        // Check validity
        println!("\n2. Handle validity:");
        println!("  num1 valid: {}", tribute_handle_is_valid(runtime, num1));
        println!("  num2 valid: {}", tribute_handle_is_valid(runtime, num2));
        println!("  bool_val valid: {}", tribute_handle_is_valid(runtime, bool_val));
        println!("  invalid handle valid: {}", tribute_handle_is_valid(runtime, TRIBUTE_HANDLE_INVALID));
        
        // Type checking
        println!("\n3. Type checking:");
        println!("  num1 type: {} ({})", tribute_handle_get_type(runtime, num1), 
                 if tribute_handle_get_type(runtime, num1) == TributeValue::TYPE_NUMBER { "NUMBER" } else { "OTHER" });
        println!("  bool_val type: {} ({})", tribute_handle_get_type(runtime, bool_val),
                 if tribute_handle_get_type(runtime, bool_val) == TributeValue::TYPE_BOOLEAN { "BOOLEAN" } else { "OTHER" });
        
        // Unboxing values
        println!("\n4. Unboxing values:");
        println!("  num1 value: {}", tribute_handle_unbox_number(runtime, num1));
        println!("  num2 value: {}", tribute_handle_unbox_number(runtime, num2));
        println!("  bool_val value: {}", tribute_handle_unbox_boolean(runtime, bool_val));
        
        // Arithmetic operations
        println!("\n5. Arithmetic with handles:");
        let sum_handle = tribute_handle_add_numbers(runtime, num1, num2);
        println!("  {} + {} = {}", 
                 tribute_handle_unbox_number(runtime, num1),
                 tribute_handle_unbox_number(runtime, num2),
                 tribute_handle_unbox_number(runtime, sum_handle));
        
        // Reference counting
        println!("\n6. Reference counting:");
        println!("  num1 ref count: {}", tribute_handle_get_ref_count(runtime, num1));
        let retained_num1 = tribute_handle_retain(runtime, num1);
        println!("  After retain, num1 ref count: {}", tribute_handle_get_ref_count(runtime, num1));
        println!("  Retained handle: {:?}", retained_num1);
        
        // Handle statistics
        let mut allocated = 0u64;
        let mut deallocated = 0u64;
        let mut peak_count = 0u64;
        tribute_handle_get_stats(runtime, &mut allocated, &mut deallocated, &mut peak_count);
        
        println!("\n7. Handle statistics:");
        println!("  Allocated: {}", allocated);
        println!("  Deallocated: {}", deallocated);
        println!("  Currently active: {}", allocated - deallocated);
        println!("  Peak count: {}", peak_count);
        
        // Cleanup
        println!("\n8. Cleanup:");
        tribute_handle_release(runtime, num1);
        println!("  Released num1, ref count now: {}", tribute_handle_get_ref_count(runtime, num1));
        
        tribute_handle_release(runtime, retained_num1);
        println!("  Released retained_num1, num1 still valid: {}", tribute_handle_is_valid(runtime, num1));
        
        tribute_handle_release(runtime, num2);
        tribute_handle_release(runtime, bool_val);
        tribute_handle_release(runtime, sum_handle);
        
        // Final statistics
        let mut allocated = 0u64;
        let mut deallocated = 0u64;
        let mut peak_count = 0u64;
        tribute_handle_get_stats(runtime, &mut allocated, &mut deallocated, &mut peak_count);
        
        println!("  Final statistics:");
        println!("    Allocated: {}", allocated);
        println!("    Deallocated: {}", deallocated);
        println!("    Currently active: {}", allocated - deallocated);
        
        // Clean up runtime
        tribute_runtime_destroy(runtime);
    }
    
    println!("\n=== Demo completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test] 
    fn test_handle_demo_runs() {
        // Just ensure the demo runs without panicking
        main();
    }
}