//! Demonstration of the handle-based API for Tribute runtime
//! 
//! This example shows how handles provide better safety and GC compatibility
//! compared to raw pointer-based APIs.

use tribute_runtime::{
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
    
    // Create some values using handles
    println!("1. Creating values with handles:");
    let num1 = unsafe { tribute_handle_new_number(42) };
    let num2 = unsafe { tribute_handle_new_number(58) };
    let bool_val = unsafe { tribute_handle_new_boolean(true) };
    
    println!("  Number 1 handle: {:?}", num1);
    println!("  Number 2 handle: {:?}", num2);
    println!("  Boolean handle: {:?}", bool_val);
    
    // Check validity
    println!("\n2. Handle validity:");
    println!("  num1 valid: {}", unsafe { tribute_handle_is_valid(num1) });
    println!("  num2 valid: {}", unsafe { tribute_handle_is_valid(num2) });
    println!("  bool_val valid: {}", unsafe { tribute_handle_is_valid(bool_val) });
    println!("  invalid handle valid: {}", unsafe { tribute_handle_is_valid(TRIBUTE_HANDLE_INVALID) });
    
    // Type checking
    println!("\n3. Type checking:");
    println!("  num1 type: {} ({})", unsafe { tribute_handle_get_type(num1) }, 
             if unsafe { tribute_handle_get_type(num1) } == TributeValue::TYPE_NUMBER { "NUMBER" } else { "OTHER" });
    println!("  bool_val type: {} ({})", unsafe { tribute_handle_get_type(bool_val) },
             if unsafe { tribute_handle_get_type(bool_val) } == TributeValue::TYPE_BOOLEAN { "BOOLEAN" } else { "OTHER" });
    
    // Unboxing values
    println!("\n4. Unboxing values:");
    println!("  num1 value: {}", unsafe { tribute_handle_unbox_number(num1) });
    println!("  num2 value: {}", unsafe { tribute_handle_unbox_number(num2) });
    println!("  bool_val value: {}", unsafe { tribute_handle_unbox_boolean(bool_val) });
    
    // Arithmetic operations
    println!("\n5. Arithmetic with handles:");
    let sum_handle = unsafe { tribute_handle_add_numbers(num1, num2) };
    println!("  {} + {} = {}", 
             unsafe { tribute_handle_unbox_number(num1) },
             unsafe { tribute_handle_unbox_number(num2) },
             unsafe { tribute_handle_unbox_number(sum_handle) });
    
    // Reference counting
    println!("\n6. Reference counting:");
    println!("  num1 ref count: {}", unsafe { tribute_handle_get_ref_count(num1) });
    let retained_num1 = unsafe { tribute_handle_retain(num1) };
    println!("  After retain, num1 ref count: {}", unsafe { tribute_handle_get_ref_count(num1) });
    println!("  Retained handle: {:?}", retained_num1);
    
    // Handle statistics
    unsafe {
        let mut allocated = 0u64;
        let mut deallocated = 0u64;
        let mut peak_count = 0u64;
        tribute_handle_get_stats(&mut allocated, &mut deallocated, &mut peak_count);
        
        println!("\n7. Handle statistics:");
        println!("  Allocated: {}", allocated);
        println!("  Deallocated: {}", deallocated);
        println!("  Currently active: {}", allocated - deallocated);
        println!("  Peak count: {}", peak_count);
    }
    
    // Cleanup
    println!("\n8. Cleanup:");
    unsafe {
        tribute_handle_release(num1);
        println!("  Released num1, ref count now: {}", tribute_handle_get_ref_count(num1));
        
        tribute_handle_release(retained_num1);
        println!("  Released retained_num1, num1 still valid: {}", tribute_handle_is_valid(num1));
        
        tribute_handle_release(num2);
        tribute_handle_release(bool_val);
        tribute_handle_release(sum_handle);
        
        // Final statistics
        let mut allocated = 0u64;
        let mut deallocated = 0u64;
        let mut peak_count = 0u64;
        tribute_handle_get_stats(&mut allocated, &mut deallocated, &mut peak_count);
        
        println!("  Final statistics:");
        println!("    Allocated: {}", allocated);
        println!("    Deallocated: {}", deallocated);
        println!("    Currently active: {}", allocated - deallocated);
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
        
        // Clean up any remaining handles
        unsafe {
            tribute_handle_clear_all();
        }
    }
}