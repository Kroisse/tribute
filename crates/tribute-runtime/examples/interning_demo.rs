//! Demonstrates the interning of true, false, and nil values

use tribute_runtime::*;

fn main() {
    println!("=== Tribute Value Interning Demo ===\n");

    println!("1. Creating boolean values:");
    let true1 = tribute_handle_new_boolean(true);
    let true2 = tribute_handle_new_boolean(true);
    let false1 = tribute_handle_new_boolean(false);
    let false2 = tribute_handle_new_boolean(false);
    
    println!("  true1 handle:  {:?}", true1);
    println!("  true2 handle:  {:?}", true2);
    println!("  false1 handle: {:?}", false1);
    println!("  false2 handle: {:?}", false2);
    println!("  => Same value returns same handle: true1 == true2: {}", 
             true1 == true2);
    println!("  => Same value returns same handle: false1 == false2: {}", 
             false1 == false2);

    println!("\n2. Creating nil values:");
    let nil1 = tribute_handle_new_nil();
    let nil2 = tribute_handle_new_nil();
    
    println!("  nil1 handle: {:?}", nil1);
    println!("  nil2 handle: {:?}", nil2);
    println!("  => Same value returns same handle: nil1 == nil2: {}", 
             nil1 == nil2);

    println!("\n3. Reference counting interned values:");
    println!("  Retaining true value...");
    let retained_true = tribute_handle_retain(true1);
    println!("  Reference count for true: {}", 
             tribute_handle_get_ref_count(true1));
    println!("  => Interned values always report ref count of 1");
    
    println!("\n4. Releasing interned values:");
    println!("  Releasing true value...");
    tribute_handle_release(true1);
    println!("  Is true still valid? {}", tribute_handle_is_valid(true1));
    println!("  => Interned values are never deallocated");
    
    println!("\n5. Comparing with regular values:");
    let num1 = tribute_handle_new_number(42);
    let num2 = tribute_handle_new_number(42);
    
    println!("  num1 handle: {:?}", num1);
    println!("  num2 handle: {:?}", num2);
    println!("  => Different handles for same number: num1 == num2: {}", 
             num1 == num2);
    
    println!("\n6. Clear all handles test:");
    println!("  Clearing all handles...");
    tribute_handle_clear_all();
    
    println!("  Is true still valid? {}", tribute_handle_is_valid(true1));
    println!("  Is false still valid? {}", tribute_handle_is_valid(false1));
    println!("  Is nil still valid? {}", tribute_handle_is_valid(nil1));
    println!("  Is num1 still valid? {}", tribute_handle_is_valid(num1));
    println!("  => Only interned values survive clear_all");
    
    println!("\n=== Demo Complete ===");
}