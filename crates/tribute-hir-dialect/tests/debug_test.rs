//! Debug test to isolate the segfault issue

use melior::{Context, ir::{Location, Module}};
use tribute_hir_dialect::dialect::TributeDialect;
use tribute_hir_dialect::ops::TributeOps;

#[test]
fn test_basic_mlir_creation() {
    // Test 1: Can we create a context?
    let context = Context::new();
    
    // Test 2: Can we create a location?
    let location = Location::unknown(&context);
    
    // Test 3: Can we create a module?
    let module = Module::new(location);
    
    println!("Basic MLIR objects created successfully");
}

#[test]
fn test_dialect_creation() {
    let context = Context::new();
    let dialect = TributeDialect::new(&context);
    let location = dialect.unknown_location();
    
    println!("Dialect created successfully");
}

#[test]
fn test_ops_creation() {
    let context = Context::new();
    let ops = TributeOps::new(&context);
    let location = Location::unknown(&context);
    
    // Test creating a constant
    let result = ops.constant_f64(42.0, location);
    assert!(result.is_ok());
    
    println!("Ops created successfully");
}

#[test]
fn test_func_creation() {
    let context = Context::new();
    let ops = TributeOps::new(&context);
    let location = Location::unknown(&context);
    
    // Test creating a function
    let param_types = vec![ops.types().value_type()];
    let result_types = vec![ops.types().value_type()];
    
    let result = ops.func("test_func", &param_types, &result_types, location);
    if let Err(e) = &result {
        println!("Function creation failed: {:?}", e);
    }
    assert!(result.is_ok());
    
    println!("Function created successfully");
}