//! Basic unit tests for tribute-hir-dialect

use melior::Context;
use tribute_hir_dialect::{
    dialect::TributeDialect,
    types::TributeTypes,
    ops::TributeOps,
};

#[test]
fn test_dialect_creation() {
    let context = Context::new();
    let dialect = TributeDialect::new(&context);
    
    // Test that we can create basic locations
    let location = dialect.unknown_location();
    // Location is a value type, so just check that we can create it
    
    // Test module creation
    let _module = dialect.create_module(location);
    // Module is a value type, so just check that we can create it
}

#[test]
fn test_type_system() {
    let context = Context::new();
    let types = TributeTypes::new(&context);
    
    // Test basic type creation - these should not panic
    let _value_type = types.value_type();
    let _string_type = types.string_type();
    let f64_type = types.f64_type();
    let _i64_type = types.i64_type();
    let _bool_type = types.bool_type();
    
    // Test runtime type conversion
    let _runtime_type = types.to_runtime_type(f64_type);
    
    // Test function type creation
    let _func_type = types.function_type(2);
    
    // If we reach here without panicking, the types were created successfully
    assert!(true);
}

#[test]
fn test_operation_creation() {
    let context = Context::new();
    let ops = TributeOps::new(&context);
    let unknown_location = melior::ir::Location::unknown(&context);
    
    // Test constant creation
    let const_f64 = ops.constant_f64(42.0, unknown_location);
    assert!(const_f64.is_ok());
    
    let const_string = ops.constant_string("hello", unknown_location);
    assert!(const_string.is_ok());
    
    // Test that operations can be created (we can't test execution without full MLIR setup)
    let _const_op = const_f64.unwrap();
    let _string_op = const_string.unwrap();
    
    // If we reach here without panicking, the operations were created successfully
    assert!(true);
}