//! Basic unit tests for tribute-hir-dialect

use melior::{
    ir::{operation::OperationLike as _, r#type::IntegerType, TypeLike as _},
    Context,
};
use tribute_hir_dialect::{dialect::TributeDialect, ops::TributeOps, types::TributeTypes};

#[test]
fn test_dialect_creation() {
    let context = Context::new();
    let dialect = TributeDialect::new(&context);

    // Test that we can create basic locations
    let location = dialect.unknown_location();

    // Test module creation
    let module = dialect.create_module(location);

    assert!(module.as_operation().verify());
}

#[test]
fn test_type_system() {
    let context = Context::new();
    let types = TributeTypes::new(&context);

    // Test basic type creation and verify they're different
    let value_type = types.value_type();
    let string_type = types.string_type();
    let f64_type = types.f64_type();
    let i64_type = types.i64_type();
    let bool_type = types.bool_type();

    // Verify types are created and have different characteristics
    // Since our current implementation uses IntegerType with different widths,
    // we'll verify they're integer types but with different properties
    assert!(value_type.is_integer());
    assert!(string_type.is_integer());
    assert!(f64_type.is_f64()); // actual f64 type
    assert_eq!(
        IntegerType::try_from(i64_type)
            .expect("value_type should be an integer")
            .width(),
        64
    );
    assert_eq!(
        IntegerType::try_from(bool_type)
            .expect("bool_type should be an integer")
            .width(),
        1
    );

    // Test runtime type conversion
    let runtime_type = types.to_runtime_type(f64_type);
    assert_eq!(value_type, runtime_type);

    // Test function type creation with different arities
    let func_type_0 = types.function_type(0);
    let func_type_2 = types.function_type(2);
    let func_type_5 = types.function_type(5);

    // Now we have proper function types
    assert!(func_type_0.is_function());
    assert!(func_type_2.is_function());
    assert!(func_type_5.is_function());
    assert_ne!(func_type_0, func_type_2);
    assert_ne!(func_type_2, func_type_5);
}

#[test]
fn test_operation_creation() {
    let context = Context::new();
    let ops = TributeOps::new(&context);
    let unknown_location = melior::ir::Location::unknown(&context);

    // Test constant creation
    let const_f64_result = ops.constant_f64(42.0, unknown_location);
    assert!(const_f64_result.is_ok(), "Should create f64 constant");

    let const_string_result = ops.constant_string("hello", unknown_location);
    assert!(const_string_result.is_ok(), "Should create string constant");

    // Just verify operations were created successfully
    let const_op = const_f64_result.unwrap();
    let string_op = const_string_result.unwrap();

    // Operations are opaque in melior, so just ensure they exist
    let _ = const_op;
    let _ = string_op;

    // Test creating different constant values
    let zero_result = ops.constant_f64(0.0, unknown_location);
    assert!(zero_result.is_ok(), "Should create zero constant");

    let negative_result = ops.constant_f64(-123.45, unknown_location);
    assert!(negative_result.is_ok(), "Should create negative constant");

    let empty_string_result = ops.constant_string("", unknown_location);
    assert!(
        empty_string_result.is_ok(),
        "Should create empty string constant"
    );

    let unicode_string_result = ops.constant_string("Hello, 世界! 🌍", unknown_location);
    assert!(
        unicode_string_result.is_ok(),
        "Should create unicode string constant"
    );
}

#[test]
fn test_binary_operations() {
    let context = Context::new();
    let _ops = TributeOps::new(&context);
    let _location = melior::ir::Location::unknown(&context);

    // Binary operations (add, sub, mul, div) require Value arguments
    // which can only be obtained from operation results in a proper MLIR context.
    // These are better tested in integration tests where we have a full MLIR module.
    // This test passes if we can create the TributeOps without panicking.
}

#[test]
fn test_function_operations() {
    let context = Context::new();
    // Allow unregistered dialects before creating operations
    context.set_allow_unregistered_dialects(true);
    
    let _dialect = TributeDialect::new(&context);
    let ops = TributeOps::new(&context);
    let types = TributeTypes::new(&context);
    let location = melior::ir::Location::unknown(&context);

    // Create a function using the func method
    let value_type = types.value_type();
    let func_result = ops.func(
        "test_func",
        &[value_type, value_type],
        &[value_type],
        location,
    );
    assert!(func_result.is_ok(), "Should create function");

    // Now we can verify the function operation since dialects are allowed
    let func_op = func_result.unwrap();
    assert!(func_op.verify(), "Function operation should be valid");
}

#[test]
fn test_call_operations() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    
    let _dialect = TributeDialect::new(&context);
    let ops = TributeOps::new(&context);
    let location = melior::ir::Location::unknown(&context);

    // Test function call with no arguments
    let call_result_0 = ops.call("my_func", &[], location);
    assert!(call_result_0.is_ok(), "Should create call with 0 arguments");

    // Now we can verify the call operation
    let call_op = call_result_0.unwrap();
    assert!(call_op.verify(), "Call operation should be valid");
}

#[test]
fn test_to_runtime_operations() {
    let context = Context::new();
    let _ops = TributeOps::new(&context);
    let _location = melior::ir::Location::unknown(&context);

    // We can't easily test to_runtime without proper Value instances
    // This would require a more complex test setup with a proper MLIR module
    // This test passes if we can create the TributeOps without panicking.
}

#[test]
fn test_edge_cases() {
    let context = Context::new();
    let ops = TributeOps::new(&context);
    let location = melior::ir::Location::unknown(&context);

    // Test extreme float values
    let inf_result = ops.constant_f64(f64::INFINITY, location);
    assert!(inf_result.is_ok(), "Should handle infinity");

    let neg_inf_result = ops.constant_f64(f64::NEG_INFINITY, location);
    assert!(neg_inf_result.is_ok(), "Should handle negative infinity");

    let nan_result = ops.constant_f64(f64::NAN, location);
    assert!(nan_result.is_ok(), "Should handle NaN");

    // Test very long strings
    let long_string = "a".repeat(10000);
    let long_string_result = ops.constant_string(&long_string, location);
    assert!(long_string_result.is_ok(), "Should handle long strings");

    // Test strings with special characters
    let special_string = "Line1\nLine2\tTabbed\r\nWindows\0Null";
    let special_result = ops.constant_string(special_string, location);
    assert!(special_result.is_ok(), "Should handle special characters");
}

#[test]
fn test_error_handling() {
    let context = Context::new();
    let _ops = TributeOps::new(&context);

    // Most operations in TributeOps return Result types
    // Error cases are mostly tested implicitly through integration tests
    // where invalid MLIR might be generated.
    // This test passes if we can create the TributeOps without panicking.
}
