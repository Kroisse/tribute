//! Basic unit tests for tribute-hir-dialect

use melior::Context;
use tribute_hir_dialect::{
    dialect::TributeDialect,
    types::TributeTypes,
    ops::TributeOps,
    evaluator::{MLIREvaluator, TributeValue},
};

#[test]
fn test_dialect_creation() {
    let context = Context::new();
    let dialect = TributeDialect::new(&context);
    
    // Test that we can create basic locations
    let location = dialect.unknown_location();
    // Location is a value type, so just check that we can create it
    
    // Test module creation
    let module = dialect.create_module(location);
    // Module is a value type, so just check that we can create it
}

#[test]
fn test_type_system() {
    let context = Context::new();
    let types = TributeTypes::new(&context);
    
    // Test basic type creation - these should not panic
    let value_type = types.value_type();
    let string_type = types.string_type();
    let f64_type = types.f64_type();
    let i64_type = types.i64_type();
    let bool_type = types.bool_type();
    
    // Test runtime type conversion
    let runtime_type = types.to_runtime_type(f64_type);
    
    // Test function type creation
    let func_type = types.function_type(2);
    
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
    let const_op = const_f64.unwrap();
    let string_op = const_string.unwrap();
    
    // If we reach here without panicking, the operations were created successfully
    assert!(true);
}

#[test]
fn test_evaluator_values() {
    // Test TributeValue operations
    let num = TributeValue::Number(42.0);
    let string = TributeValue::String("hello".to_string());
    let boolean = TributeValue::Boolean(true);
    let nil = TributeValue::Unit;
    let list = TributeValue::List(vec![num.clone(), string.clone()]);
    
    // Test string conversion
    assert_eq!(format!("{}", num), "42");
    assert_eq!(format!("{}", string), "hello");
    assert_eq!(format!("{}", boolean), "true");
    assert_eq!(format!("{}", nil), "()");
    assert_eq!(format!("{}", list), "[42, hello]");
    
    // Test truthiness
    assert!(num.is_truthy());
    assert!(string.is_truthy());
    assert!(boolean.is_truthy());
    assert!(!nil.is_truthy());
    assert!(list.is_truthy());
    
    // Test falsy values
    let zero = TributeValue::Number(0.0);
    let empty_string = TributeValue::String("".to_string());
    let false_bool = TributeValue::Boolean(false);
    let empty_list = TributeValue::List(vec![]);
    
    assert!(!zero.is_truthy());
    assert!(!empty_string.is_truthy());
    assert!(!false_bool.is_truthy());
    assert!(!empty_list.is_truthy());
}

#[test]
fn test_evaluator_binary_ops() {
    let context = Context::new();
    let location = melior::ir::Location::unknown(&context);
    let module = melior::ir::Module::new(location);
    let evaluator = MLIREvaluator::new(&context, module);
    
    // Test numeric operations
    let left = TributeValue::Number(10.0);
    let right = TributeValue::Number(5.0);
    
    let add_result = evaluator.binary_op("add", left.clone(), right.clone());
    assert!(add_result.is_ok());
    if let Ok(TributeValue::Number(n)) = add_result {
        assert_eq!(n, 15.0);
    }
    
    let sub_result = evaluator.binary_op("sub", left.clone(), right.clone());
    assert!(sub_result.is_ok());
    if let Ok(TributeValue::Number(n)) = sub_result {
        assert_eq!(n, 5.0);
    }
    
    let mul_result = evaluator.binary_op("mul", left.clone(), right.clone());
    assert!(mul_result.is_ok());
    if let Ok(TributeValue::Number(n)) = mul_result {
        assert_eq!(n, 50.0);
    }
    
    let div_result = evaluator.binary_op("div", left, right);
    assert!(div_result.is_ok());
    if let Ok(TributeValue::Number(n)) = div_result {
        assert_eq!(n, 2.0);
    }
    
    // Test string concatenation
    let left_str = TributeValue::String("hello".to_string());
    let right_str = TributeValue::String(" world".to_string());
    
    let concat_result = evaluator.binary_op("add", left_str, right_str);
    assert!(concat_result.is_ok());
    if let Ok(TributeValue::String(s)) = concat_result {
        assert_eq!(s, "hello world");
    }
    
    // Test division by zero
    let zero = TributeValue::Number(0.0);
    let num = TributeValue::Number(10.0);
    let div_zero_result = evaluator.binary_op("div", num, zero);
    assert!(div_zero_result.is_err());
}

#[test]
fn test_evaluator_builtin_functions() {
    let context = Context::new();
    let location = melior::ir::Location::unknown(&context);
    let module = melior::ir::Module::new(location);
    let mut evaluator = MLIREvaluator::new(&context, module);
    
    // Test print_line (should not error, but we can't test actual output)
    let args = vec![TributeValue::String("test message".to_string())];
    let result = evaluator.call_function("print_line", args);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!(matches!(value, TributeValue::Unit));
    }
    
    // Test input_line function exists but skip actual execution in test environment
    // (would hang waiting for stdin input)
    // We can still test argument validation
    let result = evaluator.call_function("input_line", vec![TributeValue::String("unexpected_arg".to_string())]);
    assert!(result.is_err(), "input_line should reject arguments");
    
    // Test invalid function
    let result = evaluator.call_function("nonexistent", vec![]);
    assert!(result.is_err());
    
    // Test print_line with wrong number of args
    let result = evaluator.call_function("print_line", vec![]);
    assert!(result.is_err());
    
    let result = evaluator.call_function("print_line", vec![
        TributeValue::String("arg1".to_string()),
        TributeValue::String("arg2".to_string()),
    ]);
    assert!(result.is_err());
}

#[test]
fn test_evaluator_scopes() {
    let context = Context::new();
    let location = melior::ir::Location::unknown(&context);
    let module = melior::ir::Module::new(location);
    let mut evaluator = MLIREvaluator::new(&context, module);
    
    // Test variable access in global scope
    evaluator.set_variable("global_var".to_string(), TributeValue::Number(42.0));
    let result = evaluator.get_variable("global_var");
    assert!(result.is_ok());
    if let Ok(TributeValue::Number(n)) = result {
        assert_eq!(n, 42.0);
    }
    
    // Test local scope
    evaluator.push_scope();
    evaluator.set_variable("local_var".to_string(), TributeValue::String("local".to_string()));
    
    // Should be able to access both global and local variables
    let global_result = evaluator.get_variable("global_var");
    assert!(global_result.is_ok());
    
    let local_result = evaluator.get_variable("local_var");
    assert!(local_result.is_ok());
    if let Ok(TributeValue::String(s)) = local_result {
        assert_eq!(s, "local");
    }
    
    // Pop scope
    evaluator.pop_scope();
    
    // Global should still be accessible
    let global_result = evaluator.get_variable("global_var");
    assert!(global_result.is_ok());
    
    // Local should no longer be accessible
    let local_result = evaluator.get_variable("local_var");
    assert!(local_result.is_err());
    
    // Test variable not found
    let missing_result = evaluator.get_variable("nonexistent");
    assert!(missing_result.is_err());
}