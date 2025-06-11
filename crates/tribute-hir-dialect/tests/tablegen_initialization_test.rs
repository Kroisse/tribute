use tribute_hir_dialect::{TributeDialect, Context};

#[test]
fn test_tablegen_dialect_initialization() {
    let context = Context::new();
    let _dialect = TributeDialect::new(&context);
    
    // The initialization should have been called during TributeDialect::new()
    // and should display information about the loaded dialect
    
    // Verify that our generated constants are accessible
    use tribute_hir_dialect::{dialect_info, generated_ops, operation_metadata};
    
    assert_eq!(dialect_info::NAME, "tribute");
    assert_eq!(dialect_info::NAMESPACE, "mlir::tribute");
    assert!(dialect_info::SUMMARY.contains("Tribute"));
    
    // Test that operation constants are available
    assert_eq!(generated_ops::ADD, "tribute.add");
    assert_eq!(generated_ops::CONSTANT, "tribute.constant");
    assert_eq!(generated_ops::FUNC, "tribute.func");
    
    // Test operation count
    assert_eq!(operation_metadata::OPERATION_COUNT, 11);
    
    println!("TableGen dialect initialization test completed successfully!");
    println!("Dialect: {}", dialect_info::NAME);
    println!("Namespace: {}", dialect_info::NAMESPACE);
    println!("Operations: {}", operation_metadata::OPERATION_COUNT);
}

#[test]
fn test_initialization_function_directly() {
    // Test calling the initialization function directly
    let result = tribute_hir_dialect::initialization::initialize_tribute_dialect();
    assert!(result.is_ok(), "Dialect initialization should succeed");
}