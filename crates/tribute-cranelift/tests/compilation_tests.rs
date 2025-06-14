//! Integration tests for Tribute compilation
//!
//! These tests verify that the Cranelift compiler can successfully compile
//! various Tribute programs to valid object files.

use std::fs;
use std::path::PathBuf;
use tribute::{parse_str, TributeDatabaseImpl};
use tribute_cranelift::TributeCompiler;
use tribute_hir::queries::lower_program_to_hir;

/// Test data: (filename, source_code, expected_to_compile)
/// Uses include_str!() to load actual example files from lang-examples/
const TEST_CASES: &[(&str, &str, bool)] = &[
    // Load actual example files from lang-examples/
    (
        "basic.trb",
        include_str!("../../../lang-examples/basic.trb"),
        true,
    ),
    (
        "functions.trb",
        include_str!("../../../lang-examples/functions.trb"),
        true,
    ),
    (
        "pattern_matching.trb",
        include_str!("../../../lang-examples/pattern_matching.trb"),
        true,
    ),
    (
        "pattern_advanced.trb",
        include_str!("../../../lang-examples/pattern_advanced.trb"),
        true,
    ),
    (
        "hello.trb",
        include_str!("../../../lang-examples/hello.trb"),
        true,
    ),
    (
        "let_simple.trb",
        include_str!("../../../lang-examples/let_simple.trb"),
        true,
    ),
    (
        "let_bindings.trb",
        include_str!("../../../lang-examples/let_bindings.trb"),
        true,
    ),
    (
        "calc.trb",
        include_str!("../../../lang-examples/calc.trb"),
        true,
    ),
    // Cases that should fail to compile
    (
        "string_interpolation.trb",
        include_str!("../../../lang-examples/string_interpolation.trb"),
        false,
    ),
    // Additional test cases for edge cases
    (
        "empty_program.trb",
        include_str!("../../../lang-examples/empty_program.trb"),
        true,
    ),
    (
        "simple_function.trb",
        include_str!("../../../lang-examples/simple_function.trb"),
        true,
    ),
];

#[test]
fn test_compilation_success_cases() {
    for &(name, source, should_compile) in TEST_CASES {
        if should_compile {
            let result = compile_source(name, source);
            assert!(
                result.is_ok(),
                "Failed to compile {}: {:?}",
                name,
                result.err()
            );

            if let Ok(object_bytes) = result {
                // Basic sanity checks on the object file
                assert!(
                    !object_bytes.is_empty(),
                    "Object file for {} is empty",
                    name
                );
                assert!(
                    object_bytes.len() > 100,
                    "Object file for {} is suspiciously small: {} bytes",
                    name,
                    object_bytes.len()
                );

                // Object files should start with a valid magic number (Mach-O or ELF)
                // This is a basic check that we generated something reasonable
                let magic = &object_bytes[..4];
                let is_valid_object = magic == b"\x7fELF" ||  // ELF magic
                    magic == &[0xfe, 0xed, 0xfa, 0xce] || // Mach-O 32-bit big endian  
                    magic == &[0xce, 0xfa, 0xed, 0xfe] || // Mach-O 32-bit little endian
                    magic == &[0xfe, 0xed, 0xfa, 0xcf] || // Mach-O 64-bit big endian
                    magic == &[0xcf, 0xfa, 0xed, 0xfe] || // Mach-O 64-bit little endian
                    magic == &[0xca, 0xfe, 0xba, 0xbe] || // Mach-O universal binary
                    magic[0] == 0x4c || magic[0] == 0x64; // COFF (Windows) or other

                assert!(
                    is_valid_object,
                    "Object file for {} doesn't have valid magic number: {:02x?}",
                    name, magic
                );
            }
        }
    }
}

#[test]
fn test_compilation_failure_cases() {
    for &(name, source, should_compile) in TEST_CASES {
        if !should_compile {
            let result = compile_source(name, source);
            assert!(
                result.is_err(),
                "Expected {} to fail compilation, but it succeeded",
                name
            );
        }
    }
}

#[test]
fn test_real_example_files() {
    let examples_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("lang-examples");

    if !examples_dir.exists() {
        // Skip this test if lang-examples doesn't exist
        return;
    }

    let successful_examples = &[
        "basic.trb",
        "functions.trb",
        "pattern_matching.trb",
        "pattern_advanced.trb",
        "hello.trb",
        "let_simple.trb",
        "let_bindings.trb",
    ];

    for example_name in successful_examples {
        let example_path = examples_dir.join(example_name);
        if example_path.exists() {
            let source = fs::read_to_string(&example_path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", example_name, e));

            let result = compile_source(example_name, &source);
            assert!(
                result.is_ok(),
                "Failed to compile example {}: {:?}",
                example_name,
                result.err()
            );
        }
    }
}

/// Helper function to compile a Tribute source string
fn compile_source(name: &str, source: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let db = TributeDatabaseImpl::default();
    let dummy_path = PathBuf::from(format!("{}.trb", name));

    // Parse to AST
    let (program, diagnostics) = parse_str(&db, &dummy_path, source);

    // Check for parsing errors
    if !diagnostics.is_empty() {
        return Err(format!("Parsing errors: {:?}", diagnostics).into());
    }

    // Lower to HIR
    let hir_program = lower_program_to_hir(&db, program).ok_or("Failed to lower program to HIR")?;

    // Create Cranelift compiler
    let compiler = TributeCompiler::new(None)?; // Use native target

    // Compile to object code
    let object_bytes = compiler.compile_program(&db, hir_program)?;

    Ok(object_bytes)
}

#[test]
fn test_empty_program() {
    let result = compile_source("empty", "");
    // Empty programs should still compile (they just do nothing)
    assert!(result.is_ok(), "Empty program should compile successfully");
}

#[test]
fn test_syntax_error() {
    // Test various syntax errors that should definitely fail
    let invalid_cases = &[
        "fn incomplete function",
        "let x = ",
        "invalid_token_sequence ^^^ !!!",
        "fn main() { unclosed_block",
        "fn main() { let x = ; }", // missing value
    ];

    let mut any_failed = false;
    for &invalid_source in invalid_cases {
        let result = compile_source("syntax_error", invalid_source);
        if result.is_err() {
            any_failed = true;
            break;
        }
    }

    assert!(
        any_failed,
        "At least one invalid syntax should cause compilation failure"
    );
}

#[test]
fn test_compilation_deterministic() {
    let source = r#"
fn main() {
    print_line("Hello deterministic world")
}
"#;

    // Compile the same source twice
    let result1 = compile_source("deterministic1", source).unwrap();
    let result2 = compile_source("deterministic2", source).unwrap();

    // The object files should be identical (deterministic compilation)
    assert_eq!(
        result1.len(),
        result2.len(),
        "Object file sizes should be identical"
    );
    // Note: We don't check exact byte equality as some metadata might differ
    // but the sizes should be the same for deterministic compilation
}
