//! Compile-time rejection of malformed typed `#[dialect]` definitions.

#[test]
fn typed_dialect_definitions() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
