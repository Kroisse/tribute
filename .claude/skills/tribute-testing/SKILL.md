---
name: tribute-testing
description: |
  Tribute project testing guide covering test commands, Salsa incremental
  computation framework test patterns, nextest usage, and insta snapshot
  testing. Use when:
  (1) Running or writing tests
  (2) Writing tests for Salsa tracked functions or accumulate()
  (3) Using the #[salsa_test] macro
  (4) Resolving "cannot accumulate values outside of an active tracked function" errors
  (5) Following Tribute project test conventions
---

# Tribute Testing Guide

## Running Tests

```bash
cargo nextest run --workspace           # All tests (preferred)
cargo nextest run -p tribute            # Specific crate
cargo nextest run -p tree-sitter-tribute
cargo nextest run -p tribute-passes
cargo nextest run -p trunk-ir
cargo insta review                      # Review snapshot test failures
```

## Salsa Test Patterns

### #[salsa_test] Macro

The `#[salsa_test]` macro from `salsa_test_macros` automatically provides a
Salsa database context.

```rust
use salsa_test_macros::salsa_test;

#[salsa_test]
fn test_example(db: &salsa::DatabaseImpl) {
    // Test code using db
}
```

Generated code:

```rust
#[test]
fn test_example() {
    salsa::Database::attach(&salsa::DatabaseImpl::default(), |db| {
        // Test code
    });
}
```

### accumulate() Test Constraints

**Key constraint**: `Diagnostic.accumulate(db)` must be called inside a
`#[salsa::tracked]` function.

`#[salsa_test]` attaches a database but does NOT create a tracked function context.
Unit tests that directly call code using accumulate() will fail:

```text
cannot accumulate values outside of an active tracked function
```

**Solution**: Test diagnostic accumulation at the integration level via
tracked queries.

```rust
// ❌ Direct accumulate in unit test - FAILS
#[salsa_test]
fn test_unresolved_name(db: &salsa::DatabaseImpl) {
    let resolver = Resolver::new(db, env, span_map);
    resolver.resolve_name(&unresolved);  // calls accumulate() internally → error!
}

// ✅ Integration test via tracked query - WORKS
#[salsa_test]
fn test_diagnostics(db: &salsa::DatabaseImpl) {
    let source = make_source(db, "fn main() { undefined_var }");
    let _module = resolved_module(db, source);  // tracked function
    let diagnostics = resolved_module::accumulated::<Diagnostic>(db, source);
    assert!(!diagnostics.is_empty());
}
```

### Test Level Separation

| Level | Test Target | accumulate OK |
| ----- | ----------- | ------------- |
| Unit tests | Pure logic (bindings, scopes) | ❌ |
| Integration tests | Tracked queries + diagnostics | ✅ |

## Snapshot Testing (insta)

```rust
use insta::assert_snapshot;

#[test]
fn test_ir_output() {
    let output = compile_to_ir("fn main() { 42 }");
    assert_snapshot!(output);
}
```

Update snapshots: `cargo insta review`

## Collecting Diagnostics

Collect accumulated Diagnostics from a tracked query:

```rust
let diagnostics: Vec<Diagnostic> =
    query_function::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect();
```

See `guides/salsa.md` for detailed Salsa usage documentation.
