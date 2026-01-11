# Testing Framework Plan

## Overview

Testing framework for Tribute. Design is still evolving, but will be deeply integrated with the ability system.

## Priority: Medium (7/8)

Essential for language quality and ecosystem confidence.

## Core Concepts

### Using Exception for Tests

Tests can use the `Exception` ability for failures:

```rust
fn assert(cond: Bool) ->{Exception} Nil {
    if cond { Nil } else { Exception::raise("assertion failed") }
}
```

Test declaration syntax is TBD. Possibilities include:
- Annotation/decorator style: `@test fn addition_works() { ... }`
- Explicit registration via module-level list
- Special file/module naming convention

Semantics of annotations (metadata vs transformation) need careful design.

### Effect Mocking

The ability system naturally supports mocking - just provide a test handler:

```rust
test "fetch_user calls the API" {
    let calls = ref([])

    handle fetch_user("123") {
        { value } -> value
        { IO::http_get(url) -> k } -> {
            calls := List::push_back(!calls, url)
            k(mock_response())
        }
    }

    Test::assert(!calls == ["/users/123"])
}
```

### Property-Based Testing

Property-based testing requires exploration of multiple values. Since we use one-shot continuations (no backtracking), this will likely use a functional random generator approach rather than an ability:

```rust
fn forall(gen: Gen(a), prop: fn(a) ->{Test} Nil) -> TestResult
```

Design details TBD.

## Open Questions

- Exact syntax for test declarations
- Integration with build system / test runner
- Shrinking strategy for property tests
- Code coverage instrumentation approach

## Success Criteria

- Clean integration with ability system
- Natural mocking via effect handlers
- Clear and actionable error messages
