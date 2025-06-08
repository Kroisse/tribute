# Testing Framework Plan

## Overview
Build a comprehensive testing framework for Tribute with unit testing, property-based testing, benchmarking, and code coverage.

## Priority: Medium (7/8)
Essential for language quality and ecosystem confidence, can be developed alongside other features.

## Core Features

### Unit Testing
```tribute
// Built-in test syntax
test "addition works correctly" {
  assert_eq(add(2, 3), 5)
  assert_ne(add(2, 3), 6)
  assert(add(2, 3) > 0)
}

test "error handling" {
  assert_error(divide_by_zero(), DivisionByZero)
}
```

### Property-Based Testing
```tribute
// Generative testing
property "addition is commutative" {
  forall x: Int, y: Int in {
    assert_eq(add(x, y), add(y, x))
  }
}

property "reverse twice is identity" {
  forall list: Vec<Int> in {
    assert_eq(list.reverse().reverse(), list)
  }
}
```

### Benchmarking
```tribute
// Performance testing
benchmark "list sorting" {
  let data = generate_random_list(1000)
  measure {
    data.sort()
  }
}
```

## Implementation Strategy

### Phase 1: Core Test Runner
1. Test discovery and execution engine
2. Basic assertion macros
3. Test result reporting
4. Integration with `trb test` command

### Phase 2: Advanced Assertions
1. Rich assertion library with clear error messages
2. Custom assertion macros
3. Async test support
4. Test fixtures and setup/teardown

### Phase 3: Property-Based Testing
1. Random data generation for built-in types
2. Shrinking algorithm for minimal failing cases
3. Custom generator definitions
4. Stateful property testing

### Phase 4: Performance Testing
1. Microbenchmarking framework
2. Statistical analysis of results
3. Regression detection
4. Performance CI integration

### Phase 5: Code Coverage
1. Instrumentation-based coverage
2. Line and branch coverage metrics
3. HTML and text reports
4. Coverage thresholds and CI integration

## Technical Design

### Test Discovery
- Automatic discovery of test functions
- Module-based test organization
- Tag-based test filtering
- Parallel test execution

### Assertion Framework
```tribute
// Rich assertions with helpful error messages
assert_eq(actual, expected)  // Shows diff on failure
assert_almost_eq(3.14, pi, 0.01)  // Floating point comparison
assert_contains(list, item)  // Collection membership
assert_matches(result, Ok(_))  // Pattern matching assertions
```

### Property Testing Engine
- QuickCheck-inspired API
- Configurable test case generation
- Deterministic replay with seeds
- Integration with type system for automatic generators

### Test Output
```
Running 15 tests...
✓ addition works correctly
✓ error handling
✗ division by zero (line 42)
  Expected: DivisionByZero
  Actual:   Panic("divide by zero")

Property tests:
✓ addition is commutative (100 cases)
✗ list reverse (failed after 23 cases)
  Failing input: [1, 2, -2147483648]
  Shrunk to: [-2147483648]

Results: 13 passed, 2 failed
```

## Integration Points

### Package Manager
- Test dependencies separate from runtime dependencies
- Test-only imports and modules
- CI/CD integration hooks

### LSP Support
- Test result annotations in editor
- Run individual tests from IDE
- Test coverage visualization

### Compiler Integration
- Test compilation in parallel
- Dead code elimination for test code
- Debug symbol generation for coverage

## Dependencies
- Standard library (for core utilities)
- Random number generation
- File I/O for test discovery
- Network for CI integration

## Success Criteria
- Fast test execution (< 100ms startup overhead)
- Clear and actionable error messages
- Reliable property-based test generation
- Accurate code coverage reporting
- Seamless CI/CD integration