# Static Type System Plan

## Overview
Add a gradual static type system to Tribute, allowing optional type annotations while maintaining compatibility with untyped code.

## Priority: Low (4/4)
Important for language maturity but requires stable syntax and compiler infrastructure.

## Type System Design

### Core Types
```tribute
// Primitive types
Int, Float, String, Bool, Unit

// Composite types
List<T>, Option<T>, Result<T, E>

// Function types
(Int, Int) -> Int

// User-defined types
struct Point { x: Int, y: Int }
enum Color { Red, Green, Blue }
```

### Type Annotations
```tribute
// Variable annotations
let x: Int = 42
let name: String = "Tribute"

// Function annotations
fn add(a: Int, b: Int) -> Int {
  a + b
}

// Generic functions
fn identity<T>(x: T) -> T {
  x
}
```

## Implementation Steps

### Phase 1: Type Representation
1. Create `tribute-types` crate
2. Define type AST:
   - Primitive types
   - Type variables
   - Function types
   - Generic types
3. Extend parser for type annotations

### Phase 2: Type Inference
1. Implement Hindley-Milner type inference
2. Constraint generation from HIR
3. Unification algorithm
4. Type variable substitution
5. Let-polymorphism

### Phase 3: Type Checking
1. Integrate with HIR lowering
2. Type environment management
3. Error reporting with suggestions
4. Gradual typing boundaries

### Phase 4: Advanced Features
1. Algebraic data types (ADTs)
2. Pattern matching exhaustiveness
3. Trait/interface system
4. Effect system for tracking side effects:
   - Pure functions vs impure functions
   - I/O effects tracking
   - Mutable state tracking
   - Benefits:
     - Enable aggressive compiler optimizations for pure code
     - Catch common bugs like forgetting to handle errors
     - Make code reasoning easier by explicit effect annotations
     - Allow safe parallelization of pure computations
   - Example syntax:
     ```tribute
     fn pure_add(a: Int, b: Int) -> Int { a + b }
     fn print_result(x: Int) -> Unit with IO { print_line(x) }
     ```

### Phase 5: Runtime Integration
1. Type erasure for compilation
2. Runtime type information (optional)
3. Interop with untyped code
4. Performance optimizations

## Design Decisions

### Gradual Typing
- Untyped code defaults to `Any` type
- Explicit casts at typed/untyped boundaries
- No runtime overhead for fully typed code

### Type Inference
- Local type inference (no global inference)
- Explicit annotations for function signatures
- Infer variable types from initialization

### Error Recovery
- Continue checking despite type errors
- Provide helpful error messages
- Suggest type annotations

## Technical Challenges
- Balancing inference power vs simplicity
- Error message quality
- Performance of type checking
- Integration with existing codebase

## Dependencies
- Syntax modernization (for clean annotation syntax)
- HIR stability
- Potentially: external type checker library

## Success Criteria
- Type check standard library
- Catch common type errors
- No false positives
- Reasonable compile-time performance
- Clear migration path for untyped code