# Standard Library Plan

## Overview

Development plan for Tribute's standard library. Designed around the ability system for I/O and error handling.

## Priority: Medium (5/8)

Essential for language usability, can be developed incrementally alongside other features.

## Core Modules

### Primitive Types

```rust
// Built-in types (runtime intrinsics)
Nat          // Natural numbers (0, 1, 2, ...)
Int          // Integers
Float        // Floating point
String       // UTF-8 strings
Bytes        // Raw byte sequences
Bool         // True, False
Nil          // Unit type
```

### Collections

```rust
// Built-in collections
Array(a)             // Contiguous memory array
List(a)              // RRB tree (efficient concat, slice, index)
Dict(k, v)           // HAMT, literal: #{k: v, ...}, pattern: #{k: v, ..rest}
Set(a)               // HAMT-based set, literal: #{v, ...}, pattern: #{v, ..rest}
```

### Option and Result

```rust
enum Option(a) {
    None
    Some(a)
}

enum Result(a, e) {
    Ok(a)
    Error(e)
}
```

## Core Abilities

### IO Ability

A unified ability for all I/O operations (console, file, network, etc.):

```rust
ability IO {
    // Console
    fn print(msg: String) -> Nil
    fn read_line() -> String
    
    // File system, network, etc. - TBD
}

// Usage example
fn greet() ->{IO} Nil {
    IO::print("What is your name? ")
    let name = IO::read_line()
    IO::print("Hello, \{name}!")
}
```

### Abort Ability

Corresponds to `Option` - computation that may fail without an error value:

```rust
ability Abort {
    fn abort() -> !
}

// Handler converts to Option
fn maybe(f: fn() ->{Abort, e} a) ->{e} Option(a) {
    handle f() {
        { value } -> Some(value)
        { Abort::abort() -> _ } -> None
    }
}
```

### Exception Ability

General-purpose failure (no specific error type):

```rust
ability Exception {
    fn raise(msg: String) -> a
}
```

### Throw Ability

Corresponds to `Result` - computation that may fail with a typed error value:

```rust
ability Throw(e) {
    fn throw(error: e) -> a
}

// Handler converts to Result
fn catch(f: fn() ->{Throw(e), r} a) ->{r} Result(a, e) {
    handle f() {
        { value } -> Ok(value)
        { Throw::throw(e) -> _ } -> Error(e)
    }
}
```

### Stream Ability

Generator-style ability for producing sequences of values:

```rust
ability Stream(a) {
    fn emit(a) -> Nil
}

// Handler collects to List (uses push_back, O(1) amortized for RRB tree)
fn collect(f: fn() ->{Stream(a), e} Nil) ->{e} List(a) {
    fn go(acc: List(a)) ->{e} List(a) {
        handle f() {
            { Nil } -> acc
            { Stream::emit(x) -> k } -> go(acc.push_back(x))
        }
    }
    go([])
}
```

### Async Ability

```rust
ability Async {
    fn await(promise: Promise(a)) -> a
    fn spawn(f: fn() ->{e} a) -> Promise(a)
}
```

## Utilities

### String Operations

```rust
mod String {
    fn len(s: String) -> Nat
    fn concat(a: String, b: String) -> String
    fn split(s: String, delimiter: String) -> List(String)
    fn join(parts: List(String), separator: String) -> String
    fn trim(s: String) -> String
    fn contains(s: String, substr: String) -> Bool
}
```

### List Operations

```rust
mod List {
    fn empty() -> List(a)
    fn len(xs: List(a)) -> Nat
    
    // RRB tree operations (O(1) amortized)
    fn push_front(xs: List(a), x: a) -> List(a)  // Also available as [a, ..rest]
    fn push_back(xs: List(a), x: a) -> List(a)  // Aslo available as [..init, a]
    fn concat(xs: List(a), ys: List(a)) -> List(a)  // Also available as [..xs, ..ys]
    // Destructuring via pattern matching: [a, b, ..rest] or [..init, x, y]
    
    // Higher-order functions
    fn map(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)
    fn filter(xs: List(a), p: fn(a) ->{e} Bool) ->{e} List(a)
    fn fold(xs: List(a), init: b, f: fn(b, a) ->{e} b) ->{e} b
}
```

## Implementation Strategy

### Phase 1: Core Foundation

1. Primitive types (Int, String, Bool, etc.)
2. Basic collections (List, Option)
3. IO ability (console operations)

### Phase 2: Error Handling

1. Exception ability
2. Result type and conversion functions

### Phase 3: Advanced Features

1. Async ability
2. Extended IO operations
3. Additional utilities as needed

## Design Principles

1. **Ability-first**: I/O and errors expressed as abilities
2. **Explicit effects**: Effects visible in types
3. **Handler composability**: Various execution strategies via handler composition
4. **GC-managed memory**: No explicit memory management required
5. **Persistent by default**: Collections use persistent data structures

## Success Criteria

- Complete implementation of core collections
- Working IO and Exception abilities
- Basic string/number operations provided
- Documentation and example code available
