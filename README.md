# Tribute

[![codecov](https://codecov.io/github/Kroisse/tribute/graph/badge.svg?token=T09M1R6UGG)](https://codecov.io/github/Kroisse/tribute)

A pure, practical functional language that's easy to learn with simple and familiar syntax.

## Features

- **Static Types with Inference**: Full type inference including ability (effect) inference
- **Algebraic Effects**: Unison-style abilities with delimited, one-shot continuations
- **Familiar Syntax**: ML semantics with C/Rust-style syntax (`fn`, `{}`, `;`, `struct`/`enum`)
- **Multiple Targets**: Cranelift (native) and WasmGC

## Quick Example

```tribute
struct User { name: String, age: Int }

fn greet(user: User) -> String {
    "Hello, \{user.name}!"                // string interpolation
}

fn insa(user: User) -> String {
    "안녕, " <> user.name <> "!"           // string concatenation
}

// Abilities (algebraic effects)
fn fetch_user(id: UserId) ->{Http, Async} User {
    let response = Http::get("/users/\{id}")
    response.await
}

// Function chaining powered by type-directed name resolution
fn process(data: List(Int)) -> Int {
    data
        .filter(fn(x) x > 0)
        .map(fn(x) x * 2)
        .fold(0, fn(a, b) a + b)
}

// Equivalent to:
fn process2(data: List(Int)) -> Int {
    List::fold(
        List::map(
            List::filter(
                data,
                fn(x) x > 0
            ),
            fn(x) x * 2
        ),
        0,
        fn(a, b) a + b
    )
}
```

## Setup

After cloning the repository, set up the git hooks:

```bash
git config core.hooksPath .ci/githooks
```

For Tree-sitter grammar development, install the Tree-sitter CLI:

```bash
npm install -g tree-sitter-cli
```

## Building and Running

```bash
# Build the entire workspace
cargo build

# Run all tests
cargo test

# Run the compiler on a .trb file
cargo run --bin trbc -- <file.trb>

# If snapshot tests fail (insta)
cargo insta review
```

## Development

```bash
# Package-specific tests
cargo test -p tribute-core
cargo test -p tribute-passes
cargo test -p tribute-trunk-ir
```

## Language Examples

See `lang-examples/` directory for sample `.trb` files demonstrating the language syntax.

## Design Documents

The `new-plans/` directory contains the authoritative design documents for the language and compiler. If code or tests conflict with these documents, the documents are considered correct.

## About the Name

Tribute started as a testbed for practicing language implementation before building a "real" language. It originally had a Lisp-like S-expression syntax with a simple interpreter. The name comes from the Tenacious D song — this isn't the greatest language in the world, it's just a tribute to one.
