# Tribute

[![codecov](https://codecov.io/github/Kroisse/tribute/graph/badge.svg?token=T09M1R6UGG)](https://codecov.io/github/Kroisse/tribute)

A pure, practical functional language that's easy to learn with simple and
familiar syntax.

## Features

- **Static Types with Inference**: Full type inference including ability
  (effect) inference
- **Algebraic Effects**: Unison-style abilities with delimited, one-shot
  continuations
- **Familiar Syntax**: ML semantics with C/Rust-style syntax
  (`fn`, `{}`, `;`, `struct`/`enum`)
- **Multiple Targets**: Cranelift (native) and WasmGC

Current implementation support is narrower than the language design. See the
[compiler capability matrix](new-plans/capabilities.md) for audited frontend,
native, and WasmGC status. Compilation alone is not execution evidence.

## Quick Example

```text
use abilities::Throw
use std::io::{Io, print_line}

fn fail() ->{Throw(String)} String {
    Throw::throw("the failure was handled")
}

fn recover() -> String {
    handle fail() {
        do value { value }
        op Throw::throw(message) { "Recovered: " <> message }
    }
}

fn main() ->{Io} Nil {
    print_line(recover())
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
cargo nextest run --workspace

# Compile one source file to a native executable
cargo run -- compile lang-examples/native_effects.trb \
  -o target/native-effects-example

# Run the resulting executable
./target/native-effects-example
```

The `tribute` CLI compiles one source file at a time; file-module and package
compilation are not implemented. There is no `run` subcommand. `compile`
defaults to `--target native` and accepts these targets:

- `native` writes a linked executable.
- `wasm` writes a WasmGC module. Execute it with a compatible external runtime.
- `none` validates the frontend and shared pipeline without writing an artifact.

Use `-o` or `--output` to choose the artifact path. Without it, native removes
the `.trb` extension and Wasm replaces it with `.wasm`.

## Language Examples

See [`lang-examples/README.md`](lang-examples/README.md) for exact native, Wasm,
and invalid-example commands. It also classifies historical files as regression
fixtures, design-only examples, or legacy samples.

## Development

```bash
# Focused package tests
cargo nextest run -p tribute
cargo nextest run -p tribute-passes

# Review snapshot changes when a snapshot test fails
cargo insta review
```

## Design Documents

The `new-plans/` directory contains the authoritative language and compiler
design. [`new-plans/capabilities.md`](new-plans/capabilities.md) separately
records what the current implementation has demonstrated; design intent alone
is not a support claim.

## About the Name

Tribute started as a testbed for practicing language implementation before
building a "real" language. It originally had a Lisp-like S-expression syntax
with a simple interpreter. The name comes from the Tenacious D song — this
isn't the greatest language in the world, it's just a tribute to one.
