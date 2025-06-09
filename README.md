# Tribute

A Lisp-like programming language interpreter written in Rust.

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
cargo run --bin trbc <file.trb>
```

## Development

### Tree-sitter Grammar Development

```bash
# Generate parser from grammar.js (run from tree-sitter-tribute/)
cd tree-sitter-tribute
tree-sitter generate

# Test the grammar
tree-sitter test

# Build WASM parser
tree-sitter build-wasm
```

## Language Examples

See `lang-examples/` directory for sample `.trb` files demonstrating the language syntax.