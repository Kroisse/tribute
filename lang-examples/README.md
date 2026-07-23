# Tribute language examples

The canonical examples in this directory are checked against the current
single-source-file CLI. They use public APIs and are the best starting point for
running Tribute today.

## Canonical runnable examples

### Native effects and I/O

[`native_effects.trb`](native_effects.trb) handles the public
`abilities::Throw` ability and writes through `std::io`.

```bash
cargo run -- compile lang-examples/native_effects.trb \
  -o target/native-effects-example
./target/native-effects-example
```

Expected standard output:

```text
Recovered: the failure was handled
```

Native currently has the broadest execution coverage, including handled
abilities and `std::io::read_line`.

### WasmGC dynamic output

[`wasm_dynamic_output.trb`](wasm_dynamic_output.trb) builds dynamic `String` and
`Bytes` values with concatenation and writes them through public `std::io`.

```bash
cargo run -- compile --target wasm \
  lang-examples/wasm_dynamic_output.trb \
  -o target/wasm-dynamic-output.wasm
wasmtime -Wgc=y,function-references=y \
  target/wasm-dynamic-output.wasm
```

Expected standard output:

```text
String: dynamic
Bytes: bytes
```

This command requires a recent Wasmtime with WasmGC and function references.
Wasm `read_line` is not supported. The Wasm backend compiles ability programs,
but these examples do not claim ability execution support because that path
lacks an execution test.

## Canonical invalid example

[`invalid_unresolved_name.trb`](invalid_unresolved_name.trb) deliberately refers
to an undefined value. Validate it without producing a target artifact:

```bash
cargo run -- compile --target none \
  lang-examples/invalid_unresolved_name.trb
```

The command must exit unsuccessfully and report:

```text
unresolved name `missing_value`
```

## Example classifications

The repository also contains historical samples and compiler fixtures. Their
classification is explicit so they are not mistaken for supported runnable
documentation.

### Regression fixture

- `ability_core.trb` is included directly by a native handler execution test.
  It is maintained for compiler regression coverage, produces no output, and is
  not the canonical user-facing ability example.
- `ability_core.wasm` is a legacy checked-in build artifact. It is not the
  documented output of the current CLI.

### Compile-only examples

There are currently no maintained compile-only examples. `--target none` is
useful for frontend validation, but successful compilation does not establish
native or Wasm execution support.

### Design-only examples

- `modules_file/` illustrates the planned file-module/package layout. The
  current CLI accepts one source file and does not load sibling modules, so
  these files are not compilable as a package.

### Legacy examples

All remaining `.trb` files in this directory are legacy syntax demonstrations
or milestone artifacts:

- `add.trb`, `basic.trb`, `calc.trb`, `float.trb`,
  `function_visibility.trb`, `functions.trb`, `generics.trb`, `hello.trb`,
  `lambda.trb`, `let-destructuring.trb`, `let_advanced.trb`,
  `let_bindings.trb`, `let_simple.trb`, and `let_with_function.trb`
- `milestone3_test.trb`, `modules_inline.trb`, `modules_use.trb`,
  `operator-functions.trb`, `option.trb`, `pattern_advanced.trb`,
  `pattern_matching.trb`, `performance_test.trb`, `record-patterns.trb`,
  `result.trb`, `simple_closure.trb`, `simple_function.trb`, and
  `simple_test.trb`
- `string_interpolation.trb`, `strings/`, `tuples.trb`,
  `ufcs-qualified.trb`, `ufcs-simple.trb`, `zero-arg-comprehensive.trb`,
  `zero-arg-no-parens.trb`, and `zero-arg-simple.trb`

Some legacy files still compile or run through compatibility helpers; others
fail current parsing, resolution, type checking, or entrypoint rules. They are
retained only as historical material and are not supported command examples.
