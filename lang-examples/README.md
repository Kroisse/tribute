# Tribute language examples

The canonical examples in this directory are checked against the current
single-source-file CLI. They use public APIs and are the best starting point for
running Tribute today. The
[compiler capability matrix](../new-plans/capabilities.md) is authoritative for
the wider frontend and target status.

## Canonical runnable examples

Native examples link against the runtime sysroot. Build it once, and again
after changing `crates/tribute-runtime`:

```bash
cargo xtask runtime
```

### M1 native calculator

[`native_calculator.trb`](native_calculator.trb) is the canonical M1 native
preview. It tokenizes commands with public String/Bytes APIs, uses canonical
`List(String)` patterns for dispatch, formats with `Int::to_string`, and
recursively handles `std::io::read_line` until `quit` or EOF.

```bash
cargo run --locked -- compile lang-examples/native_calculator.trb \
  -o target/native-calculator
./target/native-calculator \
  < tests/fixtures/native_calculator_scripted.stdin \
  > target/native-calculator-scripted.stdout
diff -u \
  tests/expected/native_calculator_scripted.stdout \
  target/native-calculator-scripted.stdout
./target/native-calculator \
  < tests/fixtures/native_calculator_eof.stdin \
  > target/native-calculator-eof.stdout
diff -u \
  tests/expected/native_calculator_eof.stdout \
  target/native-calculator-eof.stdout
```

The scripted session covers multiple operations, a recoverable invalid integer,
and `quit`; the EOF session proves quiet termination after its preceding result.
The product regression also supplies a NUL-leading valid UTF-8 malformed line
and invalid UTF-8, verifying respectively normal command recovery and exactly
one terminal input-failure line. On Ubuntu, CI also executes this identical
scripted session with AddressSanitizer:

```bash
cargo run --locked -- compile --sanitize=address \
  lang-examples/native_calculator.trb \
  -o target/native-calculator-asan
./target/native-calculator-asan \
  < tests/fixtures/native_calculator_scripted.stdin \
  > target/native-calculator-asan.stdout
diff -u \
  tests/expected/native_calculator_scripted.stdout \
  target/native-calculator-asan.stdout
```

### M0 native effects artifact

[`native_effects.trb`](native_effects.trb) handles the public
`abilities::Throw` ability and writes through `std::io`.

```bash
cargo run --locked -- compile lang-examples/native_effects.trb \
  -o target/native-effects-example
./target/native-effects-example
```

Expected standard output:

```text
Recovered: the failure was handled
```

This retained M0 artifact demonstrates handled abilities and native output. The
calculator above is the canonical native M1 command example.

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

Install a compatible Wasmtime release as described in the
[development prerequisites](../README.md#development-prerequisites). Wasm
`read_line` is unsupported. Source-level Wasm `fn` handlers, one-shot `op`
handlers, and Abort handlers that drop their continuation execute in focused
tests, but these examples make no Wasm ability execution claim.

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

Current boundaries relevant to these files:

- List literals, `List::prepend`, and list patterns run natively but have no
  Wasm lowering; general collection APIs are unsupported on all targets.
- Inline modules have native execution evidence, but file-module loading,
  package compilation, and separate Tribute-module linking are unsupported.
- Language-level Wasm execution is limited to the forms the
  [capability matrix](../new-plans/capabilities.md) lists as **wasm-run**:
  String/Bytes output, String equality, `Nat` comparisons, Bytes slicing, and
  the `fn`, one-shot `op`, and Abort handler forms.

### Regression fixture

- `ability_core.trb` is included directly by a native handler execution test.
  It is maintained for compiler regression coverage, produces no output, and is
  not the canonical user-facing ability example.
- `ability_core.wasm` is a legacy checked-in build artifact. It is not the
  documented output of the current CLI.

### Checked examples

Every other single-file example passes the frontend, which
[`tests/integration/lang_examples.rs`](../tests/integration/lang_examples.rs)
checks. They use `std::io` for output and declare `Io` on `main`, and most also
run natively. They demonstrate individual language features and are smaller
than the canonical examples above:

- `add.trb`, `basic.trb`, `calc.trb`, `float.trb`,
  `function_visibility.trb`, `functions.trb`, `generics.trb`, `hello.trb`,
  `lambda.trb`, `let-destructuring.trb`, `let_advanced.trb`,
  `let_bindings.trb`, `let_simple.trb`, and `let_with_function.trb`
- `milestone3_test.trb`, `modules_inline.trb`, `modules_use.trb`,
  `operator-functions.trb`, `option.trb`, `pattern_advanced.trb`,
  `pattern_matching.trb`, `performance_test.trb`, `record-patterns.trb`,
  `result.trb`, `simple_closure.trb`, `simple_function.trb`, and
  `simple_test.trb`
- `tuples.trb`, `ufcs-simple.trb`, `ufcs-qualified.trb`, `field-lenses.trb`,
  `zero-arg-comprehensive.trb`, `zero-arg-no-parens.trb`, and
  `zero-arg-simple.trb`

`--target none` validates the frontend without producing an artifact; a
successful frontend check does not establish native or Wasm execution support.

### Unimplemented-feature examples

These examples use designed features that the compiler does not implement yet,
and the frontend check expects their failure:

- `string_interpolation.trb` and `strings/string_interpolation.trb`: string
  interpolation.

### Design-only examples

- `modules_file/` illustrates the planned file-module/package layout. The
  current CLI accepts one source file and does not load sibling modules, and
  `pkg`, `self`, and `super` paths are not supported yet, so these files are
  not compilable as a package.
