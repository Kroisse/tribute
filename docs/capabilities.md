# Compiler Capabilities and Target Status

This document records what the current Tribute compiler can demonstrably
parse, type-check, lower, compile, and execute. The design documents under
`design/` describe intended semantics; they are not evidence that an
implementation exists.

Statuses and evidence are updated together. A capability not listed here is
**not-yet-verified**, not implicitly supported. README and example claims must
stay within the evidence recorded here. Ignored tests do not establish
successful compilation or execution.

## Status vocabulary

- **native-run** — Tribute source is compiled, linked as a native executable,
  executed, and its result or output is asserted.
- **wasm-run** — Tribute source or the named target boundary is emitted as
  Wasm, executed by a Wasm runtime, and its result or output is asserted.
- **compile-only** — The relevant frontend/shared pipeline or named target
  emits its artifact successfully, but no runtime execution is asserted.
- **unsupported** — The current compiler explicitly rejects the capability, or
  the required compiler/runtime path is absent.
- **not-yet-verified** — The audit found no focused evidence sufficient for
  one of the other statuses. This is not a support claim.

Compilation is not execution. Producing bytes with the Wasm magic number is
only **compile-only**.

## Language and frontend

Every capability in this section passes the frontend and the shared,
target-independent pipeline, which is **compile-only** evidence by itself. A
target status is no stronger than that target's own evidence.

Test paths below are relative to the repository root. `e2e_native.rs` means
`tests/integration/e2e_native.rs`, `wasm_compilation.rs` means
`tests/integration/wasm_compilation.rs`, and `expr_coverage.rs` means
`crates/tribute-front/tests/expr_coverage.rs`.

### Functions, calls, blocks, and `let` bindings

Native **native-run**, WasmGC **compile-only**.

- Frontend: `crates/tribute-front/tests/block_scope.rs`.
- Native: `test_native_function_call` and `test_native_let_binding` in
  `e2e_native.rs`.
- Wasm emission only: `test_compile_function_with_params` and
  `test_compile_local_variables` in `wasm_compilation.rs`.

### `Nat` literals and ordering comparisons

Native **native-run**, WasmGC **wasm-run**.

- Native: `test_native_arithmetic` in `e2e_native.rs` compiles and executes
  `Nat` literals.
- Both targets: `test_nat_comparisons_and_bytes_slice_clamping` in
  `e2e_native.rs` runs `<`, `<=`, `>`, and `>=` on `Nat` literals natively
  and with Wasmtime and asserts each result.

### `Int` literals

Native **native-run**, WasmGC **compile-only**.

- Native: `test_function_type_parameter` in `tests/integration/e2e_add.rs`
  compiles and executes the `+21` literal.
- Wasm emission only: `test_compile_open_callback_root_main` in
  `wasm_compilation.rs` emits an `Int` callback and call without executing it.

### `Float` literals

Native **native-run**, WasmGC **wasm-run**.

- Frontend: `test_float_literal` in `expr_coverage.rs` covers AST-to-IR
  lowering.
- Native: `test_float_literal_compiles` in `tests/integration/e2e_float.rs`
  executes a literal and asserts its output.
- Both targets: `test_literal_patterns` in `e2e_native.rs` passes `Float`
  literals, including `-0.0`, and matches them as patterns on native and with
  Wasmtime.

### `Bool` literals

Native **native-run**, WasmGC **wasm-run**.

- Frontend: `test_case_bool_literal` in `crates/tribute-front/tests/case_type.rs`.
- Native: `test_native_bool_case` in `e2e_native.rs` executes both literals
  and asserts their results.
- Both targets: `bool_case_runs_on_both_targets` in
  `tests/integration/e2e_nat_arithmetic.rs` matches `True`, `False`, and `Nat`
  comparison results in a `case` natively and with Wasmtime and asserts each
  arm.

### `Nil` literal

Native **native-run**, WasmGC **wasm-run**.

- Native: `test_counter_returns_correct_value` in
  `tests/integration/e2e_ability_handler.rs` executes `resume Nil` and asserts
  its output.
- Wasm emission: `test_compile_builtin_io_entrypoint` in `wasm_compilation.rs`
  emits Wasm containing `Nil`.
- Both targets: `test_literal_patterns` in `e2e_native.rs` passes and matches
  `Nil` on native and with Wasmtime.

### `Rune` literals and type checking

Native **native-run**, WasmGC **wasm-run**.

- Frontend: `test_case_rune_literal_switch` in
  `crates/tribute-front/tests/case_type.rs`. The LSP type index also has Rune
  tests in `src/lsp/type_index.rs`.
- Both targets: `test_literal_patterns` in `e2e_native.rs` passes `?a`, `?b`,
  and `?\n` and matches them as patterns on native and with Wasmtime.

### Struct construction and field access

Native **native-run**, WasmGC **not-yet-verified**.

- Frontend: type and spread coverage in
  `crates/tribute-front/tests/record_field_type.rs`.
- Native: `test_native_struct` in `e2e_native.rs` executes construction and
  access.
- Spread has frontend evidence but no focused target execution evidence.

### Enums, constructors, and variant patterns

Native **native-run**, WasmGC **wasm-run**.

- Frontend: enum and `case` coverage in
  `crates/tribute-front/tests/case_type.rs`.
- Native: `test_native_enum_case`, `test_native_enum_empty_variants`, and
  `test_native_enum_option_like` in `e2e_native.rs`.
- Both targets: `test_specialized_enum_payloads` in `e2e_native.rs`
  constructs and matches specialized generic enum payloads on native and with
  Wasmtime; `test_variants_are_told_apart_by_described_descriptor` in
  `wasm_compilation.rs` runs variant matching with Wasmtime.

### `case` over literal and wildcard patterns

Native **native-run**, WasmGC **wasm-run**.

- Frontend: literal-pattern coverage in
  `crates/tribute-front/tests/case_type.rs`.
- Native: `test_native_case_expression` in `e2e_native.rs`.
- Wasm emission: `test_compile_case_expression` in `wasm_compilation.rs`.
- Both targets: `test_literal_patterns` in `e2e_native.rs` executes `String`,
  `Bytes`, `Float`, `Rune`, `Nil`, tuple, and wildcard patterns on native and
  with Wasmtime.

### Tuple construction and patterns

Native **native-run**, WasmGC **wasm-run**.

- Frontend: pattern coverage, including destructuring a tuple returned from a
  function, in `crates/tribute-front/tests/tuple_pattern.rs`.
- Native: `test_native_tuple_create_and_match` in `e2e_native.rs`.
- Both targets: `test_literal_patterns` in `e2e_native.rs` constructs and
  matches a tuple on native and with Wasmtime.

### Closures, captures, and indirect calls

Native **native-run**, WasmGC **not-yet-verified**.

- Frontend and lowering: `test_lambda_as_argument` and related tests in
  `expr_coverage.rs`.
- Native: `test_native_closure` in `e2e_native.rs` and
  `test_closure_execution_simple` in `tests/integration/e2e_add.rs` execute
  captures and calls.

### Direct recursion

Native **native-run**, WasmGC **not-yet-verified**.

- Native: `test_native_recursion` in `e2e_native.rs` executes Fibonacci;
  `test_calc_eval` in `tests/integration/e2e_add.rs` adds recursive enum
  evaluation.

### UFCS and type-directed method resolution

Native **native-run**, WasmGC **not-yet-verified**.

- Frontend: `crates/tribute-front/tests/ufcs_method_call.rs`.
- Native: chaining and disambiguation in the `test_native_ufcs_*` tests in
  `e2e_native.rs`.

### Generated field setters/modifiers and qualified UFCS (`x.f::set(v)`)

Native **native-run**, WasmGC **not-yet-verified**.

- Native: generated `T::f::set` and `T::f::modify`, receiver-directed path
  selection, chaining, effectful modifier callbacks, and the unresolved,
  ambiguous, and duplicate-definition diagnostics in
  `tests/integration/e2e_field_lenses.rs`.

### Type/effect inference and local `let` generalization

Native **not-yet-verified**, WasmGC **not-yet-verified**.

- Frontend: ability rows and the value restriction in
  `crates/tribute-front/tests/lambda_effect_type.rs`,
  `crates/tribute-front/tests/let_generalization.rs`, and
  `tests/integration/row_var_collision.rs`. These establish typing behavior,
  not a target-specific runtime guarantee.

### Opaque `List` literals, `List::prepend`, and `[]`/exact/prefix-rest patterns

Native **native-run**, WasmGC **unsupported**.

- Frontend: `test_list_literals_use_shared_sequence_ops` and
  `test_list_sequence_patterns_use_shared_observation_ops` in
  `expr_coverage.rs` assert representation-independent `list.*` IR.
- Native: the `test_native_list_*` cases in `e2e_native.rs` execute
  left-to-right exactly-once literals, recursive dynamic `List(String)`
  construction through public `List::prepend`, empty/exact/rest observation,
  persistent tails, and reference-element ownership.
- No Wasm `list.*` lowering exists, so this entry makes no Wasm compilation or
  execution claim.

## Abilities and handlers

Backend dispatch unit tests do not establish source-level handler compilation.
Wasm ability support requires successful emission first, then execution of
that exact handler form with its observable result asserted.

### Ability declarations, operation calls, and ability rows

Native **native-run**, WasmGC **wasm-run**.

- Frontend: `tests/integration/e2e_ability_core.rs` and
  `crates/tribute-front/tests/lambda_effect_type.rs`.
- Native and Wasm execution: the handler entries below.

### Row-polymorphic effectful callbacks

Native **native-run**, WasmGC **wasm-run**.

- Native: `test_ability_row_poly_higher_order_function_slow`,
  `test_ability_row_poly_multiple_abilities`, and
  `test_ability_row_poly_unification_across_call_sites_slow` in
  `tests/integration/e2e_ability_row.rs` execute callbacks with one or more
  abilities and instantiate the row independently at different call sites.
- Both targets: handlers are selected by row position.
  `test_callback_operations_pass_an_internal_handler` runs a callback whose
  operations pass a handler its callee installs, and
  `test_callee_takes_one_handler_in_two_positions` runs a callee that names an
  instance and also takes it through its row tail; both execute on native and
  with Wasmtime and assert the same output in
  `tests/integration/e2e_ability_row_evidence.rs`.
- A callee whose declared row is a union of several row variables is not
  guaranteed a separate handler per tail.

### Tail-resumptive `fn` handlers

Native **native-run**, WasmGC **wasm-run**.

- Native: `test_fn_handler_arm` in `tests/integration/e2e_ability_handler.rs`.
- Wasm: `test_execute_tail_dispatch_ability` in `wasm_compilation.rs` runs a
  `fn` handler with Wasmtime and asserts its output.

### General `op` handlers and one-shot `resume`

Native **native-run**, WasmGC **wasm-run**.

- Native: `test_state_set_then_get` and the other State tests in
  `tests/integration/e2e_ability_handler.rs`.
- Wasm: `test_execute_cps_dispatch_ability` in `wasm_compilation.rs` runs
  `op` arms that resume once with Wasmtime and asserts its output.
- Both targets: an `op` arm that reinstalls its own handler around `resume`,
  as in a state-passing `run_state`, runs in
  `test_nested_handler_same_ability_same_type_shadowing` in
  `tests/integration/e2e_ability_nested.rs`. Handler arms run outside their
  handle: `test_op_arm_operations_reach_the_outer_handler`,
  `test_do_arm_operations_reach_the_outer_handler`, and
  `test_fn_arm_operations_reach_the_outer_handler` in
  `tests/integration/e2e_ability_row_evidence.rs`.

### Abort handlers that drop their continuation

Native **native-run**, WasmGC **wasm-run**.

- Native: the Abort tests in `tests/integration/e2e_ability_handler.rs`.
- Wasm: `test_execute_handler_that_drops_its_continuation` in
  `wasm_compilation.rs` runs an Abort arm that returns without resuming and
  asserts the result of both the aborting and the normal path.

### Throw handlers and other early-return or `Never` forms

Native **native-run**, WasmGC **not-yet-verified**.

- Native: early-return, `Never`, and Throw tests in
  `tests/integration/e2e_ability_handler.rs`.
- No focused Wasm evidence was found.

### Nested and multiple abilities

Native **native-run**, WasmGC **wasm-run**.

- Native: composition, shadowing, and deep nesting in
  `tests/integration/e2e_ability_nested.rs`.
- Both targets: `test_nested_handler_same_ability_same_type_shadowing`,
  `test_nested_handler_deep_four_levels_same_ability`, and
  `test_two_abilities_nested_handlers` in the same file run nested handlers of
  one ability and of two abilities with Wasmtime. Resumption through nested
  handlers, alternating operations of two reinstalling handlers, and a
  `resume` written in a handle nested in an arm run in
  `tests/integration/e2e_ability_row_evidence.rs`;
  `test_nested_resume_restores_the_handler_a_nested_handle_hid` covers a handle
  nested in an arm that handles an instance the arm's row also names.

### Module-qualified ability paths in inline modules

Native **native-run**, WasmGC **not-yet-verified**.

- Native: `test_handler_ability_in_module` in
  `tests/integration/e2e_ability_handler.rs` executes a module-qualified
  operation call and handler arm.
- No focused Wasm evidence was found.

## Data, numerics, and standard I/O

### `String` literals, concatenation, `String::from_bytes`, and output

Native **native-run**, WasmGC **wasm-run**.

- Native: String and dynamic output tests in `e2e_native.rs`.
- Wasm: `test_execute_string_literals_and_dynamic_bytes` in
  `wasm_compilation.rs` compiles Tribute source, runs it with Wasmtime, and
  asserts literal, Unicode, empty, `String::from_bytes`, shared, and rope
  output.

### Byte-wise `String` equality (`==` and `!=`)

Native **native-run**, WasmGC **wasm-run**.

- Frontend: `test_string_equality_operators` in `expr_coverage.rs` verifies
  resolution and lowering.
- Both targets: `test_string_equality` in `e2e_native.rs` executes
  shape-independent leaf-span comparison, mismatched leaf boundaries, empty
  leaves, shared and repeated subtrees, Unicode, length mismatch, and
  early/middle/late byte mismatch natively and with Wasmtime. The structural
  prelude test fixes the direct Leaf/Leaf span path and rejects the former
  equality-only flatten/index loop.

### `Bytes` literals, concatenation, and `String::from_bytes`

Native **native-run**, WasmGC **wasm-run** for the operations named below;
other Wasm `Bytes` APIs are **not-yet-verified**.

- Native: literal, concat, length, indexing, and slicing tests in
  `e2e_native.rs`; `test_native_bytes_get_safe` asserts both the in-range
  `Some` and the out-of-range `None` path.
- Wasm, in `wasm_compilation.rs`: concatenation and output in
  `test_execute_dynamic_bytes_write_boundary`; concatenation plus
  `String::from_bytes` in `test_execute_string_literals_and_dynamic_bytes`,
  which asserts the converted `dynamic bytes` and `shared bytes` output;
  `Bytes::slice_or_panic` in
  `test_execute_bytes_slice_or_panic_shares_the_backing_array` and
  `test_execute_bytes_slice_or_panic_traps_out_of_range`, which assert nested
  and empty slices and the trap on a reversed or past-the-end range.
- Both targets, in `e2e_native.rs`: `Bytes::slice` clamping in
  `test_nat_comparisons_and_bytes_slice_clamping` and unsigned
  `Bytes::get_or_panic` reads in
  `test_bytes_get_or_panic_reads_high_bytes_unsigned`.

### `Nat` addition and multiplication

Native **native-run**, WasmGC **wasm-run**.

- Native: `e2e_native.rs` and `tests/integration/e2e_add.rs`.
- Both targets: `nat_addition_and_multiplication_run_on_both_targets` in
  `tests/integration/e2e_nat_arithmetic.rs` reads its operands from a `Bytes`
  value, so canonicalization does not fold them, and runs nested `+` and `*`
  up to the largest current value `4294967295` natively and with Wasmtime.
- Overflow is unspecified ([numeric-types.md](../design/numeric-types.md))
  and is not exercised.

### Other `Nat` arithmetic, and `Int` arithmetic and comparison

Native **native-run**, WasmGC **not-yet-verified**.

- Native: `e2e_native.rs` and `tests/integration/e2e_add.rs`.
- No focused Wasm evidence covers these operations.

### `Int::parse` and `Int::to_string` decimal text conversion

Native **native-run**, WasmGC **not-yet-verified**.

- Frontend: `canonical_int_text_api_resolves_through_prelude` in
  `crates/tribute-front/tests/int_text.rs` verifies the public signatures and
  the canonical `Int::ParseError` identity.
- Native: `public_int_text_api_covers_decimal_contract_and_boundaries` in
  `tests/integration/e2e_int_text.rs` uses only public source APIs and
  executes signs, zero, invalid syntax, whitespace and data rejection,
  overflow, signed 32-bit boundaries, formatting, and round trips.
- The current representation accepts only `-2147483648` through
  `2147483647`; syntactically valid values outside that range return
  `OutOfRange` until the Fixnum/BigInt representation is implemented.

### `Nat::to_string` decimal text conversion

Native **native-run**, WasmGC **wasm-run**.

- Both targets: `public_nat_to_string_formats_decimal_text` in
  `tests/integration/e2e_nat_text.rs` executes zero, a single digit, multiple
  digits including inner and trailing zeroes, and the largest current value
  `4294967295` natively under AddressSanitizer and with Wasmtime.
- The current representation is unsigned 32-bit.

### `Float` arithmetic, comparison, and NaN behavior

Native **native-run**, WasmGC **not-yet-verified**.

- Native: the suite in `tests/integration/e2e_float.rs`, including
  `test_float_comparison_nan_semantics`.
- No Wasm float execution or focused emission test was found.

### Tuple collection-like grouping

Native **native-run**, WasmGC **not-yet-verified**.

- See the tuple construction and pattern evidence above.

### General collection APIs beyond list literals, patterns, and `List::prepend`

**unsupported** on the frontend and on both targets.

- The current public surface is intentionally limited to literals,
  `List::prepend`, and sequence patterns. No broader collection implementation
  is exported by `lib/std/prelude.trb`.

### `std::io::print` and `print_line`

Native **native-run**, WasmGC **wasm-run**.

- Native: output behavior in `e2e_native.rs`.
- Wasm: the two execution tests in `wasm_compilation.rs`. The Wasm lowering
  implements `tribute_io.write` in `crates/tribute-passes/src/wasm/io.rs`.

### `std::io::read_line`

Native **native-run**, WasmGC **unsupported**.

- Native: line endings, empty and partial input, EOF, invalid UTF-8, and
  system errors in the `test_native_std_io_read_line_*` tests in
  `e2e_native.rs`; lowering in `crates/tribute-passes/src/native/io.rs`.
- Wasm I/O lowering has only a `WritePattern` and rejects residual
  `tribute_io` operations in `crates/tribute-passes/src/wasm/io.rs`.

### Canonical calculator REPL

Native **native-run**, WasmGC **not-yet-verified**.

- Native: `lang-examples/native_calculator.trb` is compiled through the public
  CLI and its exact scripted and EOF stdout fixtures are asserted by
  `tests/canonical_calculator.rs`, which also executes NUL-leading valid UTF-8
  malformed input and invalid UTF-8 and requires a clean exit with exactly one
  input-failure line. The scripted session runs under AddressSanitizer locally
  and in the Ubuntu `canonical-native` CI matrix.
- Its source opens with empty LSP diagnostics and has `Int::parse` hover plus
  imported `Io` completion protocol evidence in `src/lsp/server.rs`.
- No Wasm calculator execution claim is made.

### Summary of Wasm execution claims

The current **wasm-run** language-level claims are String/Bytes output through
`std::io::print_line`, byte-wise String equality, `Bool` literals, `Nat`
addition, multiplication, and ordering comparisons, `Nat::to_string`,
`Bytes::slice_or_panic`, `Bytes::slice` clamping, and the `fn`, `op`
(including handlers that reinstall themselves), continuation-dropping Abort,
row-polymorphic callback, and nested handler forms named above. That evidence
does not establish general WasmGC parity with native.

## Diagnostics, LSP, and compilation boundaries

These statuses are independent of a target runtime.

### Structured frontend diagnostics

**compile-only.** Parsing, name, type, effect, exhaustiveness, handler,
entrypoint, and unknown or missing struct-field snapshots are in
`tests/integration/diagnostic_snapshots.rs`.

### CLI diagnostic rendering

**compile-only.** Target and frontend diagnostics are sorted and rendered by
`src/diagnostics.rs`; the CLI routes each target through this path in
`src/main.rs`.

### LSP requests

**compile-only.** Diagnostics, hover, completion, document symbols, signature
help, definition, references, rename, and code actions have in-process protocol
tests in `src/lsp/server.rs`, and index-level tests in `src/lsp/type_index.rs`,
`src/lsp/definition_index.rs`, and `src/lsp/completion_index.rs`. This is
editor-protocol evidence, not target execution.

### One root source file or package directory per CLI compile invocation

**compile-only.** `tribute compile` accepts one `PathBuf` in `src/cli.rs` and
`src/main.rs`: a source file, compiled without a manifest even inside a
package, or a package directory whose `tribute.toml` it loads. A source file
that is a package's root file gets a note naming the package directory
(`the_cli_compiles_a_packages_root_file_without_its_manifest` in
`tests/integration/manifests.rs`). It reads source files and manifests through
the database's file provider in `src/database.rs`.

### Build input file listing and LSP file watching

**compile-only.** `crates/tribute-front/src/build_inputs.rs` lists the files a
package directory or a manifest-less source file reads, from the paths its
source trees and manifests asked the file provider about, including paths with
no file yet
([implementation.md](../design/implementation.md#패키지-source-트리와-파일-입력)).

- `a_package_lists_the_files_its_build_reads` and
  `editing_a_file_reuses_the_list_unless_its_modules_change` in
  `crates/tribute-front/tests/manifests.rs` check the lists and that an edit
  that changes no package's files reuses the combined list.
- `tribute files [--deps] [--candidates]` prints them
  (`the_cli_lists_the_files_a_build_reads` in
  `tests/integration/manifests.rs`).
- The language server registers `workspace/didChangeWatchedFiles` watchers for
  the files its open documents read, reads a changed file again, and registers
  again when the files change (`the_server_watches_the_files_its_documents_read`
  in `src/lsp/server.rs`); a client without dynamic registration gets no
  watchers ([lsp.md](../design/lsp.md#입력과-무효화)).

### Inline `mod Name { ... }` modules

**native-run.** `test_native_ufcs_chained_multi_arg_user_defined` in
`e2e_native.rs` executes functions from inline `Pair` and `Triple` modules, and
`test_handler_ability_in_module` in `tests/integration/e2e_ability_handler.rs`
executes a module-qualified ability call and handler arm. Frontend AST
construction is in `crates/tribute-front/src/astgen/declarations.rs`.

### File module loading (`mod name` reading `name.trb`)

**native-run.** `crates/tribute-front/src/source_tree.rs` reads each bodyless
`mod` from the file [modules.md](../design/modules.md#모듈-선언) names, in the
root file's directory and a file module's same-named directory, into one
package AST.

- `file_modules_lower_like_inline_modules` checks that a multi-file package
  lowers to the same IR as the package written inline,
  `editing_a_file_module_parses_only_that_file` checks that an edit parses
  only the edited file, and `the_cli_compiles_and_runs_a_file_module_package`
  compiles and runs `lang-examples/modules_file/` natively, all in
  `tests/integration/file_modules.rs`, which also snapshots the missing-file,
  body-and-file, and file-reached-twice errors.
- The language server analyzes a file module's document in every root of its
  package that reads it, and an open document outside a package as a package
  root (`src/analysis_unit.rs`,
  [lsp.md](../design/lsp.md#문서가-속한-패키지)); it publishes only the
  diagnostics located in the document itself.

### Package manifests (`tribute.toml`) and path dependencies

**native-run.** `crates/tribute-front/src/manifest.rs` reads each manifest
through the file provider and builds the package graph by following
`[dependencies]` paths relative to each manifest
([modules.md](../design/modules.md#매니페스트-tributetoml)).

- `the_cli_compiles_a_package_from_any_working_directory` builds and runs a
  package with a path dependency from two working directories, and
  `an_app_lowers_its_dependency_at_its_identity_paths` checks the dependency's
  identity paths in the app's IR, in `tests/integration/manifests.rs`.
- `crates/tribute-front/tests/manifests.rs` snapshots the manifest,
  dependency-entry, cycle, and missing-root errors, and checks that a manifest
  edit reloads only the packages that depend on it.
- A package with only `src/lib.trb` is checked with `--target none` and builds
  no executable. An executable root does not depend on the same package's
  `src/lib.trb`, and compiling a package that has `src/main.trb` does not check
  its `src/lib.trb`. Two packages in one build with the same name but different
  directories are not detected.
- The language server analyzes an open `src/lib.trb` as the library package
  under its manifest's name and `src/main.trb` as the executable root with the
  manifest's dependency table, finding each source's package by its path
  (`src/analysis_unit.rs`). It does not publish manifest diagnostics or
  diagnostics located in dependencies, and its completion lists only `std`'s
  names among dependencies ([lsp.md](../design/lsp.md)).

### Backend separate compilation and linking of Tribute packages

**unsupported.** The stages from specialization through emission run on one
module for the whole build
([linking.md](../design/linking.md#compilation-unit)). Native linking consumes
one object generated from one `SourceCst` in `src/main.rs` and
`src/pipeline.rs`; it does not link independently compiled Tribute packages.

## Canonical examples

The user-facing canonical examples, how to run them, and their expected output
are recorded in [lang-examples/README.md](../lang-examples/README.md). Their
claims must stay within this document:

- `lang-examples/native_calculator.trb` is the canonical **native-run**
  example (see the calculator entry above).
- `lang-examples/wasm_dynamic_output.trb` is the canonical **wasm-run**
  example; Wasmtime execution must assert its expected String/Bytes output.
- `lang-examples/native_effects.trb` is a retained **native-run** example.
- `lang-examples/invalid_unresolved_name.trb` is the canonical frontend
  failure example; its diagnostic must include
  `` unresolved name `missing_value` ``.

## Updating this document

A status promotion requires focused evidence at the claimed boundary:

- Promote **compile-only** to **native-run** only with a linked and executed
  native program.
- Promote **compile-only** to **wasm-run** only with runtime execution and an
  asserted result.
- Replace **unsupported** only after the rejecting boundary or missing path is
  implemented and tested.
- Replace **not-yet-verified** only after a focused audit; implementation
  shape alone is insufficient.

README, examples, release notes, and target documentation must not make
stronger current-support claims than this document.
