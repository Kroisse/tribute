# Compiler Capability and Target Status

This document is the source of truth for what the current Tribute compiler can
demonstrably parse, type-check, lower, compile, and execute. The language design
documents describe intended semantics; they are not evidence that an
implementation exists.

The matrix was audited on 2026-07-23 at base commit
`f37d8bffa7bdfdb7297ccbba4999d9ecf2c0761a`. A later change must update the
status and evidence together. A capability not listed here is
**not-yet-verified**, not implicitly supported.

## Status Vocabulary

| Status | Required evidence |
| --- | --- |
| **native-run** | Tribute source is compiled, linked as a native executable, executed, and its result or output is asserted. |
| **wasm-run** | Tribute source or the named target boundary is emitted as Wasm, executed by a Wasm runtime, and its result or output is asserted. |
| **compile-only** | The relevant frontend/shared pipeline or named target emits its artifact successfully, but no runtime execution is asserted. |
| **unsupported** | The current compiler explicitly rejects the capability, or the required compiler/runtime path is absent. |
| **not-yet-verified** | The audit found no focused evidence sufficient for one of the other statuses. This is not a support claim. |

Compilation is not execution. In particular, producing bytes with the Wasm
magic number is only **compile-only**.

README and example alignment is owned by issue #797. In checkouts where that
alignment is not yet present, this matrix is authoritative over older README
or example claims.

## Language and Frontend Matrix

The shared column covers parsing through the target-independent pipeline. A
target column is intentionally no stronger than its own evidence.

| Capability | Shared/frontend | Native | WasmGC | Evidence and boundary |
| --- | --- | --- | --- | --- |
| Functions, calls, blocks, and `let` bindings | **compile-only** | **native-run** | **compile-only** | Frontend coverage is in [`block_scope.rs`](../crates/tribute-front/tests/block_scope.rs). Native execution is covered by `test_native_function_call` and `test_native_let_binding` in [`e2e_native.rs`](../tests/e2e_native.rs). Wasm emission is covered by `test_compile_function_with_params` and `test_compile_local_variables` in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). |
| `Nat` literals | **compile-only** | **native-run** | **compile-only** | `test_native_arithmetic` compiles and executes `Nat` literals in [`e2e_native.rs`](../tests/e2e_native.rs). `test_compile_simple_literal` emits Wasm for a `Nat` literal but does not execute it in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). |
| `Int` literals | **compile-only** | **native-run** | **compile-only** | `test_function_type_parameter` compiles and executes the `+21` literal in [`e2e_add.rs`](../tests/e2e_add.rs). `test_compile_tail_dispatch_ability` emits Wasm containing `Int` literals but does not execute it in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). |
| `Float` literals | **compile-only** | **native-run** | **not-yet-verified** | `test_float_literal` covers AST-to-IR lowering in [`expr_coverage.rs`](../crates/tribute-front/tests/expr_coverage.rs). `test_float_literal_compiles` executes a literal and asserts its native output in [`e2e_float.rs`](../tests/e2e_float.rs). No focused Wasm evidence was found. |
| `Bool` literals | **compile-only** | **native-run** | **not-yet-verified** | `test_bool_literal_true` and `test_bool_literal_false` cover AST-to-IR lowering in [`expr_coverage.rs`](../crates/tribute-front/tests/expr_coverage.rs). `test_native_bool_case` executes both literals and asserts their native results in [`e2e_native.rs`](../tests/e2e_native.rs). No focused Wasm evidence was found. |
| `Nil` literal | **compile-only** | **native-run** | **compile-only** | `test_nil_literal` covers AST-to-IR lowering in [`expr_coverage.rs`](../crates/tribute-front/tests/expr_coverage.rs). `test_counter_returns_correct_value` executes `resume Nil` and asserts native output in [`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs). `test_compile_builtin_io_entrypoint` emits Wasm containing `Nil` but does not execute it in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). |
| `Rune` literals and type checking | **compile-only** | **not-yet-verified** | **not-yet-verified** | `test_rune_literal` lowers a Rune literal in [`expr_coverage.rs`](../crates/tribute-front/tests/expr_coverage.rs). The LSP type index also has Rune tests in [`type_index.rs`](../src/lsp/type_index.rs). No target execution test was found. |
| Struct construction and field access | **compile-only** | **native-run** | **not-yet-verified** | Type and spread coverage is in [`record_field_type.rs`](../crates/tribute-front/tests/record_field_type.rs). Native construction/access executes in `test_native_struct` in [`e2e_native.rs`](../tests/e2e_native.rs). Spread has frontend evidence, but no focused target execution evidence. |
| Enums, constructors, and variant patterns | **compile-only** | **native-run** | **not-yet-verified** | Frontend enum/case coverage is in [`case_type.rs`](../crates/tribute-front/tests/case_type.rs). Native enum patterns execute in `test_native_enum_case`, `test_native_enum_empty_variants`, and `test_native_enum_option_like` in [`e2e_native.rs`](../tests/e2e_native.rs). No focused Wasm enum test was found. |
| `case` over literal and wildcard patterns | **compile-only** | **native-run** | **compile-only** | Frontend literal-pattern coverage is in [`case_type.rs`](../crates/tribute-front/tests/case_type.rs). `test_native_case_expression` executes natively in [`e2e_native.rs`](../tests/e2e_native.rs). Wasm `case` emission is covered by `test_compile_case_expression` in [`wasm_compilation.rs`](../tests/wasm_compilation.rs); it is not executed. |
| Tuple construction and patterns | **compile-only** | **native-run** | **not-yet-verified** | Frontend pattern coverage, including destructuring a tuple returned from a function, is in [`tuple_pattern.rs`](../crates/tribute-front/tests/tuple_pattern.rs). `test_native_tuple_create_and_match` executes the native path in [`e2e_native.rs`](../tests/e2e_native.rs). |
| Closures, captures, and indirect calls | **compile-only** | **native-run** | **not-yet-verified** | Frontend/lowering coverage is in `test_lambda_as_argument` and related tests in [`expr_coverage.rs`](../crates/tribute-front/tests/expr_coverage.rs). `test_native_closure` and `test_closure_execution_simple` execute captures and calls in [`e2e_native.rs`](../tests/e2e_native.rs) and [`e2e_add.rs`](../tests/e2e_add.rs). |
| Direct recursion | **compile-only** | **native-run** | **not-yet-verified** | `test_native_recursion` executes Fibonacci in [`e2e_native.rs`](../tests/e2e_native.rs). `test_calc_eval` adds recursive enum evaluation coverage in [`e2e_add.rs`](../tests/e2e_add.rs). |
| UFCS and type-directed method resolution | **compile-only** | **native-run** | **not-yet-verified** | Resolution tests are in [`ufcs_method_call.rs`](../crates/tribute-front/tests/ufcs_method_call.rs). Native chaining and disambiguation execute in the `test_native_ufcs_*` tests in [`e2e_native.rs`](../tests/e2e_native.rs). |
| Type/effect inference and local let generalization | **compile-only** | **not-yet-verified** | **not-yet-verified** | Effect rows and value restriction are tested in [`lambda_effect_type.rs`](../crates/tribute-front/tests/lambda_effect_type.rs), [`let_generalization.rs`](../crates/tribute-front/tests/let_generalization.rs), and [`effect_var_collision.rs`](../tests/effect_var_collision.rs). These tests establish typing behavior, not a target-specific runtime guarantee. |
| List expressions such as `[1, 2]` | **unsupported** | **unsupported** | **unsupported** | The parser and type checker contain list shapes, but AST-to-IR explicitly emits unsupported for `ExprKind::List` in [`ast_to_ir/lower/expr.rs`](../crates/tribute-front/src/ast_to_ir/lower/expr.rs). The current prelude also has no `List` definition in [`prelude.trb`](../lib/std/prelude.trb). |

## Abilities and Handlers

| Capability | Shared/frontend | Native | WasmGC | Evidence and boundary |
| --- | --- | --- | --- | --- |
| Ability declarations, operation calls, and effect rows | **compile-only** | **native-run** | **compile-only** | Frontend checks are in [`e2e_ability_core.rs`](../tests/e2e_ability_core.rs) and [`lambda_effect_type.rs`](../crates/tribute-front/tests/lambda_effect_type.rs). Native execution is covered by the handler suites. Wasm only emits artifacts in `test_compile_tail_dispatch_ability` and `test_compile_cps_dispatch_ability` in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). |
| Row-polymorphic effectful callbacks | **compile-only** | **native-run** | **not-yet-verified** | `test_effect_row_poly_higher_order_function`, `test_effect_row_poly_multiple_abilities`, and `test_effect_row_poly_unification_across_call_sites` execute native callbacks with one or more abilities and independently instantiate the row at different call sites in [`e2e_ability_effect_row.rs`](../tests/e2e_ability_effect_row.rs). No focused Wasm evidence was found. |
| Tail-resumptive `fn` handlers | **compile-only** | **native-run** | **compile-only** | `test_fn_handler_arm` executes natively in [`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs). The Wasm `fn` test only calls `expect_wasm_compilation_success`. |
| General `op` handlers and one-shot `resume` | **compile-only** | **native-run** | **compile-only** | `test_state_set_then_get` and the other State tests execute natively in [`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs). The Wasm CPS test only asserts emission. |
| Dropped continuations and abort/throw handlers | **compile-only** | **native-run** | **not-yet-verified** | Native early-return, `Never`, Abort, and Throw execution tests are in [`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs). |
| Nested and multiple abilities | **compile-only** | **native-run** | **not-yet-verified** | Native composition, shadowing, and deep nesting execute in [`e2e_ability_nested.rs`](../tests/e2e_ability_nested.rs). No corresponding Wasm execution test was found. |
| Module-qualified ability paths in inline modules | **compile-only** | **native-run** | **not-yet-verified** | `test_handler_ability_in_module` executes a module-qualified operation call and handler arm in [`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs). File-module loading remains unsupported, and no focused Wasm evidence was found. |

Wasm ability support must not be described as **wasm-run** until an emitted
program using that exact handler form is executed and its observable result is
asserted. The current two Wasm ability tests are **compile-only**.

## Data, Numerics, and Standard I/O

| Capability | Shared/frontend | Native | WasmGC | Evidence and boundary |
| --- | --- | --- | --- | --- |
| `String` literals, concatenation, `String::from_bytes`, and output | **compile-only** | **native-run** | **wasm-run** | Native String and dynamic output tests are in [`e2e_native.rs`](../tests/e2e_native.rs). `test_execute_string_literals_and_dynamic_bytes` compiles Tribute source, runs it with Wasmtime, and asserts literal, Unicode, empty, `String::from_bytes`, shared, and rope output in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). |
| `Bytes` literals, concatenation, and `String::from_bytes` | **compile-only** | **native-run** | **wasm-run** | Native literal, concat, length, indexing, and slicing tests are in [`e2e_native.rs`](../tests/e2e_native.rs); `test_native_bytes_get_safe` asserts both the in-range `Some` and out-of-range `None` paths. Wasm execution is limited to concatenation/output in `test_execute_dynamic_bytes_write_boundary` and to concatenation plus `String::from_bytes` in `test_execute_string_literals_and_dynamic_bytes`, which asserts the converted `dynamic bytes` and `shared bytes` output. Other Wasm `Bytes` APIs are **not-yet-verified**. |
| `Nat` and `Int` arithmetic/comparison | **compile-only** | **native-run** | **compile-only** | Native execution is in [`e2e_native.rs`](../tests/e2e_native.rs) and [`e2e_add.rs`](../tests/e2e_add.rs). `test_compile_arithmetic_expr` emits Wasm but does not run it. |
| `Float` arithmetic, comparison, and NaN behavior | **compile-only** | **native-run** | **not-yet-verified** | The native suite, including `test_float_comparison_nan_semantics`, is in [`e2e_float.rs`](../tests/e2e_float.rs). No Wasm float execution or focused emission test was found. |
| Tuple collection-like grouping | **compile-only** | **native-run** | **not-yet-verified** | See tuple construction and pattern evidence above. |
| General collections (`List`, map/set APIs) | **unsupported** | **unsupported** | **unsupported** | List construction cannot cross AST-to-IR, and no general collection implementation is exported by [`prelude.trb`](../lib/std/prelude.trb). |
| `std::io::print` and `print_line` | **compile-only** | **native-run** | **wasm-run** | Native output behavior is tested in [`e2e_native.rs`](../tests/e2e_native.rs). Wasm output execution is covered by the two execution tests in [`wasm_compilation.rs`](../tests/wasm_compilation.rs). The Wasm lowering implements `tribute_io.write` in [`wasm/io.rs`](../crates/tribute-passes/src/wasm/io.rs). |
| `std::io::read_line` | **compile-only** | **native-run** | **unsupported** | Native line endings, empty/partial input, EOF, invalid UTF-8, and system errors execute in the `test_native_std_io_read_line_*` tests in [`e2e_native.rs`](../tests/e2e_native.rs). Native lowering is in [`native/io.rs`](../crates/tribute-passes/src/native/io.rs). Wasm I/O lowering has only a `WritePattern` and rejects residual `tribute_io` operations in [`wasm/io.rs`](../crates/tribute-passes/src/wasm/io.rs). |

The only current **wasm-run** language-level claim is String/Bytes output
through `std::io::print_line`. That evidence does not establish general
WasmGC parity with native.

## Diagnostics, LSP, and Compilation Boundaries

These compiler/tooling statuses are independent of a target runtime.

| Capability | Status | Evidence and boundary |
| --- | --- | --- |
| Structured frontend diagnostics | **compile-only** | Parsing, name, type, effect, exhaustiveness, handler, entrypoint, and unknown/missing struct-field snapshots are in [`diagnostic_snapshots.rs`](../tests/diagnostic_snapshots.rs). |
| CLI diagnostic rendering | **compile-only** | Target and frontend diagnostics are sorted and rendered by [`diagnostics.rs`](../src/diagnostics.rs); the CLI routes each target through this path in [`main.rs`](../src/main.rs). |
| LSP diagnostics, hover, completion, document symbols, signature help, definition, references, rename, and code actions | **compile-only** | In-process protocol tests for these requests are in [`lsp/server.rs`](../src/lsp/server.rs). Index-level tests are in [`type_index.rs`](../src/lsp/type_index.rs), [`definition_index.rs`](../src/lsp/definition_index.rs), and [`completion_index.rs`](../src/lsp/completion_index.rs). This is editor-protocol evidence, not target execution. |
| One source file per CLI compile invocation | **compile-only** | `tribute compile` accepts exactly one `PathBuf`, reads only that file, and creates one `SourceCst` in [`cli.rs`](../src/cli.rs) and [`main.rs`](../src/main.rs). |
| Inline `mod Name { ... }` modules | **native-run** | `test_native_ufcs_chained_multi_arg_user_defined` executes functions from inline `Pair` and `Triple` modules in [`e2e_native.rs`](../tests/e2e_native.rs), and `test_handler_ability_in_module` executes a module-qualified ability call and handler arm in [`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs). Frontend AST construction is in [`astgen/declarations.rs`](../crates/tribute-front/src/astgen/declarations.rs). |
| File module loading (`mod name` resolving another `.trb`) | **unsupported** | The AST records external modules as `body: None` in [`astgen/declarations.rs`](../crates/tribute-front/src/astgen/declarations.rs), but the single-file CLI has no loader or source graph. Files under `lang-examples/modules_file/` are examples, not passing compilation evidence. |
| Package/project compilation and manifests | **unsupported** | The CLI accepts a source file rather than a package root or manifest in [`cli.rs`](../src/cli.rs). No package graph or manifest loader is present in the active pipeline. |
| Separate compilation/linking of Tribute modules | **unsupported** | Native linking consumes one object generated from one `SourceCst` in [`main.rs`](../src/main.rs) and [`pipeline.rs`](../src/pipeline.rs); it does not link independently compiled Tribute modules. |

## Canonical M0 Artifacts and Runtime Set

### User-Facing Canonical Artifacts

Issue #797 owns these user-facing artifacts and their documentation. Their
paths are written as code because they may be absent until that separate change
is integrated:

1. `lang-examples/native_effects.trb` is the canonical **native-run** example.
   Its expected stdout is `Recovered: the failure was handled`.
2. `lang-examples/wasm_dynamic_output.trb` is the canonical **wasm-run**
   example. Wasmtime execution must assert its expected String/Bytes output.
3. `lang-examples/invalid_unresolved_name.trb` is the canonical frontend
   failure example. Its diagnostic must include
   `` unresolved name `missing_value` ``.

`lang-examples/README.md` is the companion catalog that records how to run
these artifacts and their expected results.

### Internal Regression Selection

The user-facing artifacts above are the canonical examples. Separately, M0 CI
should keep this broader internal regression selection to cross important
compiler boundaries without treating every compile test as a runtime test:

1. Native ADT/patterns: `test_native_enum_case`.
2. Native closure capture: `test_native_closure`.
3. Native recursion: `test_native_recursion`.
4. Native UFCS: `test_native_ufcs_chained_multi_arg_user_defined`.
5. Native `fn` and `op` handlers: `test_fn_handler_arm` and
   `test_state_set_then_get`.
6. Native standard input: `test_native_std_io_read_line_contract`.
7. Native numeric semantics: `test_float_comparison_nan_semantics`.
8. WasmGC dynamic output boundary:
   `test_execute_dynamic_bytes_write_boundary`.
9. WasmGC source-level String/Bytes output:
   `test_execute_string_literals_and_dynamic_bytes`.

All names above refer to [`e2e_native.rs`](../tests/e2e_native.rs),
[`e2e_ability_handler.rs`](../tests/e2e_ability_handler.rs),
[`e2e_float.rs`](../tests/e2e_float.rs), or
[`wasm_compilation.rs`](../tests/wasm_compilation.rs).

## Audit Commands

The 2026-07-23 audit ran:

```shell
cargo nextest run --workspace -E 'test(test_native_enum_case) | test(test_native_closure) | test(test_native_recursion) | test(test_native_ufcs_chained_multi_arg_user_defined) | test(test_native_std_io_read_line_contract) | test(test_native_bytes_literal_concat) | test(test_float_comparison_nan_semantics) | test(test_state_set_then_get) | test(test_fn_handler_arm) | test(test_execute_dynamic_bytes_write_boundary) | test(test_execute_string_literals_and_dynamic_bytes) | test(diag_non_exhaustive_case) | test(test_hover_via_message) | test(test_completion_via_message)'
```

Result: 14 passed, 0 failed, 1618 skipped. The selection covered native ADTs,
closures, recursion, UFCS, String/Bytes, Float, both handler forms,
`read_line`, both Wasm execution tests, a diagnostic snapshot, and LSP hover
and completion.

```shell
cargo nextest run -p tribute --test wasm_compilation -E 'test(test_compile_tail_dispatch_ability) | test(test_compile_cps_dispatch_ability) | test(test_compile_arithmetic_expr)'
```

Result: 3 passed, 0 failed, 9 skipped. These tests verify Wasm emission only.

```shell
cargo nextest run -p tribute-front --test expr_coverage -E 'test(test_rune_literal) | test(test_string_literal) | test(test_bytes_literal)'
```

Result: 3 passed, 0 failed, 9 skipped. These tests verify the shared frontend
pipeline only.

```shell
cargo nextest run --workspace
```

Result: 1631 passed, 0 failed, 1 skipped.

```shell
npx markdownlint-cli2 "**/*.md"
```

Result: 39 Markdown files linted with 0 errors.

## Skipped Runtime Checks

- Wasm ability execution was not run because the repository's focused `fn` and
  `op` tests stop after artifact emission. This documentation-only issue does
  not own adding runtime tests, so both forms remain **compile-only**.
- Native and Wasm Rune execution were not run because no focused target
  execution fixture or output assertion exists. Rune remains
  **not-yet-verified** on both targets.
- Wasm enum/ADT, tuple, closure, recursion, UFCS, Float, and non-output `Bytes`
  runtime checks were not run because the current Wasm suite has no execution
  assertion for those features. They remain **not-yet-verified**.
- Wasm `read_line`, file-module loading, and package compilation were not
  skipped checks: their required implementation paths are absent, so they are
  **unsupported**.

The full workspace run skips one repository test:

- `test_handler_transforms_result`: TDNR reports
  `` unresolved method `+` for this receiver type `` in the `do` handler arm.
  This remains the open subset of
  [#617](https://github.com/Kroisse/tribute/issues/617).

The issue #802 re-audit returned nine previously ignored tests to the default
suite: three row-polymorphic callback tests, module-qualified ability handling,
sequential Throw operations, tuple destructuring from a function result, the
two struct-field diagnostics, and safe `Bytes::get` `Some`/`None` behavior.

## Updating This Matrix

A status promotion requires focused evidence at the claimed boundary:

- Promote **compile-only** to **native-run** only with a linked and executed
  native program.
- Promote **compile-only** to **wasm-run** only with runtime execution and an
  asserted result.
- Replace **unsupported** only after the rejecting boundary or missing path is
  implemented and tested.
- Replace **not-yet-verified** only after a focused audit; implementation shape
  alone is insufficient.

README, examples, release notes, and target documentation must not make
stronger current-support claims than this matrix.
