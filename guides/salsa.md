# Salsa Integration Guide

Salsa caches source-dependent computations. Tribute's frontend and artifact
queries use Salsa; shared and target IR passes operate in a mutable arena.

## Computation Boundaries

```text
SourceCst (URI + Rope + tree-sitter CST)
    → tracked parsing, name resolution, typechecking and TDNR
    → prepared typed AST and semantic metadata
    → fresh arena IrContext + source-logical Module
    → shared CPS and target passes
    → tracked diagnostic/artifact result
```

`SourceCst` is defined in
[`tribute-front/src/source_file.rs`](../crates/tribute-front/src/source_file.rs).
[`src/database.rs`](../src/database.rs) owns the CLI/LSP document database.
[`src/pipeline.rs`](../src/pipeline.rs) composes the queries and arena passes.

Arena `OpRef`, `ValueRef` and `TypeRef` belong to their `IrContext`; they are not
Salsa tracked values. A tracked artifact query may run a fresh compilation
session internally. Individual rewrite mutations are handled by pass and
analysis infrastructure, not Salsa dependency tracking.

## Salsa 0.28 Value and Return Contracts

Salsa field getters and tracked functions return references by default. Tribute
explicitly uses `#[returns(copy)]` for small values and Salsa handles,
`#[returns(clone)]` for owned data, and `#[returns(ref)]` or
`returns(deref)` where callers already borrow data. This keeps existing
ownership boundaries stable, especially when a caller retains a query result
before mutating an input to start another revision.

Salsa uses `PartialEq` to decide whether a recomputed value changed. Values
that carry the database lifetime and are stored in tracked results derive
`salsa::SalsaValue`; ordinary owned values still need `SalsaValue` when nested
inside containers whose implementation requires it. The generic AST's three
recursive child fields have narrow `salsa_value` proofs: they own their children,
and the phase value `V` must itself implement `SalsaValue`. Do not store a
reference tied to an old database revision in a tracked value.

## Creating a Source Input

Use the existing constructor when a test needs parsed source:

```rust
use tribute_front::SourceCst;

let source = SourceCst::from_source_str(
    db,
    "example.trb",
    "fn answer() -> Int { 42 }",
);
```

For real documents, use `TributeDatabaseImpl` and its input/document lifecycle
methods. Do not invent a second database or parsing path for a pass test.

## Incremental Updates

Source text and its CST must describe the same revision. The LSP updates both
`SourceCst::text` and `SourceCst::tree` after tree-sitter parsing. Changing only
the text can leave a stale syntax tree even though Salsa observes the text
setter. Incremental tree-sitter parsing must also apply the corresponding
`InputEdit` before reusing an old tree.

Salsa automatically records dependencies read by tracked queries. Inputs must
be passed or accessed through those queries for invalidation to work. External
file reads, environment changes and arbitrary arena mutations do not become
tracked inputs by themselves.

## Diagnostics

Use `tribute_core::diagnostic::Diagnostic` and its `CompilationPhase` for
source diagnostics. `Diagnostic::new` accepts a message, span, severity and
phase; the builder also supports secondary labels and notes.

`Diagnostic::accumulate(db)` requires an active `#[salsa::tracked]` query.
Attaching a database alone does not create a query context. Accumulate in the
existing frontend query path, then retrieve diagnostics through that query or
`compile_with_diagnostics`.

```rust
use tribute_core::diagnostic::Diagnostic;

let diagnostics: Vec<Diagnostic> =
    tribute::pipeline::parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect();
```

Pure IR passes return pass/conversion errors. The pipeline attaches source
context and reports the appropriate compilation phase at the query boundary.
Avoid continuing into lowering when frontend diagnostics already report an
error.

## Tests

Use the `tribute-testing` skill for command and snapshot conventions. The
repository's `#[salsa_test]` macro supplies and attaches a fresh database:

```rust
use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::SourceCst;

#[salsa_test]
fn valid_source_has_no_diagnostics(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "empty.trb", "fn main() {}");
    let result = compile_with_diagnostics(db, source);
    assert!(result.diagnostics.is_empty());
}
```

- Pure logic and arena transformations can use ordinary unit tests.
- Diagnostic tests must invoke a tracked frontend or pipeline query.
- Incremental tests update the existing source input and assert a meaningful
  changed result; simply calling a query twice does not prove invalidation.
- Source-logical frontend tests inspect `tribute_control` operations and
  logical signatures. CPS and backend tests enter the shared/target route.

Concrete incremental and diagnostic examples are maintained in
[`tests/salsa_integration.rs`](../tests/salsa_integration.rs), and LSP document
updates are in [`src/lsp/server.rs`](../src/lsp/server.rs).

## Analysis Cache Boundary

Arena analyses have their own cache scope and preservation/invalidation
contract. A pass that changes IR must invalidate affected analyses or declare
only analyses it actually preserves. Salsa's source-query invalidation cannot
repair a stale arena analysis. See the analysis and pass contracts in
[`new-plans/ir.md`](../new-plans/ir.md).
