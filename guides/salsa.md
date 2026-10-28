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

Salsa field getters and tracked functions return references by default, so
Tribute does not write `returns(ref)`. It chooses another mode as follows:

- `returns(copy)` for small `Copy` values and Salsa handles.
- `returns(deref)` for a `Vec` or `String` whose callers only borrow, so they
  see `&[T]` or `&str`.
- `returns(as_ref)` for an `Option<T>` or `Result<T, E>` whose callers only
  borrow, so they see `Option<&T>` or `Result<&T, &E>`; `returns(as_deref)`
  when `T` itself derefs, as for `Option<Vec<u8>>` or `Result<String, E>`.
- `returns(clone)` only when every caller needs an owned value, for example a
  module that each caller consumes. A caller that needs ownership of a
  borrowed result clones it itself; compiled binaries and IR dumps are
  returned as `&[u8]` and `&str`, and the CLI writes, links, or prints them
  while the database is alive.

A borrowed result is tied to the `&db` borrow, so a caller must finish with it
before mutating an input to start another revision.

A tracked result cannot hold a reference tied to the database lifetime, such as
`Option<&'db Module<V>>`: the memo outlives the revision that owns the target.
A function that only projects a field of another tracked result, such as
`query::parsed_module` over `parsed_ast`, is therefore a plain function that
borrows through the tracked handle. Making it tracked would store a second copy
of the field.

Salsa uses `PartialEq` to decide whether a recomputed value changed. Ordinary
`'static` values do not need `SalsaValue` at a tracked storage boundary. Values
that carry the database lifetime derive `salsa::SalsaValue`; owned leaf types
also need the trait when nested in containers whose implementation requires it.
The generic AST holds both `UnresolvedName` and database-lifetime references,
so its storage requirement is `V: SalsaValue`, not `V: 'static`. The node types
declare no bound on `V`; the derive bounds each `SalsaValue` impl by the field
types, so a node implements the trait exactly when `V` does, and code that only
walks or rebuilds the tree needs no Salsa bound. Its three recursive child
fields have narrow `salsa_value` proofs: they own their children, and `V` must
itself implement `SalsaValue`. Do not store a reference tied to an old database
revision in a tracked value.

A tracked struct's identity is the hash of its untracked fields. A struct that
a query creates again on every run, such as `ParsedAst` from each reparse,
marks its fields `#[tracked]`. Its identity then does not depend on their
content, so the query keeps returning the same struct, and each field keeps its
previous revision while its value compares equal. A query that reads only an
unchanged field is not rerun. Never leave a field whose `Hash` follows an
allocation, such as a pointer-hashed `Arc`, untracked: every run would then
allocate a new struct, and everything that reads it would rerun.

Derived data borrows what it is derived from instead of copying it into its
own result. Source positions stay in the parse that produced them: type
checking and specialization results hold none, and a stage that needs the
positions of several sources, such as lowering the module merged with package
`std`, looks them up through `SpanMaps`, which borrows each source's
`SpanMap`.

A parse keeps positions in two tracked fields. `item_spans` holds each node's
span relative to the start of the item that owns it (a `fn`, a type, an
ability, a `mod` or a `use` declaration). `item_layout` holds where each item
starts and the span of the root `Module`, which no item owns. Moving items
without changing them changes only `item_layout`; since the root spans the
whole source, so does any edit that changes the source's length. No node's
span covers another item, so an inline `mod` declaration's span is its
header. `SpanMap` borrows both and computes absolute spans, so a query that
reads it still reruns when an item moves.

The parse also hands out each item as a `ParsedItem` tracked struct,
identified by its owner key, with its declarations, the items of an inline
module's body, and its relative spans as tracked fields. A query that reads
one item's fields reruns only when that item changes, while `module` and
`item_spans` are assembled from every item.

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
- Reuse tests assert which queries an edit reruns. The `LoggingDb` in
  `crates/tribute-front/tests/common/execution_log.rs` records each
  `EventKind::WillExecute`, the event Salsa sends before running a query. A
  query whose inputs did not change sends `DidValidateMemoizedValue` instead
  and is not recorded. A query that reruns but returns an equal value keeps
  its previous revision, so assert reuse on a query that depends on it, as
  `tests/query_reuse.rs` does with `function_count` over `func_names`.
- Source-logical frontend tests inspect `tribute_control` operations and
  logical signatures. CPS and backend tests enter the shared/target route.

Concrete incremental and diagnostic examples are maintained in
[`tests/integration/salsa_integration.rs`](
../tests/integration/salsa_integration.rs), and LSP document updates are in
[`src/lsp/server.rs`](../src/lsp/server.rs).

## Analysis Cache Boundary

Arena analyses have their own cache scope and preservation/invalidation
contract. A pass that changes IR must invalidate affected analyses or declare
only analyses it actually preserves. Salsa's source-query invalidation cannot
repair a stale arena analysis. See the analysis and pass contracts in
[`new-plans/ir.md`](../new-plans/ir.md).
