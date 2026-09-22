# Crate Structure

The Tribute compiler is organized as a Rust Cargo workspace.

## Responsibilities and Dependency Direction

- The frontend owns parsing, name resolution, type checking, monomorphization,
  and source-logical `tribute_control` IR construction. Shared passes own CPS
  legalization; target passes own physical callable ABI and storage.
- Tribute-specific dialects and transformation passes build on the
  language-agnostic IR infrastructure. Passes include shared and target-specific
  transformations.
- The IR infrastructure and code-generation backends remain language-agnostic;
  they must not depend on Tribute-specific compiler layers.
- The CLI and language server compose these layers into compilation pipelines.

## Inspecting the Workspace

Use the root [Cargo.toml](../../../Cargo.toml) for workspace membership and each
member's `Cargo.toml` for its declared dependencies. Inspect the resolved
dependency graph with:

```bash
cargo tree --workspace --edges normal
```

Pipeline structure is documented in the top-of-file comment in
[src/pipeline.rs](../../../src/pipeline.rs).
