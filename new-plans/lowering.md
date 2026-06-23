# Lowering Pass Architecture

TrunkIR lowering is dialect-based and incremental. Passes should lower one
semantic layer at a time, keep mixed-dialect IR valid between steps, and verify
the boundaries they claim to establish.

## Design Decisions

| Topic | Decision |
| ---- | ---- |
| Pass shape | Dialect-specific passes, not one monolithic lowerer |
| Rewriting | `PatternApplicator` fixpoint rewrites |
| Conversion | One configured target controls rewriting and verification |
| Type adaptation | Explicit opt-in, independent of conversion legality |
| State | Precomputed analyses or immutable plans, not mutable pass state |
| Boundaries | `ConversionTarget` verification, not `Module<Phase>` types |

## Conversion Modes

Use partial conversion when a pass only promises to remove a declared illegal
set from one step. Unknown operations may remain for later passes.

Use full conversion at pipeline boundaries. Full conversion fails on both
illegal and unknown operations, so every allowed operation must be declared
legal by the boundary target.

Shared lowering passes must not call unchecked `apply_partial` when their API or
name claims to complete a lowering stage. Use named `ConversionTarget`s for
pipeline boundaries instead of open-coding legality rules per pass.

## Pattern Rewriting

Patterns match one operation family and mutate through `PatternRewriter`.
`PatternApplicator` walks nested regions and applies patterns to a fixpoint. Its
conversion target controls legality-aware rewriting and final verification.
Type adaptation is enabled separately when a pass needs it.

Rules:

- Keep patterns stateless except for read-only analysis data.
- Prefer focused patterns over large manual tree walks.
- Use explicit partial or full conversion entry points when legality matters.
- Keep exact API behavior in Rustdoc near `ConversionTarget`,
  `ConversionMode`, `PatternApplicator`, and `PatternRewriter`.

## Wasm Lowering

The Wasm pipeline lowers target-independent dialects into `wasm.*` while using
analysis plans for type indices, function indices, data segments, imports, and
memory layout.

Typical pass order:

```text
adt_to_wasm
scf_to_wasm
func_to_wasm
arith_to_wasm
intrinsic_to_wasm
const_to_wasm
```

Expected boundary: backend-ready Wasm IR contains `wasm.*` plus explicitly
allowed infrastructure operations.

## Native Lowering

The native pipeline lowers Tribute-specific runtime operations and
target-independent dialects into `clif.*`, then validates the native backend
boundary before emission.

Typical pass groups:

```text
effect/ability lowering
native runtime lowering
arith_to_clif
scf_to_clif
adt_to_clif
func_to_clif
const/intrinsic lowering
```

Expected boundary: backend-ready native IR contains `clif.*` plus explicitly
allowed infrastructure operations.

## Analyses And Plans

Module or function analyses should be pure computations over IR. Passes may use
analysis results, but should not accumulate hidden mutable state that affects
rewrite behavior.

Planner-style data, such as data segments, WASI imports, memory layout, or
native layout metadata, should be represented as immutable plans that can be
computed before the pass that consumes them.

## Open Questions

- Which effect/ability boundary targets should be public reusable APIs?
- Which Wasm backend boundary should be restored when that backend is revisited?
- Whether an analysis-only conversion mode is useful enough to add.

## References

- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
