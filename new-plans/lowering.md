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
| Failure | Structured pass errors propagated by `PassManager` |

## Pass Failures

Expected pass failures return a typed source through the pass `run` result; they
do not panic. `PassManager` attaches the failing pass name as `PassError`, stops
before later passes or nested managers, and propagates `PassResult<T>` to the
compiler pipeline. Verifier failures use the same path.

Instrumentation runs only after a pass succeeds. The verifier then runs on the
successful result; a verifier failure stops the pipeline before further work.

## Operation-Anchored Scheduling

Use nested pass managers for transformations whose contract is scoped to one
operation instance instead of the whole module. In particular, intraprocedural
passes should target `func.func` through `PassManager::nest::<func::Func>()`
when they do not create, erase, or retarget sibling functions and do not need
module-wide symbol or call-site decisions.

Nested pass-manager anchors must be registered as `IsolatedFromAbove`.
This keeps nested pipelines on operations whose regions can be reasoned about
without capturing values from the parent scope. `func.func` and `core.module`
are valid anchors; region-bearing control-flow operations such as `scf.if`
remain part of their enclosing function pipeline unless a documented exception
is added.

Run a nested pipeline coherently for one anchor before advancing to the next
matching operation. Module passes remain responsible for symbol-table changes,
function creation or deletion, cross-function call rewrites, global DCE, and
pipeline-boundary conversion checks.

`PatternApplicator` supports `RewriteScope`-scoped application for these
anchored passes. For example, `CanonicalizeFunc` runs generic canonicalization
under one `func.func` scope, while module-wide cleanup uses the same
`canonicalize` entry point with a module scope.

## Conversion Modes

Use partial conversion when a pass only promises to remove a declared illegal
set from one step. Unknown operations may remain for later passes.

Use full conversion at pipeline boundaries. Full conversion fails on both
illegal and unknown operations, so every allowed operation must be declared
legal by the boundary target.

Shared lowering passes must not call unchecked `apply_partial` when their API or
name claims to complete a lowering stage. Use named `ConversionTarget`s for
pipeline boundaries instead of open-coding legality rules per pass.
Failed boundaries report `rewrite::ConversionError`.

The shared effect pipeline establishes the `ability-lowered` partial boundary
after `LowerHandleDispatch`: residual `ability.*` operations are illegal, while
operations owned by later lowering stages remain unknown and are allowed.

The effect ABI boundary is represented by `effect.*` operations. Shared ability
lowering may produce `effect.extend`, `effect.dispatch_tail`, and
`effect.dispatch_cps`; backend-specific lowering must eliminate them before a
backend-ready full conversion target. Shared passes must not lower these
operations by inspecting concrete Marker fields or backend closure layouts.
Native lowering maps the ABI to runtime evidence calls and native closure
pointers. WasmGC lowering uses the same shared ability IDs and marker field
order, but stores dispatch closures as `anyref` closure structs and lowers
dispatch to evidence lookup, closure unpacking, and `wasm.call_indirect`.

## Legality Precedence

`ConversionTarget` legality is structural, not callback-order based:

1. operation dynamic rule
2. operation static rule
3. dialect dynamic rule
4. dialect static rule
5. unknown-operation dynamic fallback
6. `Unknown`

Dynamic rules return `Legal` or `Illegal` to decide at that tier. Returning
`Defer` continues to the next structural tier. Re-registering the same static
or dynamic key is invalid; operation-level rules may intentionally override
dialect-level rules.

## Recursive Legality

A legal operation may be marked recursively legal. Conversion verification and
pattern application then treat the operation as an opaque legal boundary: the
operation itself is checked, but its nested regions are not inspected by that
conversion target.

Recursive legality applies in both partial and full conversion modes. Without
the recursive marker, descendants of a legal operation are still visited and
checked normally. Static recursive legality requires the same operation to be
registered as statically legal. Dynamic recursive legality is operation-specific
and applies only when that operation's dynamic rule returns `Legal`.

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
arith_to_wasm
scf_to_wasm
func_to_wasm
wasm_func_signature_conversion
tribute_rt_to_wasm
adt_to_wasm
evidence_to_wasm
const_to_wasm
intrinsic_to_wasm
wasm_lowerer
assign_gc_type_indices
```

Backend-ready Wasm lowering must eliminate residual `effect.*` operations.
The current `wasm-backend-ready` partial conversion boundary rejects residual
`ability.*` and `effect.*` operations while still allowing later-stage
infrastructure such as unresolved casts. `evidence_to_wasm` generates the
evidence lookup/extend helpers and lowers effect dispatch to closure unpacking
plus `wasm.call_indirect`.

The lower-level `trunk-ir-wasm-backend` pass group handles target-independent
dialect conversion:

```text
arith_to_wasm
scf_to_wasm
func_to_wasm
adt_to_wasm
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
effect ABI to native runtime lowering
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
