# TrunkIR Design

TrunkIR is Tribute's central intermediate representation between typed source
programs and target-specific Wasm or native backends.

## Principles

- SSA-based, using block arguments instead of phi nodes.
- Dialect namespaced: operations are written as `<dialect>.<operation>`.
- Multi-level: high-level, mid-level, and low-level dialects may coexist.
- Structured control flow is preferred until a backend requires CFG lowering.
- Lowering boundaries are declared with `ConversionTarget`, not Rust phase
  types such as `Module<Phase>`.

## Dialect Layers

```text
Infrastructure
  core      module structure and conversion glue

High-level
  tribute   unresolved or source-level frontend constructs
  ability   evidence and handler dispatch semantics
  effect    target-independent effect ABI
  closure   closure construction and decomposition
  adt       structs, variants, arrays, references, literals

Mid-level
  func      functions, calls, returns, function references
  scf       structured control flow and region yields
  arith     constants, arithmetic, comparisons, casts
  mem       low-level memory/data operations

Low-level
  wasm.*    Wasm and WasmGC-oriented operations
  clif.*    Cranelift-oriented operations
```

## Conversion Legality

`ConversionTarget` classifies operations as `Legal`, `Illegal`, or `Unknown`.
An unspecified operation is `Unknown`, not legal.

Partial conversion may leave unknown operations for later passes. Full
conversion boundaries, such as backend-ready native IR, must reject unknown
operations.

## Validation Layers

TrunkIR validation is layered by responsibility. The compiler should use the
smallest layer that can state an invariant precisely, and should not introduce a
separate semantic-contract framework unless these layers leave a concrete gap.

| Layer | Responsibility | Failure timing | Examples |
| ---- | ---- | ---- | ---- |
| Operation verifier | Local invariants of one operation: operand/result counts, required attributes, attribute domains, region shape, and terminator requirements that can be checked without global analysis. | During parsing/building when possible, or during explicit operation validation checkpoints. | `arith.cmpf` accepts only supported predicates; an op with regions requires the expected region count and terminator form. |
| Conversion target | Dialect and type legality at a named lowering boundary. | Immediately after a pass or pass group claims a conversion boundary. Partial conversion rejects explicitly illegal operations; full conversion also rejects unknown operations. | Ability lowering leaves no `ability.perform`; backend-ready native IR contains only `clif.*` plus allowed infrastructure ops. |
| Pass-manager verifier | Whole-IR consistency after transformations, with the offending pass identified. | After each pass registered in a `PassManager` when a verifier hook is installed. | SSA use-chain consistency, value visibility across isolated regions, or other graph-wide invariants. |
| Operation interface | Shared behavior queried generically across dialects. | At the consumer that needs dialect-independent behavior. Interfaces should be introduced only for multiple concrete consumers or one generic transform. | `PureOps` for DCE removability; `IsolatedFromAboveOps` for nested pass-manager anchoring. |

Local operation verifiers must not depend on conversion state or pass ordering.
Conversion targets must not duplicate local semantic checks. Pass-manager
verifiers should remain responsible for graph-wide invariants that require
walking use-def chains, symbol tables, or nested region relationships. Operation
interfaces describe behavior, not validation phases.

## Core Invariants

- A `core.module` owns the top-level region for a compilation unit.
- `core.unrealized_conversion_cast` is temporary conversion glue and must not
  remain at backend-ready boundaries.
- Operation and type names are interned `Symbol`s. Qualified paths are stored as
  `::`-separated symbols.
- Nested regions use normal SSA visibility rules: values defined inside a
  nested region are not visible outside it unless yielded or otherwise modeled
  by the operation.

## High-Level Dialects

`tribute.*` represents source-level constructs that should disappear after
resolution, type checking, TDNR, and AST-to-IR lowering.

`ability.*` represents effect evidence and handler dispatch. Ability operations
are lowered through the effect pipeline; ability-related types may remain until
their target-specific representation is selected.

`effect.*` represents the target-independent ABI between high-level ability
semantics and backend-specific evidence/callable layouts. It carries semantic
inputs such as evidence, ability identity, operation name, payload,
continuation, and handler closures. It must not expose Marker field indices,
handler-table storage layout, closure field positions, or backend function
pointer representation.

`closure.*` represents closure allocation and projection. Closures lower
differently per backend: Wasm uses function references plus GC structures, while
native uses function pointers plus heap environments.

`adt.*` represents target-independent product, sum, array, reference, and
literal operations.

## Mid-Level Dialects

`func.*` represents function definitions, direct calls, indirect calls, function
references, returns, tail calls, and unreachable control flow.

`scf.*` represents structured control flow, including pattern/case regions and
region yields. Loop-like forms may be introduced by optimization passes such as
tail-call lowering.

`arith.*` represents constants, integer and floating arithmetic, comparisons,
bit operations, and numeric conversions.

`mem.*` represents low-level data, load, and store operations for runtime or FFI
support.

## Low-Level Dialects

`wasm.*` is the Wasm backend dialect. It models Wasm control flow, calls,
numeric operations, memory operations, and WasmGC constructs.

`clif.*` is the native backend dialect. It models Cranelift-style functions,
calls, arithmetic, CFG control flow, memory access, stack slots, symbol
addresses, and numeric conversions.

Backend-ready full conversion targets must explicitly list which infrastructure
operations are still allowed next to the backend dialect.

## Pipeline Shape

```text
source
  -> parse / AST
  -> name resolution
  -> type checking
  -> TDNR
  -> AST-to-IR
  -> shared lowering and optimization
  -> Wasm or native lowering
  -> backend-ready full conversion target
  -> emit
```

Important stage invariants:

| Stage | Required invariant |
| ---- | ---- |
| Resolution | Names, constructors, and variable references are resolved |
| Type check | Type variables and effect rows are solved |
| TDNR | Method-style calls are converted to resolved calls |
| AST-to-IR | IR structure and SSA use chains are valid |
| Shared lowering | Source-level and high-level ability dispatch operations are removed at claimed boundaries |
| Effect ABI | `effect.*` operations preserve dispatch semantics without backend layout details |
| Backend lowering | Backend-ready target verification succeeds and no `effect.*` operations remain |

## Type Model

Primitive scalar, pointer, reference, bytes, array, tuple, function, and nil
types are represented in TrunkIR. Library data types such as `Option`, `Result`,
`List`, and `Text` lower through ADT and runtime/library conventions rather than
special IR phases.

## Open Questions

- Final closure environment representation for each backend.
- Reusable conversion targets for effect ABI lowering boundaries.
- Debug/source-map representation.
- WasmGC reactivation strategy for lowering shared tail-call CPS IR into
  closure, table, and evidence representations.
