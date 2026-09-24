# TrunkIR System

TrunkIR is Tribute's multi-level dialect IR, inspired by MLIR's dialect concept.

## Core Structures

- **`IrContext`** — Arena-based context holding all operations, blocks,
  regions, and types
- **`OpRef`**, **`ValueRef`**, **`BlockRef`**, **`RegionRef`**, **`TypeRef`**
  — Arena references to IR entities
- **`Symbol`** — Interned identifier (4 bytes, O(1) comparison).
  Qualified paths via `ModulePathExt` trait in `tribute-ir`.

## Dialect Organization

Dialects are split across two crates:

- **trunk-ir** (`crates/trunk-ir/src/dialect/`):
  Language-agnostic dialects (core, func, scf, arith, mem, cf, clif,
  wasm, adt)
- **tribute-ir** (`crates/tribute-ir/src/dialect/`):
  Tribute-specific dialects (tribute_control, ability, effect, closure, list,
  tribute_io, tribute_rt)

Dialect levels (high → low):

- **High-level**: tribute_control, ability, effect, closure, list, tribute_io,
  tribute_rt — Tribute language concepts
- **Mid-level**: func, scf, arith, mem, adt — structured operations
- **Low-level**: cf, wasm, clif — target-specific

## Source-logical Control

The frontend emits `tribute_control` callable/control operations with exact
source signatures and checked operation kinds. Shared `tribute_control_to_cps`
legalization alone constructs continuations and the `func`/`closure` graph.
Target ABI lowering validates that graph before physicalizing CPS results and
selecting closure storage. See [the IR contract](../../new-plans/ir.md) and
[the shared pipeline](../../new-plans/cps-effects.md#shared-middle-end-pipeline).

## `#[dialect]` Macro

Operations and types are defined using the `#[dialect]` attribute macro.
Within trunk-ir: `#[crate::dialect(crate = crate)]`.
From external crates: `#[trunk_ir::dialect]`.

**Annotations**:

- `#[attr(...)]` — Attributes (metadata stored on operation); `name?: Ty`
  marks an optional attribute
- `#[region(...)]` — Regions (nested control flow); `#[region(name?)]` marks
  the last region optional, e.g. the body of an external function
- `#[rest]` — Variadic operands
- `-> result` — Operation produces one result; `-> Option<result>` produces
  zero or one; `#[rest_results] -> results` produces any number
- `struct` definitions — Generate typed type wrappers

**Typed syntax**: an operation may instead declare its entities and type
constraints in the signature, following
[the declarative schema contract](../../new-plans/ir.md#선언적-operation-schema):

```rust
fn cmpi<T: IntegerLike>(
    predicate: Attr<Symbol>,
    lhs: Value<T>,
    rhs: Value<T>,
) -> Value<impl BoolLike> {}

fn call_indirect<S: clif::FuncSig>(
    sig: Attr<S::Type>,
    callee: Value<core::Ptr>,
    args: Values<S::Inputs>,
) -> Values<S::Results> {}
```

- Parameters are `Value<C>`, `Variadic<C>`, `Values<L>`, `Attr<K>`, and
  `Option<Attr<K>>`. Results are `Value<C>` or `Option<Value<C>>` (accessor
  `result`) or `Variadic<C>` / `Values<L>` (accessor `results`). Regions and
  successors keep the `#[region(..)]` / `#[successor(..)]` body form.
- Bounds are Rust types implementing `type_constraint::TypeConstraint`:
  `core` scalar categories, macro-defined type wrappers (projections are their
  declared parameters), and the `func`/`clif`/`wasm` `FuncSig` wrappers
  (`Inputs`/`Results`). Unknown, ambiguous, or wrong-kind projections and
  conflicting exact bounds fail to compile.
- An operation uses either the legacy annotations above or the typed syntax,
  never both. Legacy definitions remain supported and are unconstrained in the
  schema.

Typed operations generate a builder that groups inputs by entity kind instead
of a positional constructor:

```rust
let cmp = arith::Cmpi::operands(lhs, rhs) // or `Op::builder()` without operands
    .predicate(Symbol::new("slt"))        // attributes by name
    .results(i1_ty)                       // result types
    .build(loc, ctx);
```

Result types, regions (`.regions(..)`), and successors (`.successors(..)`) are
each set in one call, in declaration order. Missing required inputs panic in
`build`.

**Operation schema**: every generated operation wrapper exposes
`DialectOp::SCHEMA`, a static `op_schema::OpSchema` registered by operation
name. `OpSchema::of(ctx, op)` looks it up for any operation.
`validate_operation_verifiers` checks each operation's counts, required
attributes, and attribute kinds against its schema before running
operation-specific checks, which may then assume the declared shape. Typed
accessors do not check the schema; for an optional region or result, inspect
the operation before calling the accessor.

## Working with IR

Types are created via module-level constructors and converted with
`.as_type_ref()`:

```rust
let nil_ty = core::nil(ctx).as_type_ref();
let func_ty = func::func_sig(ctx, params, [return_ty]).as_type_ref();
```

Operations use the same pattern:

```rust
let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
```

Matching uses typed wrappers (see [code conventions](conventions.md) for
✅/❌ patterns):

```rust
if let Ok(func) = func::Func::from_op(&ctx, op) { ... }
if func::Call::matches(&ctx, op) { ... }
```
