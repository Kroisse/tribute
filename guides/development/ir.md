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

- `#[attr(...)]` — Attributes (metadata stored on operation)
- `#[region(...)]` — Regions (nested control flow)
- `#[rest]` — Variadic operands
- `-> result` — Operation produces a result value
- `struct` definitions — Generate typed type wrappers

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
