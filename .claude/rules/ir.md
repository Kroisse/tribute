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
  Language-agnostic dialects (core, func, cont, scf, arith, mem, cf, clif,
  trampoline, wasm, adt)
- **tribute-ir** (`crates/tribute-ir/src/dialect/`):
  Tribute-specific dialects (ability, closure, tribute_rt)

Dialect levels (high → low):

- **High-level**: ability, closure, tribute_rt — Tribute language concepts
- **Mid-level**: func, cont, scf, arith, mem, adt — structured operations
- **Low-level**: cf, wasm, clif, trampoline — target-specific

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
let func_ty = core::func(ctx, return_ty, params, effect).as_type_ref();
```

Operations use the same pattern:

```rust
let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
```

Matching uses typed wrappers (see conventions.md for ✅/❌ patterns):

```rust
if let Ok(func) = func::Func::from_op(&ctx, op) { ... }
if func::Call::matches(&ctx, op) { ... }
```
