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

Operations and types are defined with the `#[trunk_ir::dialect]` attribute
macro on a `mod`; it takes no arguments.

- `struct` definitions generate typed type wrappers. `#[attr(..)]` declares
  type attributes, and `#[rest]` marks a variadic last type parameter.
- `fn` definitions declare operations. The signature declares each entity and
  its type constraint, following
  [the declarative schema contract](../../new-plans/ir.md#선언적-operation-schema).
  `_` leaves an entity unconstrained. For example:

```rust
fn addi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}

fn i32_add(lhs: Value<I32>, rhs: Value<I32>) -> Value<I32> {}

fn cmpi<T: IntegerLike>(
    predicate: Attr<Symbol>,
    lhs: Value<T>,
    rhs: Value<T>,
) -> Value<I1> {}

fn resume<T: ResumeToken>(
    resume_token: Value<T>,
    value: Value<T::Input>,
) -> Value<T::Answer> {}

#[verify]
fn call_indirect<S: FuncSig>(
    signature: Attr<S::Type>,
    callee: Value<_>,
    args: Values<S::Inputs>,
) -> Values<S::Results> {}
```

- Parameters are `Value<C>`, `Variadic<C>`, `Values<L>`, `Attr<K>`, and
  `Option<Attr<K>>`. Results are `Value<C>` or `Option<Value<C>>` (accessor
  `result`) or `Variadic<C>` / `Values<L>` (accessor `results`).
- Regions and successors are declared in the body: `#[region(name)] {}`,
  `#[region(name?)] {}` for an optional last region such as the body of an
  external function, and `#[successor(name)] {}`.
- Bounds are Rust types implementing `type_constraint::TypeConstraint`:
  `core` scalar categories (`IntegerLike`, `BoolLike`, `FloatLike`), exact
  `core` scalars (`I1`–`I64`, `F32`, `F64`), macro-defined type wrappers
  (projections are their declared parameters), and the `func`/`clif`/`wasm`
  `FuncSig` wrappers (`Inputs`/`Results`). Unknown, ambiguous, or wrong-kind
  projections and conflicting exact bounds fail to compile.
- A bound written directly in a result, without `impl`, is that one fixed
  type (`-> Value<I32>`); it must denote exactly one type.
- `#[verify]` on an operation requires an implementation of
  `trunk_ir::ops::Verify` (`fn verify(self, ctx: &IrContext) ->
  Result<(), String>`) for its wrapper, and reserves the entity name
  `verify`. It checks what the schema cannot express and runs only after
  every generated check passed.
Each operation gets a builder that groups inputs by entity kind. For the
declarations above:

```rust
let sum = arith::Addi::operands(lhs, rhs).build(ctx, loc); // result is `T`
let cmp = arith::Cmpi::operands(lhs, rhs) // or `Op::operands()` without operands
    .predicate(Symbol::new("slt"))          // attributes by name
    .build(ctx, loc);                       // result is `core.i1`
let resumed = Resume::operands(token, value).build(ctx, loc); // `T::Answer`
```

The builder infers result types that are fixed types, variables bound by a
single operand or a required attribute, or projections of such variables;
`.results(..)` exists only when they are not all inferred, as for
`-> Value<impl BoolLike>`. Result types,
regions (`.regions(..)`), and successors (`.successors(..)`) are each set in
one call, in declaration order. Missing required inputs panic in `build`; the
builder does not check input types.

**Operation definition**: every generated operation wrapper exposes
`DialectOp::DEF`, a static `op_def::OpDef` registered by operation name.
`OpDef::of(ctx, op)` looks it up for any operation. An `OpDef` holds the
declarative `op_schema::OpSchema` and the hooks its declaration opts into,
such as `#[verify]`. `OpSchema::verify` runs the declarative stages: counts
and attributes, individual type constraints, variable bindings, then
projections and type lists. `OpDef::verify` adds the `#[verify]` hook after
them; each stage runs only if the earlier ones passed.
`validate_operation_verifiers` runs `OpDef::verify` on each operation.
Remaining operation-specific checks run afterwards and may assume the
declared shape. The `tribute_control` local validator and the native backend
boundary (`validate_clif_ir`) run `OpDef::verify` the same way before their
own checks, so those checks cover only what the definition cannot express,
such as symbol lookups, enclosing callables, and region contents.
Debug builds also run `validate_op_schemas`, the declarative stages without
the `#[verify]` hook, after every shared middle-end pass, reporting the pass
that left an operation in violation; target lowering passes are checked at the
backend boundary.
Typed accessors do not check the schema; for an optional region or result,
inspect the operation before calling the accessor. Interface queries that may
see unverified IR read attributes fallibly instead.

## Working with IR

Types are created via module-level constructors and converted with
`.as_type_ref()`:

```rust
let nil_ty = core::nil(ctx).as_type_ref();
let func_ty = func::func_sig(ctx, params, [return_ty]).as_type_ref();
```

Operations are created with their builders:

```rust
let c = arith::Const::operands()
    .value(Attribute::Int(42))
    .results(i32_ty)
    .build(&mut ctx, loc);
```

Matching uses typed wrappers (see [code conventions](conventions.md) for
✅/❌ patterns):

```rust
if let Ok(func) = func::Func::from_op(&ctx, op) { ... }
if func::Call::matches(&ctx, op) { ... }
```
