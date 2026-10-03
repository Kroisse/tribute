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
  wasm)
- **tribute-ir** (`crates/tribute-ir/src/dialect/`):
  Tribute-specific dialects (tribute_control, ability, effect, closure, adt,
  list, tribute_io, tribute_rt)

Dialect levels (high → low):

- **High-level**: tribute_control, ability, effect, closure, adt, list,
  tribute_io, tribute_rt — Tribute language concepts
- **Mid-level**: func, scf, arith, mem — structured operations
- **Low-level**: cf, wasm, clif — target-specific

## Source-logical Control

The frontend emits `tribute_control` callable/control operations with exact
source signatures and checked operation kinds. Shared `tribute_control_to_cps`
legalization alone constructs continuations and the `func`/`closure` graph.
Target ABI lowering validates that graph before physicalizing CPS results and
selecting closure storage. See [the IR contract](../../new-plans/ir.md) and
[the shared pipeline](../../new-plans/cps-effects.md#shared-middle-end-pipeline).

The target [representation/ABI boundary](../../new-plans/ir.md#representationabi-경계)
ends with an exit verification that rejects any violation in every build; later
passes read only the physical contracts it checked. `pipeline::dump_ir`
(`--dump-ir`) prints the module at that exit, and tests split a compilation
there with `run_target_to_boundary_exit` and `emit_from_boundary_exit`. The
stage order is in the top-of-file comment in [src/pipeline.rs](../../src/pipeline.rs).

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
    predicate: Attr<String>,
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

- Parameters are `Value<C>`, `Variadic<C>`, `Values<L>`, `Attr<K>`,
  `Attr<[K]>` (a list whose every element has kind `K`), and
  `Option<Attr<..>>`. Results are `Value<C>` or `Option<Value<C>>` (accessor
  `result`) or `Variadic<C>` / `Values<L>` (accessor `results`).
- Attribute kinds are Rust types implementing `attr_kind::AttrKind`, resolved
  in the dialect's scope like bounds: primitives (`bool`, `i32`, `i64`, `u32`,
  `u64`, `f32`, `f64`), `String`, and the markers
  `attr_kind::{Type, SymbolRef, Bytes}`. A kind defines its schema domain
  and the values its accessor returns and its builder setter takes, so a
  dialect adds a kind by implementing the trait. `_` accepts any attribute.
- A list attribute's accessor iterates its elements, and its builder setter
  takes an iterator. For `Attr<[String]>` the accessor yields `&str` and
  `<name>_ref` yields the `StringRef`s, as for a single string. `String` is
  the only kind the macro recognizes by name, to generate `<name>_ref`.
- `Attr<Dict<V>>` (`attr_kind::Dict`) is a dictionary whose every value has
  kind `V`. Its accessor returns a view with `get(key)`, `len()`, and
  `iter()` in key order, and its builder setter takes the
  `(Symbol, value)` entries.
- Regions and successors are declared in the body: `#[region(name)] {}`,
  `#[region(name?)] {}` for an optional last region such as the body of an
  external function, and `#[successor(name)] {}`.
- Bounds are Rust types implementing `type_constraint::TypeConstraint`:
  `core` scalar categories (`IntegerLike`, `BoolLike`, `FloatLike`,
  `NumericLike`, `ScalarLike`), exact `core` scalars (`I1`–`I64`, `F32`,
  `F64`), macro-defined type wrappers
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
- A type whose data must satisfy rules beyond its generic shape registers
  `inventory::submit! { TypeVerifier::new::<T>(verify_fn) }`
  (`trunk_ir::type_verifier`). IR validation runs it on every interned type
  of that kind; `func.func_sig` and `adt.struct` register theirs this way.

Each operation gets a builder that groups inputs by entity kind. For the
declarations above:

```rust
let sum = arith::Addi::operands(lhs, rhs).build(ctx, loc); // result is `T`
let cmp = arith::Cmpi::operands(lhs, rhs) // or `Op::operands()` without operands
    .predicate("slt")                      // attributes by name
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
the `#[verify]` hook, after every pipeline pass, reporting the pass that left
an operation in violation. A pass that retypes a value before target type
conversion inserts `core.unrealized_conversion_cast` back to the type its uses
declare. Shared cleanup's `materialize_unrealized_casts` materializes only
casts that need real operations and keeps such retyping casts. A target
conversion converts every value's type, including cast results through
`UnrealizedCastConversionPattern`; its materializer builds only real
representation changes and never forwards a value of another type. A target
whose type system accepts subtype references elides those upcasts itself; the
Wasm target's `ReferenceUpcastElisionPattern` follows the backend's physical
assignability rule. The converter-free `reconcile_unrealized_casts` then folds
identities and cast chains, and a cast left after that is rejected by the
target emission boundary.
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

String attribute values live in the context's string pool; an attribute
holds only a `StringRef`. Create and read them through the context, and do
not carry a `StringRef` to another context:

```rust
let abi = ctx.string_attr("C");
attrs.insert("abi", abi);
let is_c = ctx.op(op).attributes.get_str(ctx, "abi") == Some("C");
let value: &str = string_const.value(ctx); // generated accessor
```

A builder setter for a string attribute takes a `StringArg`: a `StringRef`,
or text as a `Cow<'static, str>` (a `&'static str` or an owned `String`).
The builder interns text when it creates the operation, so
`.predicate("slt")` needs no context.

A generated string accessor returns the text, like MLIR's `getValue()`.
`<name>_ref` returns the `StringRef`; use it to copy the
value into another operation without borrowing the context or interning
again:

```rust
let copy = adt::StringConst::operands()
    .value(string_const.value_ref(ctx))
    .results(ty)
    .build(ctx, loc);
```

Attributes that embed types (directly or inside `List`/`Dict` values) are
converted and inspected through the shared traversal instead of matching
individual variants:

```rust
let converted = attribute.map_types(|ty| converter.convert_type_or_identity(ctx, ty));
attribute.visit_types(&mut |ty| seen.push(ty));
```

A name that refers to a symbol table definition is an `Attribute::SymbolRef`
(`callee = @foo`, declared `Attr<SymbolRef>`). Definition names and other
fixed names are strings. Collect the references an operation makes with the
shared traversal rather than by listing attribute names:

```rust
ctx.op(op).attributes.visit_symbol_refs(&mut |symbol| referenced.push(symbol));
```

A direct call operation registers `CallLike` so analyses can tell its callee
from an address reference. `CallLikeOps::callee(ctx, op)` returns the callee;
every other reference an operation holds takes its target's address, which
the call graph records as an escape:

```rust
impl CallLikeModel for Call {}
inventory::submit! {
    CallLikeOps::register::<Call>()
}
```

Per-parameter attributes (`param_attrs`, see
[the type model](../../new-plans/ir.md#타입-매개변수-속성)) are built with the
parameter they describe and read back by position:

```rust
let sig = func::func_sig_with_param_attrs(
    ctx,
    [(env_ty, AttributeMap::new()), (value_ty, attrs)],
    [],
    AttributeMap::new(),
);
let first = sig.input_attrs(ctx).next();
let data = ctx.get_type(sig.as_type_ref());
let value_attrs = data.param_attrs(1);
```

A rebuild that inserts or removes parameters edits the parameter/attribute
pairs through `FuncSig::rebuild`, so each attribute stays with its parameter.
Copying the attributes into `func_sig_with_attrs` would keep the old
positions:

```rust
let rebuilt = sig.rebuild(ctx, |inputs, _| {
    inputs.insert(index, (env_ty, AttributeMap::new()));
});
```

An inserted parameter carries whatever its inserting layer's contract assigns.
Inside the representation/ABI boundary, a parameter inserted into a physical
callable takes `target_abi::physical_parameter_attrs(convention)`, so it has
the same ownership contract as the callable's other inputs.

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

An operation's regions and successors are read through the context, not
through `OperationData`: `ctx.op_regions(op)` and `ctx.op_successors(op)`
iterate `RegionRef`/`BlockRef`, `ctx.op_region(op, i)` and
`ctx.op_successor(op, i)` index them, and `op_region_count` /
`op_successor_count` give their lengths. Ask `op_has_regions` /
`op_has_successors` whether a list is empty: the region list is linked, so
counting it walks every region. Change them through the context's
methods (`push_op_region`, `clear_op_regions`, `detach_region`,
`set_op_successor`, `truncate_op_successors`), which keep region parent
links consistent. Prefer a typed accessor such as `func.body(ctx)` when the
operation's wrapper declares the region.

`op_region` and `op_successor` return `Option`. Outside tests, do not
`unwrap()` them: propagate a missing entry with `?` or `ok_or_else` into the
function's error, fall back as a printer does to generic output, or use
`expect` with a message that names the violated contract.

When a loop must mutate the IR while walking such a list, collect a snapshot
into the list's `SmallVec` alias rather than a `Vec`. These lists rarely
exceed four entries, so the alias does not allocate:

```rust
let regions: RegionList = ctx.op_regions(op).collect();
let successors: BlockList = ctx.op_successors(op).collect();
let ops: OpList = ctx.block(block).ops.clone();
for region in regions {
    rewrite_region(ctx, region);
}
```

Tests compare such a list with a `smallvec_inline!` literal, or compare its
slice with an array, instead of collecting into a `Vec`:

```rust
let regions: RegionList = ctx.op_regions(op).collect();
assert_eq!(regions, smallvec_inline![body, completion]);
assert_eq!(regions[..], [body, completion]);
```
