# TrunkIR System

TrunkIR is Tribute's multi-level dialect IR, inspired by MLIR's dialect concept.

## Core Structures

- **`IrContext`** — Arena-based context holding all operations, blocks,
  regions, and types
- **`OpRef`**, **`ValueRef`**, **`BlockRef`**, **`RegionRef`**, **`TypeRef`**
  — Arena references to IR entities
- **`Symbol`** — Interned name (8 bytes, O(1) equality, `Clone` but not
  `Copy`). Take `&Symbol` where the name is only read, and clone it only to
  store it. Names of at most 7 bytes are inline, trunk-ir's own declared names
  are static, and other names are reference-counted and released with their
  last symbol. A qualified name is a `SymbolPath`, one `Symbol` per component.
- **`OpName`**, **`TypeName`** — An operation's or type's `dialect.name`,
  `Copy` and compared by handle, not text. See
  [operation and type names](#operation-and-type-names).

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
selecting closure storage. Each operation's own contract is the doc comment
on its declaration in
[`tribute_control`](../../crates/tribute-ir/src/dialect/tribute_control.rs) and
[`ability`](../../crates/tribute-ir/src/dialect/ability.rs). See
[the IR contract](../../new-plans/ir.md) for the rules that span operations, and
[the shared pipeline](../../new-plans/cps-effects.md#shared-middle-end-pipeline).

The target [representation/ABI boundary](../../new-plans/ir.md#representationabi-경계)
ends with an exit verification that rejects any violation in every build; later
passes read only the physical contracts it checked. After the frontend, the
pipeline is one ordered chain of stages that `stage_passes` in
[src/pipeline.rs](../../src/pipeline.rs) builds; each entry point runs a
contiguous range of it as one `PassManager`. `pipeline::dump_ir`
(`--dump-ir`) runs the chain to that exit and prints the module, and tests
split a compilation there with `run_target_to_boundary_exit` and
`emit_from_boundary_exit`. The stage order is in the top-of-file comment in
[src/pipeline.rs](../../src/pipeline.rs).

## `#[dialect]` Macro

Operations and types are defined with the `#[trunk_ir::dialect]` attribute
macro on a `mod`; it takes no arguments.

The doc comment on an operation or type declaration is copied onto its
generated wrapper. It states that one entity's contract: its shape, meaning,
verification, ownership and value flow, and location, with its textual form
in a `text` code block. It is authoritative for that entity; rules that span
entities stay in [`new-plans/`](../../new-plans/). Write it in English and
refer to `new-plans/` sections by file and heading rather than Markdown links,
which rustdoc cannot resolve.

- `struct` definitions generate typed type wrappers. IR type parameters are
  declared in `<..>`, where `#[rest]` marks a variadic last one, and type
  attributes as fields typed like operation attributes:
  `struct Ref<Pointee> { nullable: Attr<bool> }`, with optional ones as
  `Option<Attr<K>>`. A wrapper accepts only types with the declared
  parameters and attributes.
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
    signature: Attr<TypeOf<S>>,
    callee: Value<_>,
    args: Values<S::Inputs>,
) -> Values<S::Results> {}

fn perform<F: Frame, C>(frame: Value<F>, resumption: Value<C>)
where
    C: CpsClosure<Inputs = (Evidence, F, impl NotNever)>,
{}
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
- `Attr<TypeOf<V>>` is a type attribute whose value binds the type variable
  `V`. `V` needs no bound.
- A list attribute's accessor iterates its elements, and its builder setter
  takes an iterator. For `Attr<[String]>` the accessor yields `&str` and
  `<name>_ref` yields the `StringRef`s, as for a single string. `String` is
  the only kind the macro recognizes by name, to generate `<name>_ref`.
- `Attr<Dict<V>>` (`attr_kind::Dict`) is a dictionary whose every value has
  kind `V`. Its accessor returns a view with `get(key)`, `len()`, and
  `iter()` in key order, and its builder setter takes the
  `(StringArg, value)` entries.
- Regions and successors are declared in the body: `#[region(name)] {}`,
  `#[region(name?)] {}` for an optional last region such as the body of an
  external function, and `#[successor(name)] {}`. `#[successors(name)] {}`
  declares the rest of the successor list and must come last; its accessor
  iterates those blocks.
- Bounds are Rust types implementing `type_constraint::TypeConstraint`:
  `core` scalar categories (`IntegerLike` for fixed-width integers,
  `IntegerOrIndex` for those and `core.index`, `BoolLike`, `FloatLike`,
  `NumericLike`, `ScalarLike`), exact `core` scalars (`I1`–`I64`, `Index`,
  `F32`, `F64`), macro-defined type wrappers
  (projections are their declared parameters), and the `func`/`clif`/`wasm`
  `FuncSig` wrappers (`Inputs`/`Results`). Unknown, ambiguous, or wrong-kind
  projections and conflicting exact bounds fail to compile.
- A variable's bound may constrain its projections with associated item
  constraints, `C: CpsClosure<Inputs = (Evidence, F)>`, in the generic list
  or in `where`. The right-hand side is a single-type constraint, an exact
  list `(A, B)`, or another projection of the same kind. A named variable on
  the right-hand side is bound there if nothing bound it earlier, so a type
  that occurs only inside an operand's type can be named and projected
  again (`F::Result`). A constraint that projects a variable nothing binds
  fails to compile.
- A bound written directly in a result, without `impl`, is that one fixed
  type (`-> Value<I32>`); it must denote exactly one type.
- `#[verify]` on an operation requires `trunk_ir::ops::Verify` for its
  wrapper. It checks what the schema cannot express and runs only after
  every generated check passed.
- `#[asm_format]` on an operation requires `trunk_ir::asm_format::OpAsmFormat`
  for its wrapper, which prints and parses the operation's custom syntax.
  Generic syntax stays readable for every operation.
- On a type `struct`, `#[verify]` requires `trunk_ir::type_verifier::VerifyType`
  and `#[asm_format]` requires `trunk_ir::asm_format::TypeAsmFormat`. A
  hand-written `TypeDef` adds the same hooks with `with_verifier` and
  `with_asm_format`. A `#[verify]` type's wrapper accepts only types that
  pass the verifier, so its `params(..)` builder's `intern` runs the verifier
  and returns a `Result`. `#[alias_hint]` requires
  `trunk_ir::asm_format::TypeAliasHint`, the alias name the printer gives
  the type when it generates aliases; a hand-written `TypeDef` adds it with
  `with_alias_hint`. `#[implements(..)]` on a type registers the wrapper's
  [type interfaces](#type-interfaces).
- A hand-written `TypeDef` is a `static` registered with
  `trunk_ir::register_type_def!(&DEF)`. Dialect crates never use
  `inventory` directly; registration goes through `#[dialect]` and these
  macros.

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
one call, in declaration order; variadic successors are passed as an
iterator. Missing required inputs panic in `build`; the builder does not check
input types.

**Operation definition**: every generated operation wrapper exposes
`DialectOp::DEF`, the `static` `op_def::OpDef` the declaration emits.
`OpDef::of(ctx, op)` reads it from the operation's name handle. An `OpDef`
holds the declarative `op_schema::OpSchema` and the hooks its declaration
opts into, such as `#[verify]` and `#[asm_format]`. `OpSchema::verify` runs
the declarative stages: counts and attributes, individual type constraints,
variable bindings, then projections and type lists. `OpDef::verify` adds the
`#[verify]` hook after them; each stage runs only if the earlier ones passed.
`validate_operation_verifiers` runs `OpDef::verify` on each operation.
Remaining operation-specific checks run afterwards and may assume the
declared shape. The `tribute_control` local validator and the native backend
boundary (`validate_clif_ir`) run `OpDef::verify` the same way before their
own checks, so those checks cover only what the definition cannot express,
such as symbol lookups, enclosing callables, and region contents.
Debug builds also run `validate_op_schemas`, the declarative stages without the
`#[verify]` hook, after every pipeline pass, reporting the pass that left an
operation in violation. Wasm emission is the exception: after its upcast elision
(below) the IR follows the backend's assignability rule rather than the declared
schemas, so emission runs its cast passes without this check. A pass that
retypes a value before target type conversion inserts
`core.unrealized_conversion_cast` back to the type its uses declare.
`PatternApplicator` does this for a staged conversion with
`with_auto_type_conversion(true)` and a `ConversionTarget`. It casts the
operands only of operations the target declares illegal. Where it retypes a
block argument or a replaced result that an operation the target has no rule
for still uses, it casts back to the old type if the new one violates that
operation's schema. A pass that erases or recovers a value's type emits any
representation change itself: `tribute_rt::convert_value` boxes or unboxes a
scalar crossing `tribute_rt.anyref` and casts every other pair, so a cast
before target conversion only retypes a value. A target
conversion converts every value's type, including cast results through
`UnrealizedCastConversionPattern`; its materializer builds only real
representation changes and never forwards a value of another type. A target
whose type system accepts subtype references writes those upcasts as its own
explicit operation, since schemas compare types exactly; the Wasm target's
`ReferenceUpcastPattern` turns them into `wasm.ref_upcast` by the backend's physical
assignability rule. The converter-free `ReconcileUnrealizedCasts` pass then folds
identities and cast chains, and a cast left after that is rejected by the
target emission boundary.
Typed accessors do not check the schema; for an optional region or result,
inspect the operation before calling the accessor. An interface queried on
unverified IR, such as `Branch` for the validator, reads attributes fallibly
and reports failure in its result; an interface queried only on IR that
satisfies its schema, such as `CallLike`, reads through the accessors.

## Operation Interfaces

An operation interface is a trait that operation wrappers implement,
declared with `#[trunk_ir::op_interface]`:

```rust
#[op_interface]
pub trait Branch: DialectOp {
    fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError>;
}
```

An operation declares the interfaces it implements in its `#[dialect]`
declaration, and implements each one next to it:

```rust
#[trunk_ir::dialect]
mod cf {
    #[implements(Branch)]
    fn br(args: Variadic<_>) {
        #[successor(dest)]
        {}
    }
}

impl Branch for Br {
    fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> { .. }
}
```

Each method takes `&self`, then the context (`&IrContext` or
`&mut IrContext`), and may be generic over lifetimes only; default bodies
are dispatched like other methods. A result borrowed from the context names
the context's lifetime, since elision would tie it to `&self`, as
`direct_callee` does below. The trait hooks a declaration opts into, such as
`Verify` and `OpAsmFormat`, take `&self` the same way. `#[implements(..)]`
registers the implementation by the operation's name, and a missing `impl`
is reported at its entry. The paths resolve where the `#[dialect]` module is declared.
Registering an operation twice for one interface panics when that interface
is first queried. An operation that `#[dialect]` does not declare, or an
interface added to another crate's operation, is registered with
`register_interfaces!(Op: InterfaceA, InterfaceB; Other: InterfaceC)`.

The macro generates `DynBranch`, a copyable handle to an operation and its
implementation:

```rust
if let Some(branch) = DynBranch::get(ctx, op) {
    let successors = branch.successors(ctx)?;
}
let is_branch = DynBranch::is_implemented(ctx, op);
```

Every interface query starts from the handle. A query that combines
interfaces, or turns a missing implementation into an error, is an
associated function in a hand-written `impl DynFoo` block next to the trait,
such as `DynRegionBranch::value_transfer(ctx, op, point, successor)` and
`DynCallableExit::prove(ctx, op)`. Other queries call the handle's methods
directly, getting the handle once when they ask it several things:

```rust
let call = DynIndirectCallLike::get(ctx, op)?;
let (callee, signature) = (call.indirect_callee(ctx)?, call.exact_signature(ctx));
```

Any crate can declare an interface; an implementation follows the orphan
rule, so it lives with the interface or with the operation.

### Operation Flags

An operation flag is a property without behavior, such as `Pure` and
`IsolatedFromAbove` in `trunk_ir::op_flag`. An operation lists its flags in
`#[flags(..)]`; they are stored in its `OpDef`, so a flag has no `impl` and
no registry:

```rust
#[trunk_ir::dialect]
mod arith {
    #[flags(Pure)]
    fn addi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
}

let removable = Pure::is_set(ctx, op); // `OpFlag::is_set`
let isolated = def.has_flag::<IsolatedFromAbove>();
```

Only the dialect that defines an operation sets its flags. A property that
another crate attaches to an operation it does not define is an interface,
even without methods. Any crate declares a flag as a unit struct with
`#[trunk_ir::op_flag]`, which adds the `static` `OpFlagDef` that identifies
it by address and its `OpFlag` impl:

```rust
#[op_flag]
pub struct Pure;
```

### Type Interfaces

A type interface is the same mechanism for types, declared with
`#[trunk_ir::type_interface]` on a trait over `DialectType`. A `#[dialect]`
type lists it in `#[implements(..)]`, and `register_interfaces!` registers
the impl for a type declared elsewhere:

```rust
#[type_interface]
pub trait ManagedRefType: DialectType {}

#[trunk_ir::dialect]
mod closure {
    #[implements(ManagedRefType)]
    struct Closure<FuncType>;
}

impl ManagedRefType for Closure {}
impl ManagedRefType for core::Bytes {}
register_interfaces!(core::Bytes: ManagedRefType);

let managed = DynManagedRefType::is_implemented(ctx, ty);
```

The registry is keyed by type name, and `DynX::get(ctx, ty)` returns a
handle whose `ty()` is the queried type. A method is called on the type's
wrapper, so query a type interface's methods only on types that pass their
definition's shape and verifier; a marker interface answers by name alone.
A hook that core infrastructure reaches through the definition, such as a
verifier, an assembly format, or an alias hint, belongs on `TypeDef`
instead.

## Working with IR

A declared type is built like an operation: start from its parameters in
declaration order, set attributes by name, and intern it. `param_attrs` sets
the parameters' own attributes. A `#[verify]` type is verified when it is
interned, so its `intern` returns a `Result`; other types return the
wrapper. Convert a wrapper with `.as_type_ref()`:

```rust
let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
let ref_ty = core::Ref::params(pointee).nullable(false).intern(ctx);
let func_ty = func::func_sig(ctx, params, [return_ty]).as_type_ref();
```

The source signature `tribute_control.func_sig` is declared by hand but is
built the same way. A signature without a convention is one that class
specialization has not assigned yet:

```rust
let callback = tribute_control::FuncSig::params([i32_ty], i32_ty)
    .row(row)
    .convention(CallingConvention::Cps)
    .intern(ctx);
let direct = callback.with_convention(ctx, CallingConvention::Direct);
```

String attribute values live in the context's string pool; an attribute
holds only a `StringRef`. Create and read them through the context, and do
not carry a `StringRef` to another context:

```rust
let abi = ctx.string_attr("C");
attrs.view_mut(ctx).insert("abi", abi);
let is_c = ctx.op_attrs(op).get_str("abi") == Some("C");
let value: &str = string_const.value(ctx); // generated accessor
```

A builder setter for a string attribute takes a `StringArg`: a `StringRef`,
or text as a `Cow<'static, str>` (a `&'static str` or an owned `String`).
The builder interns text when it creates the operation, so
`.predicate("slt")` and `.sym_name("main")` need no context. Pass text directly
instead of constructing a `Symbol` solely for a string attribute. For a borrowed
runtime name, pass `ctx.intern_str(name)`. `Symbol` and `SymbolPath` do not convert
to a string argument automatically. For an existing symbol, pass
`ctx.intern_symbol_text(&symbol)` explicitly. `SymbolPath` represents a symbol
reference; use `path.to_string()` when its textual spelling is needed as a
string value.

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

Attribute keys are names, not symbols: an `AttributeMap` keys its entries by
`StringRef` in the same pool, ordered by their text. The map itself offers
only what needs no key text (`len`, `iter` over `StringRef` keys, `values`);
every lookup and change goes through a view bound to the context:

- `ctx.op_attrs(op)`, `ctx.type_attrs(ty)`, and `attrs.view(ctx)` for any
  other map read by key text or handle, `get_type("type")`, and print keys
  and string values as text with `{:?}`.
- `ctx.op_attrs_mut(op)`, `ctx.block_arg_attrs_mut(block, i)`, and
  `attrs.view_mut(ctx)` for a map outside the context change it, interning
  new keys. The view borrows the context, so build any value that needs the
  context before taking it.

```rust
let callee = ctx.op_attrs(op).get_symbol_ref("callee");
let is_bodyless = !ctx.op_attrs(op).contains_key("body");
let name = ctx.string_attr("x");
let mut attrs = ctx.op_attrs_mut(op);
attrs.insert("type", Attribute::Type(ty));
attrs.remove("signature");
ctx.block_arg_attrs_mut(block, 0).insert("bind_name", name);
```

`AttributeMap::from_entries(ctx, [("bind_name", name)])` builds a map, and
`AttributeMap::from_refs(ctx, entries)` builds one from keys already pooled
in the context, such as the entries of another map. A type is built with
`TypeDataBuilder`, whose `attr` keys are interned when it is built:
`TypeDataBuilder::new("core.i32").intern(ctx)` builds and interns it.

Attributes that embed types (directly or inside `List`/`Dict` values) are
converted and inspected through the shared traversal instead of matching
individual variants:

```rust
let converted = attribute.map_types(|ty| converter.convert_type_or_identity(ctx, ty));
attribute.visit_types(&mut |ty| seen.push(ty));
```

A name that refers to a symbol table definition is an `Attribute::SymbolRef`
(`callee = @foo`, declared `Attr<SymbolRef>`). It holds a `SymbolPath`, one
`Symbol` per component, resolved as MLIR resolves symbols: the first component
in the module nearest to the operation that holds the reference, never in an
enclosing one, and each further component in the module the previous one
names. Definition names and other fixed names are strings. Collect the
references an operation makes with the shared traversal rather than by
listing attribute names:

```rust
ctx.op(op).attributes.visit_symbol_refs(&mut |path| referenced.push(path.clone()));
```

A path is built from its components: `SymbolPath::new(["main"])` is a
definition in the root module, and `SymbolPath::new(["left", "helper"])` is
`helper` in the module `left`. Text and symbols do not convert into a path,
so the components are always spelled out. A name is never split:
`SymbolPath::new(["left::helper"])` is one component, a root definition whose
name contains `::`. The printer writes
`@left::@helper` for the nested path and `@"left::helper"` for the single
name. Read a path with `leaf()`, `modules()`, and `as_simple()`; it
dereferences to the slice of its components, so a function that only reads
a path takes `&[Symbol]`. A path compares with another path, or with an
array of its components: `path == ["std", "io", "Io"]` compares text
component by component, and `path == [symbol!("main")]` compares atoms.
The path without components, `SymbolPath::root()`, is the root module, and
`parent()` is the module that encloses a path. Build a nested path with
`module.child(name)`, replace the last component with `sibling(name)`, and
walk nested modules with `push` and `pop`. A definition's path and a symbol
reference have at least one component, so `leaf()` is `None` only for the
root module; IR validation rejects a reference to it.
A definition is identified by its operation, not by a path. `SymbolTable`
holds one scope per symbol table operation, and the analyses that name
functions (call graph, global DCE, the inliner) are keyed by the defining
`OpRef`. Resolve a reference from the operation that holds it with
`SymbolTable::resolve(ctx, op, path)`, or from a given table with
`resolve_in(ctx, table, path)`; an unresolved or duplicated name yields
`None`. `symbol_table::qualified_name` spells a definition's path from the
root for diagnostics only; do not look a written reference up by it.

A compiler-owned runtime item is identified by its canonical path under
`std::runtime`: lowering reads a runtime function with
`tribute_ir::runtime_items::RuntimeFunction::lookup` and a runtime type with
`runtime_items::runtime_type`, which reads the root module's `tribute.types`
dictionary through `tribute_ir::metadata`. See
[the runtime module](../../new-plans/modules.md#compiler-소유-runtime-모듈).

A direct call operation implements the `CallLike`
[operation interface](#operation-interfaces), declared with
`#[implements(CallLike)]`, so analyses can tell its callee
from an address reference. Each implementation names its callee through the
wrapper's accessor:

```rust
impl CallLike for Call {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}

let callee = DynCallLike::get(ctx, op).map(|call| call.direct_callee(ctx));
```

Every other reference an operation holds takes its target's address, which
the call graph records as an escape.

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

### Operation and Type Names

`OperationData::name` is an `OpName` and `TypeData::name` a `TypeName`, one
handle for the dialect and the name together, as MLIR's `OperationName`
is (see [the IR contract](../../new-plans/ir.md)). A name declared by
`#[dialect]` is registered: its handle points to the `static` definition the
declaration emits (`OpDef`, `TypeDef`), so it is valid in every context and
compares by address. A typed wrapper's `NAME` is that handle, and `matches`
compares it. Any other name, such as an operation the parser reads in
generic syntax or a test dialect's, keeps its text in the context's string
pool and is valid only in that context.

Builders take a definition, a handle, or the `dialect.name` text, split at
its first `.`; text is resolved when the builder builds, to the registered
definition when there is one:

```rust
let op = OperationDataBuilder::new(loc, "test.op").build(ctx);
let copy = OperationDataBuilder::new(loc, ctx.op(op).name).build(ctx);
let ty = TypeDataBuilder::new(core::I32::NAME).intern(ctx);
let name: OpName = ctx.intern_op_name("test.op");
```

Read a name's text through the context, the way a string attribute is read:
`name.dialect(ctx)`, `name.strip_dialect(ctx)`, and `name.display(ctx)` for
`dialect.name`. Prefer a wrapper's `matches` to comparing text; compare text
with `name.is(ctx, "dialect.name")` only for a name no wrapper declares.
Compare a dialect with the constant `#[dialect]` generates for it,
`name.dialect(ctx) == wasm::DIALECT_NAME`, rather than a string literal.

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
