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
  tribute_control
            target-independent direct-style effect control
  ability   evidence and handler dispatch semantics
  effect    target-independent effect ABI
  closure   closure construction and decomposition
  adt       structs, variants, arrays, references, literals
  list      opaque persistent-sequence construction and observation

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

The direct-control migration uses these named boundaries:

| Boundary | `ConversionTarget` mode | Required legality |
| ---- | ---- | ---- |
| `tribute-control-pre-cps` | Full for frontend conformance; partial while composing migration passes | `tribute_control.*` may coexist with `core`, `func`, `scf`, `arith`, `adt`, `list`, `closure`, `tribute_rt`, and `tribute_io`. Existing `ability.*`, `effect.*`, and legacy CPS construction operations are illegal. |
| `tribute-control-post-cps` | Partial after the shared CPS conversion | The entire `tribute_control` dialect is illegal. Existing `ability.perform`, `ability.call`, handler/evidence operations, `effect.*`, `closure.*`, and ordinary dialects may coexist for their subsequent shared passes. |
| `tribute-backend-ready-native` | Full Tribute boundary followed by the generic Cranelift boundary | `tribute_control`, `ability`, `effect`, `closure`, `list`, `tribute_io`, and conversion casts are absent. Only the explicitly listed native infrastructure and `clif.*` operations remain. |
| `tribute-backend-ready-wasm` | Partial high-level rejection followed by the existing emission-ready checks | `tribute_control`, `ability`, and `effect` are explicitly illegal before backend-ready Wasm IR; `wasm_gc` is additionally illegal before emission. |

The post-CPS helper is implementable with the current API as
`ConversionTarget::new().illegal_dialect("tribute_control")` plus partial
verification. Frontend conformance and backend full targets must enumerate
their legal dialects and operations because unknown is not legal in full mode.
They must not mark `func.func`, `closure.lambda`, or another region-owning
container recursively legal: doing so would hide illegal control operations in
nested regions. The generic `trunk-ir` Cranelift boundary remains
Tribute-agnostic; the Tribute pipeline owns the explicit rejection of
Tribute-specific high-level dialects before invoking it.

Transiently, one rewrite may contain both source `tribute_control.*` and its
new `ability.*`/`effect.*` output. That is an implementation state inside
partial conversion, not a successful named boundary. Successful
`tribute-control-post-cps` verification reports every residual
`tribute_control.*` operation with its source location as a conversion
failure.

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

### Direct-style control

`tribute_control.*` is the target-independent, direct-style boundary between
typed frontend lowering and shared CPS conversion. The dialect identifier is
exactly `tribute_control`. TrunkIR parses a qualified operation as one
`<dialect>.<operation>` pair, so spellings with another separator, such as a
dot inside the dialect identifier, are invalid.

The dialect models only control constructs that the CPS pass must transform.
ANF is an input invariant, not the dialect's identity. Arithmetic, ordinary
direct and indirect calls, ADT/list/tuple/record construction, closures, and
structured selection remain in their existing dialects. In particular, there
is no `tribute_control.invoke`: `func.call`, `func.call_indirect`, and
`closure.lambda` already carry the callable type and
`Direct < EvidenceDirect < Cps` convention metadata needed by shared
conversion. A new invocation operation would duplicate those operations
without adding a control fact.

At the pre-CPS boundary, all `func.func`, `closure.lambda`, `func.call`, and
`func.call_indirect` signatures are source-logical regardless of convention:
source parameters in, source result out, with no hidden evidence or `done_k`
operands. Convention metadata is required on the definition/lambda and its
callable type. Shared conversion rewrites definitions and all direct/indirect
call sites together to the existing `CallableAbi`; the complete rule is in
[cps-effects.md](cps-effects.md#pre-cps-callable-shape).

The minimal operation set is:

| Operation | Purpose |
| ---- | ---- |
| `tribute_control.perform` | Invoke one source `fn` or general `op` in direct style, retaining its semantic kind |
| `tribute_control.handle` | Delimit a direct-style computation, its completion arm, and its handler table |
| `tribute_control.handler` | Describe one `fn` or general `op` handler arm inside a handle |
| `tribute_control.resume` | Consume the affine resumption bound by a resumptive general handler arm |
| `tribute_control.yield` | Terminate one executable `tribute_control` region with its logical value |

The dialect also owns the opaque type
`tribute_control.resume_token<input, answer>`. `input` is the value accepted
by the suspended operation's continuation, and `answer` is the logical result
of running that continuation through the enclosing handle. The type is not a
source type, callable ABI, backend carrier, or permission to inspect a
continuation representation. A source general operation whose logical result
is `Never` uses the canonical `core.never` TypeRef, creates no resumption, and
exposes no `resume_token`. The current pre-M2 frontend's `anyref` placeholder
for `Never` is not valid at this new boundary because a verifier must identify
the non-resumptive case without guessing from an erased type.

#### `tribute_control.perform`

```text
%result = tribute_control.perform %arg0, ... {
  ability_ref = !State,
  op_name = @get,
  operation_kind = @op
} : ResultType
```

- **Operands:** zero or more already-evaluated source arguments in declaration
  order. Values retain their logical types; tuple packing and erasure are
  conversion work.
- **Results:** exactly one logical operation result. A source `Never` result is
  `core.never`; it does not select a physical `Never` control carrier.
- **Attributes:** required `ability_ref: Type`, `op_name: Symbol`, and
  `operation_kind: Symbol`. `operation_kind` is exactly `fn` or `op` and is
  copied from the typechecked operation declaration. It is source-semantic
  metadata, not a lowering hint inferred from the body or use site.
- **Regions and block arguments:** none.
- **Terminator:** none; this is not a terminator in direct-style IR.
- **Semantics:** invokes the source operation with its declared kind. For
  `operation_kind = @fn`, the selected handler result automatically resumes
  direct-style evaluation; shared conversion does not capture a
  continuation and uses the existing tail-dispatch path. For
  `operation_kind = @op`, a matching handler may resume, in which case
  execution continues after the operation and the result becomes `%result`.
  If it does not resume, the selected general handler completes the matching
  handle and the apparent suffix is not evaluated. A source `op -> Never`
  cannot resume and therefore never constructs a logical resumption or
  captures the apparent suffix.

| `operation_kind` | Direct-style result | Required shared lowering | Matching handler shape |
| ---- | ---- | ---- | ---- |
| `@fn` | Declared source result | No suffix capture; `ability.call` then tail dispatch | No resume token; yield the declared operation result for automatic resumption |
| `@op` | Declared source result, using canonical `core.never` for `Never` | Capture the suffix for `ability.perform` and CPS dispatch, except use the reject adapter without suffix capture for `Never` | Final resume token except for `Never`; yield the handle answer on non-resumption |

- **Local verification:** requires the three attributes, checks the
  `operation_kind` domain, requires exactly one result and no regions, and
  requires resolved operand/result types rather than inference variables.
  Symbol-aware frontend conformance additionally checks that `ability_ref` and
  `op_name` resolve to an operation declaration whose declared `fn`/`op` kind,
  parameter types, and result type equal the attributes, operands, and result.
  Neither verifier may infer the kind from control flow, handlers, result type,
  or calling convention. The `tribute_control.handler` and containing-handle
  verifiers apply the corresponding handler row above to handler entries.
- **Ownership and value flow:** operands are ordinary SSA uses. The operation
  creates no source-visible continuation value; shared CPS conversion owns
  continuation construction. For `@fn`, conversion produces
  `ability.call`/tail dispatch without a continuation. For `@op`, conversion
  produces `ability.perform`/CPS dispatch with the suffix continuation. For
  `op -> Never`, it instead supplies the existing ability/effect ABI with a
  real zero-capture reject continuation whose body is `func.unreachable`; it
  does not capture the source suffix. Null, an in-band sentinel, or arbitrary
  `anyref` is not a continuation.
- **Location:** the source location of the ability-operation call, covering
  the qualified callee and arguments when available.

Every source ability invocation uses `tribute_control.perform` at this
boundary. The shared conversion is the first phase that chooses a dispatch
representation, using `operation_kind` without reclassifying it. In
particular, an `@op` whose handler body appears always tail-resumptive remains
an `@op`; recognizing and optimizing that shape is a later IR optimization
after semantic lowering.

The reject continuation is compatibility glue for the existing
`ability.perform` and `effect.dispatch_cps` operand contract, both of which
require an explicit continuation. It has the same callable ABI as an ordinary
lowered continuation, contains no captures, and traps if invoked. It may be
deduplicated per compilation unit, but each use passes a typed closure value
through the normal closure-to-`anyref` conversion. This rule preserves source
`Never` semantics without adding an operation or choosing the physical CPS
result carrier.

#### `tribute_control.handle`

```text
%answer = tribute_control.handle : AnswerType
  body {
    ...
    tribute_control.yield %body_value
  }
  completion(%completed: BodyType) {
    ...
    tribute_control.yield %answer_value
  }
  handlers {
    tribute_control.handler ... { ... }
    ...
  }
```

- **Operands:** none. Values captured by executable regions use normal
  enclosing SSA visibility.
- **Results:** exactly one logical handle result.
- **Attributes:** none are required. Dynamic prompt/owner tags and backend
  carrier choices are deliberately absent.
- **Regions:** exactly three, in the fixed order `body`, `completion`,
  `handlers`.
- **Block arguments:** `body` has one block and no arguments. `completion` has
  one block with exactly one argument, whose type equals the value yielded by
  `body`. `handlers` has one block with no arguments and contains only
  `tribute_control.handler` entries.
- **Terminators:** `body` and `completion` end in
  `tribute_control.yield`. The `handlers` block is a declarative table and has
  no terminator.
- **Semantics:** normal completion of `body` evaluates `completion` exactly
  once and returns its value. A general handler arm that finishes without
  resuming returns its arm value as the handle result and bypasses
  `completion`. A tail-resumptive `fn` arm returns an operation result that is
  fed automatically to the suspended computation.
- **Local verification:** enforces the fixed region count, single-block
  shapes, block-argument counts, terminators, yielded type equalities, and that
  every direct child of `handlers` is a unique
  `(ability_ref, op_name)` `tribute_control.handler`. Its result type must
  equal the completion yield type and every general handler's answer type.
- **Ownership and value flow:** the operation owns the delimited resumption
  capabilities created when its body performs resumptive general operations.
  They are exposed only as `resume_token` block arguments of resumptive
  general handler entries. Values leave executable regions only through
  `tribute_control.yield`.
- **Location:** the complete source `handle` expression. Region/block
  locations use the corresponding body, completion arm, and handler-list
  spans.

The frontend always materializes a completion region. When source omits a
`do` arm, the region is the identity operation on the body result. This removes
an optional structural case from conversion without changing source
semantics.

#### `tribute_control.handler`

```text
tribute_control.handler {
  ability_ref = !State,
  op_name = @get,
  kind = @op,
  operation_result_type = ResultType
} (%arg0: Arg0Type, ..., %resume:
    tribute_control.resume_token<ResultType, AnswerType>) {
  ...
  tribute_control.yield %answer
}
```

- **Operands and results:** none. It is a declarative entry owned by the
  surrounding `tribute_control.handle`.
- **Attributes:** required `ability_ref: Type`, `op_name: Symbol`,
  `kind: Symbol`, and `operation_result_type: Type`. `kind` is exactly `fn` or
  `op`.
- **Regions:** exactly one executable `body` region with one block.
- **Block arguments:** source operation arguments appear first, in declaration
  order and at logical types. A resumptive `op` entry has one final
  `resume_token<operation_result_type, handle-result-type>` argument. An `op`
  whose `operation_result_type` is source `Never` has no token, and a `fn`
  entry has no token.
- **Terminator:** the body ends in `tribute_control.yield`. For `fn`, the
  yielded type equals `operation_result_type` and is automatically resumed.
  For `op`, the yielded type equals the token's `answer` type and is the
  enclosing handle result when the arm completes without transferring control
  through `resume`.
- **Local verification:** checks required attributes and domains, one block,
  the final terminator, token position/parameters, and the yield rules above.
  A general handler whose `operation_result_type` is `Never` must have no
  token argument and no `tribute_control.resume` anywhere in its body,
  including nested regions.
  Parent placement, uniqueness, and equality with the enclosing handle result
  are checked by the local verifier of the containing handle. Symbol-aware
  frontend conformance also checks that the referenced declaration has the
  same `kind`, argument types, and `operation_result_type`; no verifier
  reclassifies a general `op` from its body shape.
- **Ownership and value flow:** when present, the final token argument is
  affine. It may be unused, meaning the continuation is dropped, or have one
  static ownership path to a `tribute_control.resume`. Capturing it in a
  closure transfers that static path to the closure; copying, storing,
  returning, yielding, or otherwise escaping it is invalid. Static SSA
  validation cannot prove that a captured closure is dynamically invoked only
  once, so the lowered resumption must also enforce one-shot consumption at
  runtime and reject or trap a second invocation. A `fn` arm and an
  `op -> Never` arm have no continuation capability.
- **Location:** the source handler arm, including its operation header.

#### `tribute_control.resume`

```text
%answer = tribute_control.resume %resume, %value : AnswerType
```

- **Operands:** exactly two. `%resume` has
  `resume_token<InputType, AnswerType>` and `%value` has `InputType`.
- **Results:** exactly one `AnswerType`.
- **Attributes and regions:** none.
- **Terminator:** none. Strict work after `resume` remains explicit in the
  enclosing region and executes only after the resumed computation returns.
- **Semantics:** consumes the nearest lexically enclosing general handler's
  one-shot resumption, supplies `%value` to the suspended `perform`, and
  returns the logical result obtained when that resumed computation reaches
  the handle boundary. It is invalid in a `fn` or `op -> Never` arm.
- **Local verification:** enforces operand/result arity and the three type
  equalities implied by `resume_token<InputType, AnswerType>`.
- **Ownership and value flow:** consumes the token. The token may reach this
  operation through explicit closure capture, but must retain a single static
  use-def path from its handler block argument. Affine-use validation is a
  whole-IR check because it follows captures and nested regions. If capture
  makes repeated dynamic invocation possible, the converted continuation's
  runtime one-shot state is the final enforcement boundary.
- **Location:** the source `resume` expression.

#### `tribute_control.yield`

```text
tribute_control.yield %value
```

- **Operands:** exactly one logical value.
- **Results, attributes, and regions:** none.
- **Terminator:** this operation is the terminator of a `handle` body,
  completion region, or handler body. It is invalid elsewhere.
- **Local verification:** enforces its own shape. The owning operation verifies
  placement and the yielded type.
- **Ownership and value flow:** transfers an ordinary logical value to the
  owning structured operation. A `resume_token` may never be yielded.
- **Location:** the source expression producing the region result, or the
  owning `handle` location for a synthesized identity completion.

#### Structured continuation invariant

Frontend output is in strict ANF inside every executable region. Strict
children are evaluated once, left to right. A selected case/conditional arm,
case guard, or short-circuit right-hand side remains inside its selected
`scf.*` region and is not hoisted. Handler bodies and nested handle bodies are
independent executable regions.

Shared CPS conversion lowers a region with an explicit logical continuation
for its remaining operations and its enclosing region exits:

1. The continuation at an operation includes the strict suffix in its current
   block.
2. At a case, conditional, or short-circuit operation, each branch receives a
   continuation that first reaches that structured operation's merge and then
   the enclosing suffix. Only the selected branch evaluates.
3. A handle body receives a delimiter continuation. Normal body completion
   enters `completion` and then the enclosing continuation.
4. An `operation_kind = @fn` perform does not capture a continuation. Shared
   conversion dispatches it through the tail path, and the automatically
   resumed operation result flows into the ordinary remaining block suffix.
5. A resumptive general handler's resume token denotes the suspended body
   continuation. `tribute_control.resume` invokes it and then continues with
   the arm-local strict suffix. If the arm never resumes, its yield completes
   the matching handle directly and bypasses the suspended suffix and
   completion region. A source `op -> Never` arm receives no token and can
   only take this non-resuming path.
6. A nested handle installs its own delimiter. A perform is handled by the
   nearest dynamically installed matching handler; resuming re-enters every
   selected structured frame between the perform and that handler. Non-resume
   completion abandons those frames.

This single region/suffix rule covers case arms and guards, conditionals,
short-circuit right-hand sides, nested handle bodies and arms, resume paths,
and strict work enclosing all of them. No AST containment scan or
construct-specific continuation convention is part of the dialect contract.

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

`list.*` represents the opaque canonical `List(a)` sequence contract. M1 uses
`list.empty`, `list.prepend`, `list.is_empty`, `list.head`, and `list.tail`.
These operations carry element/result types but no variant tags, node field
indices, allocation sizes, or target layout metadata. `list.prepend` is
semantically persistent: it returns a new sequence and does not mutate its tail.
Shared lowering may build a literal by first evaluating all elements left to
right and then applying `list.prepend` in reverse value order.
`list.head` and `list.tail` are internal observation operations with a
non-empty input precondition. Compiler-generated uses must establish
non-emptiness before executing either operation. A backend must trap if the
precondition is violated; it must not return a type-default head, a null tail,
or any other fallback value.
The public `List::prepend(value, tail)` prelude wrapper delegates to a private
ABI-marked compiler intrinsic, whose calls lower to the same `list.prepend`
operation. A source-defined function merely spelled `List::prepend` remains an
ordinary call. The private intrinsic ABI is a compiler/prelude boundary, not an
additional public symbol or a layout contract.

List patterns lower to sequence observations. Exact-length patterns require an
empty remainder; prefix-rest patterns return the remainder as the same canonical
List type. A backend must eliminate `list.*` before its backend-ready boundary.

The root `core.module` carries Tribute-specific well-known type identities as
`TypeRef` attributes. In particular, `tribute.type.string` is the exact
`adt.enum` type produced from the prelude `String` declaration. String-literal
lowering must consume this identity directly; it must not rediscover `String`
by type name, variant names, or field layout. This metadata is semantic compiler
state rather than a nominal-layout convention. The frontend preserves the
prelude declaration's stable identity through type checking and compares that
identity, not its name, when materializing this `TypeRef`.

The current specialized textual printer for `core.module` does not serialize
arbitrary module attributes. Consequently, parsing printed IR conservatively
drops well-known type metadata. A backend may still process byte constants, but
must reject `adt.string_const` when `tribute.type.string` is absent rather
than scanning types for a plausible replacement. A future textual format may
make these attributes round-trip explicitly.

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

`wasm_gc.*` is a typed intermediate dialect for WasmGC lowering. Its operations
carry semantic heap types as mandatory `TypeRef` attributes and must not infer
nominal identity from an erased operand such as `anyref`. A module-wide type
layout pass assigns binary type-section indices and fully converts these
operations to `wasm.*` operations, whose required integer attributes correspond
to WebAssembly instruction immediates. This pass runs once, after unrealized
conversion casts have been materialized, because materialization may introduce
additional typed GC operations.

```text
wasm_gc.struct_get { type = !String$Leaf, field_idx = 0 }
  -- module GC type layout -->
wasm.struct_get { type_idx = 9, field_idx = 0 }
```

Builtin layouts follow the same rule. Lowering refers to canonical semantic
types such as `core.bytes` or the marker/evidence ADT types; the layout pass
maps those identities to reserved indices. The Wasm emitter accepts no residual
`wasm_gc.*` operations and does not infer missing indices. Function-signature
indices used by `call_indirect` are a separate concern and are not GC heap-type
identities.

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
  -> AST-to-IR (source-logical callable + tribute_control IR)
  -> shared CPS legalization
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
| AST-to-IR | Source-logical callable signatures, verified `tribute_control` structure, and valid SSA use chains |
| Shared CPS legalization | Callable signatures and all call sites use the physical `CallableAbi`; no `tribute_control.*` remains |
| Shared lowering | High-level ability dispatch operations are removed at their claimed boundaries |
| Effect ABI | `effect.*` operations preserve dispatch semantics without backend layout details |
| Backend lowering | Backend-ready target verification succeeds and no `effect.*` operations remain |

## Type Model

Primitive scalar, pointer, reference, bytes, array, tuple, function, and nil
types are represented in TrunkIR. Library data types such as `Option`, `Result`,
and `Text` lower through ADT and runtime/library conventions. `List` is an
opaque nominal builtin whose shared construction and sequence-view observations
use `list.*`; target-specific passes choose and eliminate its private
representation.

## Open Questions

- Final closure environment representation for each backend.
- Reusable conversion targets for effect ABI lowering boundaries.
- Debug/source-map representation.
- WasmGC reactivation strategy for lowering shared tail-call CPS IR into
  closure, table, and evidence representations.
