# Reference Counting (Native Backend)

> This document defines the RC memory management strategy for the Cranelift
> native backend. The WASM backend uses WasmGC and is not affected.
>
> See also: [cranelift-backend.md](cranelift-backend.md),
> [implementation.md](implementation.md)

## Overview

The native backend uses **reference counting** for heap-allocated objects
(structs, enums, arrays, boxed primitives). Key principles:

- **No runtime library** — all RC logic is compiler-generated code
- **libc only** — depends solely on `malloc`/`free` (via allocator indirection)
- **Dialect-based** — RC operations are `tribute_rt.retain`/`tribute_rt.release`
  dialect ops, lowered to inline code
- **Typed planning** — ownership and RTTI decisions precede type erasure

## Allocator Interface

All heap allocation goes through two indirection symbols:

```text
__tribute_alloc(size: i64) -> ptr
__tribute_dealloc(ptr: ptr, size: i64)
```

### Default Implementation

The compiler generates default implementations as simple `malloc`/`free`
wrappers. These are declared with `Import` linkage so they can be overridden
at link time (e.g., with a custom allocator via weak symbols).

### Alloc Sequence (compiler-generated inline)

```text
raw_ptr = call @__tribute_alloc(size + 8)    // include header
store refcount=1       at raw_ptr
store rtti_idx=<type>  at raw_ptr + 4
obj_ptr = raw_ptr + 8                        // caller sees offset 0
```

### Free Sequence (compiler-generated inline)

```text
raw_ptr = obj_ptr - 8
call @__tribute_dealloc(raw_ptr, size + 8)
```

### Symbol Convention

Internal/runtime symbols use the `__tribute_` prefix to avoid collisions
with user code and to clearly mark compiler-generated functions.

---

## Memory Layout

### Object Header

Every heap-allocated RC object has an 8-byte header prepended before the
payload. Compiled code always sees the pointer at offset 0 (first field);
header access uses `ptr - 8`.

```text
[-8] refcount: u32   — reference count (1 on allocation)
[-4] rtti_idx: u32   — runtime type info index
[ 0] payload...      — first field (naturally aligned)
```

### Struct Layout

Structs are laid out with fields in declaration order, naturally aligned:

```text
Struct: [fields in order, naturally aligned]
Enum:   [tag: i32] [padding] [payload: max(variant sizes)]
Array:  [length: i64] [elements...]
```

Field offsets are computed by `adt_layout.rs` at compile time.

### Boxed Primitives

Boxed primitives are the simplest heap objects — just the raw value:

| Type | Payload Size | Layout |
| ---- | ----------- | ------ |
| boxed i32 (Int/Nat/Bool) | 4 bytes | `[i32 value]` |
| boxed f64 (Float) | 8 bytes | `[f64 value]` |

### Private native List nodes

The native `List(a)` representation uses immutable RRB nodes with the ordinary
RC object header. Internal nodes own their child references, leaves own their
reference-typed elements, and the root owns the reachable tree. Sequence views
may borrow nodes transiently, but a view that escapes or outlives the original
root must retain the referenced structure under the normal ownership rules.
Branching factor, node packing, and the empty representation are target-private
and are not visible as source constructors or shared-IR field indices.

---

## RC Operations

The `tribute_rt` dialect provides two RC operations:

```text
tribute_rt.retain(ptr) -> ptr    // refcount++, return same pointer
tribute_rt.release(ptr)          // refcount--, free if zero
```

These are dialect-level operations that will be:

1. **Inserted** by the RC insertion pass (SSA-based liveness analysis)
2. **Lowered** to inline code by the RC lowering pass

Type erasure 전 RC planning은 검증된 모든 `adt.typeref`를 semantic type에 따라
managed로 취급한다. `core.ptr`는 항상 unmanaged이며 cast, symbol, ABI spelling,
operand position 또는 pointer provenance로 managed 성질을 얻을 수 없다.
`adt.ref_null`은 nominal managed reference type의 null 값이고, 생성된 retain과
release는 이 값에 대해 no-op이어야 한다. Callable과 managed-reference validation은
ownership plan 또는 IR mutation이 이 분류를 사용하기 전에 완료된다.

Type erasure가 managed reference를 `core.ptr`로 바꾼 뒤에는 pointer type 자체가
ownership을 증명하지 않는다. Pre-erasure ownership plan이 물리 IR에 명시적으로
materialize한 `tribute_rt.retain`과 `tribute_rt.release`만 RC 의미를 보존한다. 따라서
이 operation의 `core.ptr` operand가 RC allocation을 가리킬 수 있다는 사실과
`core.ptr` 자체가 unmanaged라는 규칙은 모순되지 않는다.

### Inline Lowering

```text
// tribute_rt.retain(ptr):
if ptr == null:
    return ptr
refcount = load(ptr - 8)
refcount = refcount + 1
store(refcount, ptr - 8)

// tribute_rt.release(ptr):
if ptr == null:
    return
refcount = load(ptr - 8)
refcount = refcount - 1
store(refcount, ptr - 8)
if refcount == 0:
    call @__tribute_dealloc(ptr - 8, size + 8)
```

The abbreviated sequence above omits type-specific field destruction. Every
complete retain and release path returns for a null managed reference before
accessing its header, refcount, or RTTI. The complete release path uses RTTI
dispatch as described below.

---

## Boxing / Unboxing

Boxing converts unboxed primitives to heap-allocated pointers for use in
polymorphic contexts (e.g., passing `Int` where `any` is expected).

### Implementation

Boxing and unboxing are handled at two levels:

1. **Explicit ops** (`tribute_rt.box_int`, `tribute_rt.unbox_int`, etc.)
   — generated by the `insert_boxing` pass, lowered by
   `tribute_rt_to_clif` to `clif.*` allocation + store/load.

2. **Implicit casts** (`unrealized_conversion_cast(i32 → ptr)`)
   — resolved by materializations in the native type converter.

Both paths generate equivalent code:

```text
// Boxing (e.g., i32 → ptr):
%size = clif.iconst(4)
%ptr  = clif.call @__tribute_alloc(%size)
clif.store(%value, %ptr, offset=0)
// result: %ptr

// Unboxing (e.g., ptr → i32):
%value = clif.load(%ptr, offset=0)
```

### Comparison with WASM Backend

| Aspect | WASM | Native |
| ------ | ---- | ------ |
| Int/Nat/Bool boxing | `wasm.ref_i31` (i31ref) | heap alloc + store |
| Float boxing | `wasm.struct_new` (BoxedF64) | heap alloc + store |
| Int/Nat unboxing | `wasm.i31_get_s/u` | `clif.load` |
| Float unboxing | `wasm.struct_get` | `clif.load` |
| Representation | GC-managed refs | raw pointers |

---

## RC Pipeline

The RC implementation is divided into three passes:

### 1. RC Insertion Pass (PR 3)

**Location:** `tribute-passes/src/native/rc_insertion.rs`

**Purpose:** Insert `tribute_rt.retain` and `tribute_rt.release` operations
based on SSA liveness analysis.

**Algorithm:**

1. **Liveness analysis** — determine last use of each SSA value
2. **Ownership rules:**
   - Function parameters: `retain` at entry block
   - Return values: ownership transfer (no retain/release)
   - Local allocations: refcount=1 at creation, `release` at last use
   - Struct field loads: `retain` after load, `release` at last use
   - Struct field stores: `release` old value before store
   - Branch targets: `retain` for each successor path

**Pointer detection:** Use `core::Ptr::from_type()` to identify RC-managed values.

### 2. RC Optimization Pass

**Location:** `tribute-passes/src/native/rc_optimization.rs`

**Purpose:** Eliminate redundant retain/release pairs before they are expanded
into control flow and atomic operations by RC lowering.

**Optimizations:**

- **Paired elimination:** Within one basic block, remove a `retain` followed by
  a matching `release` when every intervening use is proven not to let the
  reference escape. The release operand may be either the original pointer
  passed to `retain` or the `retain` result. The initial safe-use whitelist is
  deliberately narrow: loads through the reference and stores that use it only
  as the destination address. Storing the reference as a value, passing it to
  a call or branch, crossing an unknown operation, entering a nested region,
  or encountering an alias/cast prevents elimination. If the `retain` result
  is used, its uses are replaced with the original pointer before erasing the
  pair. This optimization does not cross basic-block boundaries or chase
  aliases.
- **Borrowed parameter elision:** Before RC insertion, classify an `anyref`
  function parameter as borrowed only when every use is proven to remain within
  the dynamic extent of the call. Loads through the parameter, comparisons,
  and stores that use it only as the destination address are borrowed uses.
  A chain of unrealized conversion casts is transparent only when every use of
  its result is itself proven borrowed. Returning or storing the parameter as a
  value, passing it to a call or branch, using it through any other alias or a
  nested region, and every unknown operation are escapes. Closure, ability
  handler, and continuation capture therefore preserve owned-parameter RC.
  Analysis failure preserves the existing owned-parameter RC.
  Calls remain escape barriers unless the call is direct and its callee has a
  trusted ownership summary proving the corresponding parameter borrowed.
  Indirect, external, unresolved, or otherwise unknown calls are always escape
  barriers.
  The callee must also have the ordinary synchronous caller-lifetime guarantee:
  C ABI entry points, direct tail-call targets, and functions whose address
  escapes are ineligible because their caller may not retain an owning frame
  for the full invocation.
  For a proven borrowed parameter, RC insertion omits both the entry `retain`
  and every parameter `release`; this keeps acquisition and release decisions
  under one ownership proof instead of matching generated releases afterward.
- **Temporary field borrows:** RC insertion may omit ownership acquisition and
  release for an `anyref` result of `clif.load` when the load is proven to be a
  field-derived temporary borrow. The analysis runs on lowered Clif, where an
  `adt.struct_get` is represented by `clif.load`; it does not match
  `adt.struct_get` directly.

  The load address must resolve directly to an RC-managed owner, optionally
  through transparent unrealized pointer casts and address calculations. The
  resolved owner must be an RC-managed value tracked by RC liveness; a raw
  `core.ptr` derived from `__tribute_alloc` is not sufficient ownership proof.
  The owner definition must dominate the field load, the field load must
  dominate every use of its result, and every result use must remain in the
  same function region. The only initially accepted uses are further field
  loads through the temporary, pointer comparisons, and stores that use it
  solely as the destination address. Unrealized pointer casts are transparent
  only when every use of the cast result satisfies these same rules. Returning
  or storing the temporary or an alias as a value, passing either to any call
  or branch, forwarding either as a block argument, capture by a nested region,
  or any unknown operation prevents elision.

  The owner's lifetime must also be extended through every accepted use of the
  temporary. RC liveness therefore treats those uses as uses of the owner before
  insertion. A temporary is rejected when its uses occur in sibling dominator
  subtrees, after a join not dominated by its load, on a loop-carried path, or
  anywhere else that does not establish one dominated, non-escaping lifetime.
  Nested field loads are considered independently and are eligible only when
  each loaded owner's lifetime is proven by the same rules. Missing CFG edges,
  unreachable blocks, malformed regions, or any analysis uncertainty preserve
  the original retain/release operations.
- **Constant propagation (planned):** Elide RC for compile-time-known lifetimes

The paired-elimination, borrowed-parameter, and temporary-borrow policies are
selected independently by the native pipeline options, not stored in an IR
lowering context. Production enables proven optimizations; the baseline profile
disables them for conformance comparisons.

**Pipeline position:** Ownership summaries are computed before `func_to_clif`.
Borrowed-parameter and temporary-field-borrow analysis run as independent parts
of RC insertion before their respective RC operations are created. Temporary
borrow lifetime dependencies extend owner liveness before insertion; parameter
summary validation does not replace or bypass that dominance/lifetime analysis.
Paired elimination runs immediately after RC insertion. All three decisions
occur before unrealized cast resolution and RC lowering, which keeps alias
handling conservative and makes the inserted and optimized RC boundaries
directly observable in tests.

#### Proper-tail ownership transfer

Native RC distinguishes a callable's parameter-entry contract from the action
at one call site. The existing `borrowed`/`owned` summary remains a borrow
optimization and does not encode proper-tail transfer.

Each RC-managed physical parameter has one exact entry mode:

- **borrowed:** the callee receives no ownership unit, performs no entry retain,
  and must neither release nor transfer the parameter;
- **retained:** the callee acquires its own unit with an entry retain and must
  release or return that unit; this is the existing ordinary owned-parameter
  behavior; or
- **consumed:** the caller supplies one ownership unit, the callee performs no
  entry retain, and the callee must eventually release, return, or proper-tail
  transfer that unit.

Physically empty CPS callables use `consumed` for every parameter whose exact
native conversion is RC-managed. Non-RC parameters have no RC action. This is
a native callable contract, not a conclusion inferred by RC insertion from a
name, operand type, body shape, or calling-convention integer alone.

An ordinary call to a consumed parameter acquires a new unit with `retain`
immediately before the call, leaving the caller's existing unit live. A
non-returning proper-tail call transfers the caller's existing unit without a
caller-side release. When the caller only borrows the value, it first retains
once to create the transferred unit. If the same underlying value is supplied
to `N` consumed parameters, the edge supplies `N` units: an owned value
transfers one existing unit and retains `N - 1`, while a borrowed value retains
`N`. Alias-transparent casts count as the same underlying ownership unit.

Every RC-managed proper-tail operand must target a consumed parameter. A tail
edge to a borrowed parameter would outlive the caller dynamic extent; a tail
edge to a retained parameter would require cleanup after the terminator. Both
are rejected. Dying RC values not transferred by the edge are released before
the tail terminator. No RC operation may follow `clif.return_call` or
`clif.return_call_indirect`.

The native ownership producer runs after target-ABI validation and immediately
before `func_to_clif`. It records versioned, positional parameter-entry and
call-edge actions plus a fresh contract identity, and returns an opaque
in-memory trust token. Direct edges must resolve one exact module-local symbol.
Indirect edges additionally require the exact physical callable signature and
explicit CPS provenance already carried by the transfer. `func_to_clif`
preserves this metadata; RC insertion cross-checks it against the trust token
after lowering. Textual metadata is never trusted.

RC insertion validates the complete module before applying any mutation. Once
that preflight succeeds, insertion planning has no remaining failure path.
Missing, malformed, duplicate, external, stale, or inconsistent contracts; an
indirect signature mismatch; and a tail operation that is not the final block
operation all fail before mutation. The contract metadata is removed after
successful consumption and does not become part of the emitted ABI.

#### Trusted ownership summaries across `func_to_clif`

Borrowed forwarding uses explicit pre-lowering metadata rather than rebuilding
a call graph after `func` operations have been erased. Immediately before
`func_to_clif`, the native pipeline computes a module-local summary for every
defined function and stores it on `func.func` as the versioned
`tribute.rc.parameter_ownership_v1` attribute. `func_to_clif` preserves this
attribute unchanged on the corresponding `clif.func`; RC insertion is its only
consumer. The producer also returns an opaque in-memory trust token containing
the expected summaries. RC insertion requires that token and cross-checks every
attribute against it, so textual metadata alone is never trusted.

The attribute is a list with exactly one entry per function parameter. Each
entry is the symbol `borrowed` or `owned`. A summary is trusted only when all of
the following hold:

- it was recomputed by the current native pipeline invocation, not merely found
  on input IR;
- its version, shape, and parameter count are exact;
- its function symbol resolves uniquely to a module-local definition; and
- the summarized parameter is still `tribute_rt.anyref` at RC insertion.

Missing, malformed, stale, duplicate, or inconsistent metadata is ignored as a
whole for that function. Ignored summaries mean every parameter is owned. This
fail-closed rule also applies if lowering drops or changes the metadata.

Summary computation is a monotone fixed point over the direct-call graph.
Every `anyref` parameter starts `borrowed` and is demoted permanently to `owned`
when a use escapes. Loads, pointer comparisons, destination-address stores, and
transparent unrealized casts are local borrowed uses. Forwarding to parameter
`i` of a uniquely resolved direct callee is borrowed only when that callee's
entry `i` is borrowed. Return/tail calls, storing as a value, nested-region
capture, indirect calls, external calls, unresolved calls, and unknown
operations demote the parameter.

Strongly connected components are solved to a fixed point, but recursive SCCs
are not trusted for borrowed forwarding: all parameters participating in a
direct or mutual recursive cycle are owned. This avoids circular proofs whose
only evidence is the cycle itself. Acyclic direct-call chains can therefore
propagate borrowed parameters transitively while recursion stays conservative.

RC insertion consumes only summaries produced and validated by the current
pipeline run. Its local parameter-use analysis still treats every call without
a trusted borrowed callee entry as an unknown-call barrier. Summary validation
and temporary-field-borrow analysis are independently selectable and compose:
trusted forwarding may elide parameter ownership while dominance and lifetime
dependencies separately govern field-derived temporaries.

### 3. RC Lowering Pass (PR 4)

**Location:** `tribute-passes/src/native/rc_lowering.rs`

**Purpose:** Lower `tribute_rt.retain` and `tribute_rt.release` to inline
`clif.*` operations.

**Lowering patterns:**

```text
tribute_rt.retain(ptr) ->
    if ptr == null: return ptr
    %rc_addr = clif.iadd(ptr, clif.iconst(-8))
    %rc = clif.load(%rc_addr)
    %new_rc = clif.iadd(%rc, clif.iconst(1))
    clif.store(%new_rc, %rc_addr)
    // result: ptr (unchanged)

tribute_rt.release(ptr) ->
    if ptr == null: jump continue_block
    %rc_addr = clif.iadd(ptr, clif.iconst(-8))
    %rc = clif.load(%rc_addr)
    %new_rc = clif.isub(%rc, clif.iconst(1))
    clif.store(%new_rc, %rc_addr)
    %is_zero = clif.icmp(%new_rc, clif.iconst(0), cond="eq")
    clif.brif(%is_zero, then_dest=free_block, else_dest=continue_block)

free_block:
    %raw_ptr = clif.iadd(ptr, clif.iconst(-8))
    %size = clif.iconst(<object_size> + 8)
    clif.call(@__tribute_dealloc, %raw_ptr, %size)
    clif.jump(continue_block)

continue_block:
    // continue execution
```

**Pipeline position:** After `resolve_unrealized_casts`, before `emit_module_to_native`.

---

## Ownership and Lowering Order

RC lowering follows a semantic-to-physical order:

1. Validate callable origins and managed-reference boundaries on typed IR.
2. Compute ownership actions and RTTI field information while `adt.typeref`
   identity is still available.
3. Materialize explicit `tribute_rt.retain` and `tribute_rt.release` operations.
4. Erase managed references to `core.ptr` and lower the explicit RC operations
   to physical refcount updates and type-specific destruction.

No later pass may reconstruct managedness from a raw pointer, symbol spelling,
ABI marker, operand position, or erased provenance. A shallow release may be
used only as an explicitly documented intermediate implementation stage; the
semantic contract requires type-specific release of owned managed fields.

### Type-specific release

The compiler generates a release function for each managed aggregate type. When
the refcount reaches zero, RTTI dispatch selects that function, which releases
owned managed fields before deallocating the object.

**Example (struct with pointer field):**

```text
struct Point { x: Int, y: Ref<Node> }

// Compiler generates:
__tribute_release_Point(ptr):
    %y_addr = clif.iadd(ptr, clif.iconst(4))  // field offset
    %y_val = clif.load(%y_addr)
    tribute_rt.release(%y_val)                 // recursive release
    %raw = clif.iadd(ptr, clif.iconst(-8))
    call @__tribute_dealloc(%raw, clif.iconst(16))
```

**RTTI dispatch:**

```text
tribute_rt.release(ptr) ->
    if ptr == null:
        return
    %rc = decrement_refcount(ptr)
    if %rc == 0:
        %rtti_idx = load(ptr - 4)
        %release_fn = RTTI_TABLE[%rtti_idx].release_fn
        call %release_fn(ptr)
```

### Continuation ownership

Tail-call CPS represents a continuation as a typed closure with an explicit
ContinuationFrame. Capturing a managed value into that frame creates an
independent owned reference and therefore materializes a retain. Destroying an
unresumed one-shot continuation releases every owned frame field through the
ordinary type-specific destructor. Resuming transfers the frame-owned values
according to the continuation callable contract.

There is no stack-copying continuation runtime, TLS root buffer, `mp_yield`, or
special double-release protocol. Proper-tail lowering emits releases for dying,
non-transferred values before `func.tail_call` or
`func.tail_call_indirect`; no RC operation may follow the tail terminator.

---

## RTTI Table

**Location (future):** Emitted as static data by `trunk-ir-cranelift-backend`

**Structure:**

```rust
struct TypeInfo {
    release_fn: extern "C" fn(*mut u8),  // Type-specific destructor
    size: u32,                            // Object size (excluding header)
    // Future: field_count, field_offsets, name, etc.
}

// Emitted as static data in each compiled module
static TRIBUTE_RTTI_TABLE: [TypeInfo; N] = [...];
```

**Index allocation:**

- Compile-time sequential assignment per module
- Reserved indices:
  - `0` = boxed i32 (Int)
  - `1` = boxed f64 (Float)
  - `2` = boxed i32 (Bool/Nat)
  - `3+` = user-defined structs/enums

`release` uses the stored RTTI index to select the type-specific destructor.

---

## Field Reordering

**Goal:** Minimize struct padding by reordering fields by alignment.

**Rules:**

- Compiler MAY reorder struct fields for optimal layout
- Original field order preserved in `field_offsets` mapping
- All access via `adt.struct_get(field_idx)` uses offset from mapping

**Example:**

```text
// Source:
struct Foo { a: i8, b: i64, c: i16 }

// Reordered layout (8-byte alignment):
[b: i64] [c: i16] [a: i8] [padding: 5 bytes]  // total: 16 bytes

// vs. original order:
[a: i8] [padding: 7] [b: i64] [c: i16] [padding: 6]  // total: 24 bytes

// field_offsets mapping:
field_offsets[0] = 9   // a at byte 9
field_offsets[1] = 0   // b at byte 0
field_offsets[2] = 8   // c at byte 8
```

**Future:** Add `@repr(c)` attribute to disable reordering for FFI compatibility.

---

## Testing Strategy

### Unit Tests

- RC insertion: Verify retain/release placement via hand-crafted IR
- RC lowering: Verify inline code generation (refcount ops, conditional free)
- Boxing: Verify allocation + store sequences

### Integration Tests

- E2E: Tribute source → native binary with RC
- Memory safety: Valgrind/AddressSanitizer (no leaks, no double-frees)
- Optimization conformance: compile and execute the same fixture with one RC
  optimization disabled and enabled, preserving exit status and output
- Before/after IR: snapshot the named boundary after RC insertion and before RC
  lowering, and assert the exact retain/release operations removed
- Conservative negatives: calls, stores, branch arguments, aliases, closure
  captures, handler captures, and continuation captures must block elision

RC optimization tests follow the shared validation contract in
`optimizations.md`. AddressSanitizer runs use the same sanitizer configuration
for both sides of the optimization comparison.

### Test Scenarios

- **Pointer parameters:** Retain at entry, release at last use
- **Struct fields:** Release old value on field update
- **Polymorphic boxing:** Int → any → Int round-trip
- **Control flow:** Retain for multiple successors
- **Cyclic references:** (Future) Detect and handle cycles

---

## Deferred Decisions

- **Cycle detection:** Weak references? Tracing GC fallback?
- **Thread-safety:** Atomic refcount for multi-threaded code?
- **FFI boundaries:** How to handle RC objects at C FFI boundaries?
- **Optimization:** Compile-time escape analysis to elide RC?
