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
[-4] rtti_idx: u32   — runtime type descriptor index
[ 0] payload...      — first field (naturally aligned)
```

### Struct Layout

Structs are laid out with fields in declaration order, naturally aligned:

```text
Struct: [fields in order, naturally aligned]
Enum:   [the variant's fields in order, naturally aligned], sized per variant
Array:  [length: i64] [elements...]
```

Field offsets are computed at compile time by the `adt` dialect's `layout`
module in `tribute-ir`.

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

`tribute_rt` dialect는 두 RC operation과 한 ownership 경계 operation을 제공한다.

```text
tribute_rt.retain(ptr) -> ptr    // refcount++, return same pointer
tribute_rt.release(ptr)          // refcount--, free if zero
tribute_rt.into_raw(value) -> core.ptr // typed ownership unit 하나를 소비
```

이 연산은 dialect 수준에서 다음 순서로 처리된다.

1. 네이티브 evidence lowering이 필요한 `into_raw`를 먼저 만든다.
2. 검증된 type-erasure 전 ownership plan은 `retain`/`release`만 **materialize**하고
   검증한 `into_raw`를 보존한다.
3. `into_raw`를 explicit native representation conversion으로 **lower**한다.
4. RC lowering pass가 retain/release를 inline code로 **lower**한다.

Type erasure 전 RC planning은 검증된 모든 `adt.typeref`를 semantic type에 따라
managed로 취급한다. `core.ptr`는 항상 unmanaged이며 cast, symbol, ABI spelling,
operand position 또는 pointer provenance로 managed 성질을 얻을 수 없다.
`adt.ref_null`은 nominal managed reference type의 null 값이고, 생성된 retain과
release는 이 값에 대해 no-op이어야 한다. Callable과 managed-reference validation은
ownership plan 또는 IR mutation이 이 분류를 사용하기 전에 완료된다.

`tribute_rt.into_raw`는 typed managed value의 정확히 하나인 ownership unit을 소비하고
항상 unmanaged `core.ptr`를 만든다. 이 operation은 generic call metadata, symbol,
ABI, operand 위치 또는 pointer provenance를 사용하지 않는다. Type erasure가 managed
reference를 `core.ptr`로 바꾼 뒤에는 pointer type 자체가 ownership을 증명하지 않는다.
Pre-erasure ownership plan이 물리 IR에 명시적으로 materialize한
`tribute_rt.retain`/`tribute_rt.release`와 검증한 `tribute_rt.into_raw`만 RC 의미를
보존한다. 따라서 이 operation의 `core.ptr` operand가 RC allocation을 가리킬 수 있다는 사실과
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

Boxing은 explicit operation(`tribute_rt.box_int`, `box_nat`, `box_bool`,
`box_float`)으로만 일어난다. 이 operation은 boxing materializer가
`generic_type_converter`에서 만들고, `tribute_rt_to_clif`가 할당, RC header,
payload store로 내린다. Header의 RTTI index는 operation이 아는 소스 타입의 예약
번호다.

Native type converter는 scalar를 boxing하는 materialization을 하지 않는다.
`core.i32`나 `core.f64`만으로는 `Int`, `Nat`, `Bool`을 구별할 수 없어 descriptor를
고를 수 없기 때문이다. Boxing이 필요한 implicit cast가 남으면 emission 경계가
거부한다. Unboxing은 의미가 같으므로 type converter가 payload load로 materialize할
수 있다.

```text
// tribute_rt.box_int:
%size = clif.iconst(12)
%ptr  = clif.call @__tribute_alloc(%size)
clif.store(1, %ptr, offset=0)          // refcount
clif.store(RTTI_INT, %ptr, offset=4)   // descriptor index
clif.store(%value, %ptr, offset=8)

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

The RC implementation is divided into four stages:

### 1. Typed ownership and RTTI plan

**Location:** `tribute-passes/src/native/ownership_plan.rs`

검증된 typed native IR에서 immutable ownership plan을 만든다. 이 단계는 IR을
변경하지 않으며 `func_to_clif`와 native type erasure보다 먼저 실행한다.

이 단계의 policy-neutral 사실 계산은 같은 경계의 fallible 분석이 소유한다.
Module 범위 분석이 function 정의, 검증된 managed nominal layout과
`adt.struct_set`이 쓰는 layout을, function 범위 분석이 flat CFG, managed 값,
exact alias root, managed projection-owner 관계, 그중 borrow할 수 있는
projection과 liveness 입력을 제공한다. 이 사실은 `NativeOwnershipPlanOptions`와
무관하며 borrow elision과 entry ownership 정책은 사실을 소비하는 planner와
action planner가 적용한다.

Block별 `defs`, `live_in`, `live_out`은 function ownership facts에 의존하는
`NativeManagedLiveness` 분석의 두 view가 제공한다. 보수적 view는 managed
value의 일반 use만 반영하고, owner-extended view는 borrow할 수 있는 managed
projection의 use를 owner의 use로도 반영한다. `elide_proven_field_borrows`만 view를
선택하며 borrowed-parameter 정책은 선택에 관여하지 않는다. 각 view의 고정점은
최초 질의 때만 계산되어 사용하지 않는 고정점 계산을 피하고, action planner는
선택된 결과를 그대로 소비한다. 런타임 정책 값은 분석 캐시의 identity에
참여하지 않는다. 선행 facts 분석 실패는 그대로 전파되고 실패한 liveness
결과는 캐시되지 않는다.

**Algorithm:**

1. Exact semantic type과 callable contract로 managed value, field와 parameter
   entry mode를 분류한다.
2. SSA liveness로 entry, call, owning store/copy, borrowed load, final-use,
   return과 proper-tail action을 정한다.
3. 같은 typed predicate로 aggregate별 RTTI managed-field bitmap을 만든다.
4. 모든 identity, signature, CFG, action target과 ordering을 검증한 뒤 하나의
   deterministic plan을 반환한다.

Block argument로의 전달은 source의 unit을 destination으로 옮긴다. 옮길 unit은
source마다 하나뿐이다. Source가 선택된 liveness view에서 branch 뒤에도 살아 있으면
그 unit은 source에 남고, 같은 branch가 한 source를 여러 destination에 넘기면 첫
destination만 unit을 받는다. Unit을 받지 못한 destination마다 branch 직전에 새
unit을 획득한다.

한 block 안에서 마지막으로 쓰이고 죽는 값은 그 사용 직후에, 사용이 terminator면 그
직전에 release한다. Predecessor의 끝에서 살아 있지만 successor의 시작에서 살아 있지
않은 값은 edge에서 죽으며 successor의 시작에서 release한다. 이 위치는 successor의
모든 predecessor가 그 값을 살려 둔 채 끝날 때만 올바르다. 함수 진입은 entry block의
predecessor이며 어떤 값도 갖고 있지 않다. 일부 predecessor만 값을 살려 두는 edge와,
값을 다시 정의하는 block으로 들어가는 edge에는 release를 둘 자리가 없으므로 plan
생성을 실패시킨다.
`scf_to_cf`는 multi-successor branch의 target마다 그 branch만 predecessor로 갖는
block을 만들므로 이런 edge를 만들지 않는다.

`adt.typeref`는 type 자체로 managed다. Native RC-header allocation을 표현하는
검증된 internal ADT/closure layout과 `tribute_rt.anyref`/`intref`도 각자의 typed
contract로 분류한다. Evidence, function/code address, borrowed buffer,
`core.ptr`, `core.bytes`, `core.array`는 unmanaged다. 변환 결과가 pointer라는
사실은 이 분류에 참여하지 않는다.

Residual structured region, stale nominal identity, malformed callable metadata,
duplicate/conflicting action 또는 SSA remapping ambiguity는 plan 생성 전체를
실패시킨다. 실패하기 전후의 input IR은 동일해야 한다.

Representation rewrite가 allocation layout의 `TypeRef`를 바꾸면 해당 pass는
exact source-target identity mapping을 반환한다. RTTI bitmap은 이 mapping으로만
이동하며 dialect, name 또는 layout shape로 target을 다시 찾지 않는다.

### 2. Explicit RC materialization

**Location:** `tribute-passes/src/native/rc_materialization.rs`

Typed plan의 action을 `tribute_rt.retain`과 `tribute_rt.release`로 materialize하고
검증된 `tribute_rt.into_raw`를 보존한 뒤 managed type을 physical type으로 바꾼다.
Materializer는 type이나 pointer
provenance에서 action을 새로 발견하지 않는다. Duplicate owning destination마다
별도 unit을 확보하고 proper-tail operand는 선택된 unit을 이전한다. 이전되지 않고
죽는 값은 terminator 앞에서 release하며 proper-tail terminator 뒤에는 RC operation을
둘 수 없다.

각 materialize된 release는 plan action이 가리키는 type-erasure 전 nominal layout에서
계산한 **payload size + RC header size**를 가진다. Replaced-field release도 field의
exact declared layout으로 같은 size를 사용한다. `tribute_rt.anyref`와 `intref`처럼
static nominal layout이 없는 opaque dynamic reference와, variant마다 할당 크기가
다른 enum 값의 `alloc_size = 0`은 deallocation size가 아니라 RTTI dispatch 전용
신호다. backend는 header RTTI로 이를
exact release function으로 해소한 뒤에만 deallocate하며, entry가 없으면 zero-size
deallocation 대신 fail-closed한다. physical def-chain에서 size를 추론하지 않는다.

Materialization은 먼저 plan, module identity, 모든 action anchor, exact
replacement-field layout/index, insertion schedule을 검증한다. 이 검증이 실패하면
module을 변경하지 않는다. Materialization은 검증된 typed plan만 소비하며, 지워진
pointer에서 liveness, alias 또는 소유권을 다시 추론하지 않는다.

### 3. RC Optimization Pass

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
- **Borrowed parameter elision:** Typed planning은 managed 매개변수의 모든 사용이
  호출의 동적 범위 안에 머문다고 증명될 때만 그 매개변수를 borrowed로 분류한다.
  Borrowed 매개변수는 다음 불변식을 따른다: callee가 매개변수를 쓸 수 있는 동안
  그 referent는 caller 또는 조상 frame의 owned root에서 계속 도달 가능하다.
  Ordinary direct call의 caller는 argument의 unit, 또는 argument가 파생된 root의
  unit을 호출이 반환할 때까지 유지하므로 이 불변식을 제공한다.

  Borrowed 사용은 다음뿐이다.

  - 매개변수를 읽기 대상으로 삼는 `adt.struct_get`, `adt.variant_get`,
    `adt.variant_is`, `adt.ref_is_null`.
  - `adt.ref_cast`와 unrealized conversion cast. Cast result의 모든 사용이 다시
    borrowed 사용일 때만 transparent하다.
  - Direct `func.call`의 argument. Callee의 대응 매개변수가 borrowed summary를
    가질 때만 해당한다.

  그 밖의 모든 사용은 escape이며 retained를 선택한다. 값으로서의 return과 저장,
  `adt.struct_set`의 대상, branch와 block argument 전달, indirect call과 tail
  call, 다른 region에서의 사용, 알 수 없는 operation이 여기에 속한다. Closure,
  ability handler, continuation capture는 저장이므로 escape다.

  Summary는 direct call graph 위의 fixed point다. 후보 매개변수를 borrowed로 두고
  시작해 escape를 찾을 때마다 retained로 내린다. 강등은 단조이고 매개변수 수가
  유한하므로 종료하며, 남은 borrowed 매개변수는 모든 사용이 위 목록에 속한다.

  후보가 되려면 정의가 caller의 동기적 lifetime 보장을 가져야 한다.

  - Recursive SCC에 속한 정의의 매개변수는 후보가 아니다.
  - `abi`를 가진 정의의 매개변수는 후보가 아니다. 그 caller는 호출 동안 owning
    frame을 유지한다는 계약을 주지 않는다.
  - [`consumed`](#proper-tail-ownership-transfer)로 표시된 매개변수는 후보가
    아니다. Proper tail transfer의 target은 caller frame보다 오래 살 수 있다.
    Representation/ABI 경계가 module 내부 callable의 모든 매개변수를 consumed로
    기록하므로, 이 추론은 경계가 표시하지 않은 매개변수에만 적용된다.

  Body가 없는 `extern "C"` 선언은 별도의 신뢰 경계다. 그 managed argument는 호출
  동안 borrowed이고 managed 결과는 새 owned 값이다.

  For a proven borrowed parameter, RC insertion omits both the entry `retain`
  and every parameter `release`; this keeps acquisition and release decisions
  under one ownership proof instead of matching generated releases afterward.
- **Temporary field borrows:** Typed planning은 `adt.struct_get` 또는
  `adt.variant_get`이 읽은 exact declared managed field를 그 projection의 owner에서
  파생된 temporary borrow로 분류할 수 있다. Owner는 projection operand의 typed alias
  root이다. 변환된 `clif.load` result type, address provenance, `core.ptr` operand는
  ownership evidence가 아니다.

  파생 borrow는 자신의 ownership unit을 갖지 않는다. 대신 다음 불변식이 성립해야
  한다: 파생 값의 모든 사용이 끝날 때까지 그 값에 도달하는 owned root가 살아 있고,
  root에서 파생 값까지의 field 경로가 그동안 바뀌지 않는다.

  - Nested projection의 owner가 다시 파생 borrow이면 owner 관계를 따라 올라가 만나는
    첫 owned 값이 root이다. Liveness는 파생 값의 모든 사용을 root의 사용으로도 세므로
    root의 final release는 마지막 파생 사용 뒤에 온다. 이 규칙은 block 경계와 loop를
    포함한 CFG 전체의 liveness fixed point에 그대로 적용된다.
  - 파생 값이 현재 함수의 동적 범위를 벗어나거나 독립된 unit이 필요한 사용에서는 그
    사용 직전에 unit을 획득한다. `func.return`, block argument로의 전달, proper-tail
    transfer, aggregate field로의 저장, consumed 매개변수로의 ordinary call이 여기에
    해당한다. Retained 매개변수로의 ordinary call은 callee가 스스로 unit을 획득하고
    caller의 root가 호출 동안 살아 있으므로 caller-side 획득이 없다.
  - Module 안의 어떤 `adt.struct_set`이든 쓰는 layout은 교체 가능한 layout이다.
    교체 가능한 layout에서 읽은 projection은 borrow하지 않고 projection 직후 자신의
    unit을 획득한다. 교체는 이전 field 값의 unit을 release하므로, 이 unit 없이는
    같은 함수의 뒤따르는 쓰기나 호출된 함수의 쓰기가 파생 값을 해제할 수 있다. 쓰는
    쪽은 cast와 closure environment를 거쳐 같은 객체에 도달할 수 있으므로 판정은
    특정 SSA 값이 아니라 layout의 nominal identity로 한다.

  이 최적화를 끈 기준 동작은 모든 managed projection이 projection 직후 자신의 unit을
  획득하는 것이다. Exact typed projection 계약을 만족하지 않는 IR은 borrow로도 기준
  동작으로도 계획하지 않고 ownership planning 오류로 거부한다.
- **Constant propagation (planned):** Elide RC for compile-time-known lifetimes

The paired-elimination, borrowed-parameter, and temporary-borrow policies are
selected independently by the native pipeline options, not stored in an IR
lowering context. Production enables proven optimizations; the baseline profile
disables them for conformance comparisons.

**Pipeline position:** Typed ownership/RTTI planning은 `scf_to_cf` 뒤와
`func_to_clif` 앞에서 실행한다. Temporary borrow lifetime dependency는 같은
plan의 owner liveness를 연장한다. Explicit materialization 뒤 paired elimination을
실행하고, 그 뒤 unrealized cast 변환·reconciliation과 RC lowering을 실행한다.

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

Module 내부의 모든 physical callable은 호출 규약과 무관하게 매개변수에 `consumed`를
쓴다. 어떤 내부 callable이든 proper tail transfer의 target이 될 수 있기 때문이다
([cranelift-backend.md](cranelift-backend.md#꼬리-호출-규약)). Platform `abi`를 가진
callable은 platform 계약을 유지하며 표시를 받지 않는다.

The representation/ABI boundary records this in the exact physical signature as
the per-parameter attribute `tribute.ownership = "consumed"` on every input,
because it owns the physical callable convention. The marker is inert
on a parameter the typed managed-reference contract does not select: unmanaged
parameters have no RC action. Only `consumed` is encoded; a managed parameter
without the marker has the `retained` callable contract, and `borrowed` is an
optimization of module-local direct calls to unmarked parameters, never part of
a signature. This is a native callable contract, not a conclusion inferred from
a converted type, name, operand position, body shape, or calling-convention
integer alone.

An ordinary call to a retained parameter performs no caller-side RC operation:
the caller keeps its own unit live across the call while the callee acquires
its own. An ordinary call to a consumed parameter acquires a new unit with
`retain` immediately before the call, leaving the caller's existing unit live. A
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

Native ownership plan은 module-local direct symbol과 complete signature를 정확히
resolve한다. Indirect edge는 exact callable signature가 필요하고 proper tail은
signature가 명시한 `consumed` 매개변수 계약도 필요하다. 이 계약은
[representation/ABI 경계](ir.md#representationabi-경계) 출구의 physical signature에서
읽으며 의미적 호출 규약에서 추론하지 않는다. Borrowed forwarding은 direct call graph의
monotone fixed point이며 recursive SCC, external/indirect/unknown call과 escape는
conservative retained ownership을 선택한다. 이 정보는 textual attribute가 아니라
현재 `IrContext`의 `OpRef`/`ValueRef`를 가리키는 opaque in-memory plan이다.

### 4. RC Lowering Pass

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

**Pipeline position:** After unrealized cast legalization and
`ReconcileUnrealizedCasts`, before `emit_module_to_native`.

---

## Ownership and Lowering Order

RC lowering follows a semantic-to-physical order:

1. Validate callable origins and managed-reference boundaries on typed IR.
2. Compute ownership actions and RTTI field information while `adt.typeref`
   identity is still available.
3. Evidence lowering이 이미 만든 `tribute_rt.into_raw` consuming transfer를 plan이
   검증·기록하고, `tribute_rt.retain`과 `tribute_rt.release`만 materialize한다.
4. Erase managed references to `core.ptr` and lower the explicit RC operations
   to physical refcount updates and type-specific destruction.

No later pass may reconstruct managedness from a raw pointer, symbol spelling,
ABI marker, operand position, or erased provenance. A shallow release may be
used only as an explicitly documented intermediate implementation stage; the
semantic contract requires type-specific release of owned managed fields.

네이티브 evidence ABI에는 명시적인 typed ownership handoff가 있다. 네이티브 lowering은
runtime 경계를 넘는 compiler-generated managed `_closure`마다 `tribute_rt.into_raw`를
만든다. Ownership plan은 현재의 정확한 closure layout, managed input, result 하나의
`core.ptr` 형상과 action anchor를 검증하고, 그 unit을 소비된 transfer로 기록한다.
이미 raw/null인 dispatch operand는 `into_raw`를 만들지 않으며 계속 unmanaged다.
같은 exact closure root의 same-block `into_raw`가 N개이면 plan은 첫 transfer 앞에
N-1 copy-acquire를 둔다. 그 밖의 live use, alias, cross-block group과
malformed/stale `into_raw`는 typed planning 경계에서 fail-closed한다.

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

Proper-tail lowering emits releases for dying,
non-transferred values before `func.tail_call` or
`func.tail_call_indirect`; no RC operation may follow the tail terminator.

---

## RTTI Table

RTTI index는 [runtime 타입 descriptor](runtime-types.md)의 번호다. Index는 layout이
아니라 descriptor마다 정해지므로, 같은 layout을 쓰는 두 소스 타입은 서로 다른 index를
가지며 release 함수는 공유할 수 있다.

RTTI table `__tribute_rtti`는 index마다 고정 크기 descriptor 레코드 하나를 담는 읽기
전용 배열이다. RTTI index는 이 배열의 레코드 번호이며, 모든 index는 레코드를 가진다.
해제와 값 출력은 모두 객체 header의 RTTI index로 같은 레코드를 읽는다.

```text
__tribute_rtti: [record; max_index + 1]
```

`__tribute_rtti`는 8바이트 정렬의 데이터 하나다. Index 레코드 배열 뒤에 index를
받지 않는 enum 레코드, 필드 배열, 이름 바이트를 둔다. 레코드 안의 참조는
`__tribute_rtti` 시작 기준의 `u32` offset이며, `0`은 참조가 없음을 뜻한다. 이름은
UTF-8 바이트이고 NUL로 끝나지 않는다. 링크 시점에야 정해지는 값은 release 함수
주소뿐이다.

```text
record:  ptr release_fn_or_null | u32 kind | u32 field_count | u32 name
         | u32 name_len | u32 tag_index | u32 enum_record | u32 fields | u32 0
field:   u32 name | u32 name_len | u32 field_kind
```

`release_fn`은 그 index의 release 함수다. Null이면 얕은 해제를 뜻한다. Native RTTI
생성은 table을 함수 재배치가 달린 `clif.data`로, table을 통해 해제를 디스패치하는
`__tribute_deep_release`를 `clif.func`로 IR에 선언한다. 해제할 크기는 table에 두지
않고 `__tribute_deep_release(ptr, size)`의 인자로 받는다.

| `kind` | 뜻 |
| ---- | ---- |
| `0` | struct |
| `1` | variant. `enum_record`가 소속 enum 레코드, `tag_index`가 선언 순서의 variant 번호 |
| `2` | enum. Index를 받지 않으며, 소속 variant 레코드의 `enum_record`가 이 레코드를 가리킨다 |
| `3` | compiler 소유 builtin 값(예약 index) |

정수 칸은 target의 native byte order를 따른다. `field_kind`의 하위 8비트는 분류,
그다음 8비트는 scalar의 bit 폭이다. 분류는 ownership 계획이 의미 타입으로 정한다.
Managed 판정을 받은 타입은 managed 참조나 동적 값이고, `Int`·`Nat`·`Bool`·`Float`는
각자의 scalar 분류다. 부호 정보가 없는 `core` 정수는 부호 없는 정수로 기록한다.
그 밖의 unmanaged 포인터, 코드 참조, runtime buffer는 따라가지 않는 unmanaged
포인터로 기록한다.

| 분류 | 뜻 |
| ---- | ---- |
| `0` | managed 참조 |
| `1` | 동적 값 |
| `2` | unmanaged 포인터 |
| `3` | 부호 있는 정수 |
| `4` | 부호 없는 정수 |
| `5` | 부동소수 |
| `6` | bool |

Variant의 필드 이름은 선언 순서의 위치 번호(`"0"`, `"1"`, …)다. 예약 index의 값은
`kind = 3`인 레코드를 가지며, payload가 하나인 boxing된 scalar는 그 scalar를 필드
하나로 설명한다.

**Index allocation:**

| Index | 의미 |
| ---- | ---- |
| `0` | Runtime이 할당하는 `Bytes`. Release 함수 없음, 얕은 해제. |
| `1`–`4` | boxing된 `Bool`, `Nat`, `Int`, `Float`. 고정 크기 release |
| 예약 범위 다음 | ownership planning이 할당 순서대로 정한 struct와 variant의 descriptor |

예약 범위는 compiler가 생성하는 할당 operation 없이 runtime이나 boxing lowering이
만드는 값에만 쓴다. 할당 operation으로 만드는 값은 모두 예약 범위 다음 번호를 받는다.
소스 struct와 variant뿐 아니라 closure layout처럼 compiler가 소유한 layout의 할당도
여기에 속하며, 그 descriptor는 compiler 소유 이름과 필드 종류를 가진다. Evidence처럼
RC header 없이 unmanaged로 다루는 값은 RC 객체가 아니므로 RTTI index와 descriptor를
갖지 않는다.

RTTI index는 전체 프로그램 컴파일을 전제로 한 프로그램 내부 번호다. Table과
`__tribute_deep_release`는 그 프로그램의 모듈 안에서만 index를 해석하며, runtime과
공유하는 번호는 `0`뿐이다. 따라서 예약 범위를 늘릴 때 호환 단계가 필요 없고, 사용자
layout index는 예약 범위 바로 다음부터 시작한다. 따로 컴파일한 단위 사이에서 객체가
오가게 되면 이 전제가 깨지므로, 그때는 번호 대신 header나 descriptor가 스스로 layout을
설명하는 방식으로 바꿔야 한다.

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
