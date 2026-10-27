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
module in `tribute-ir`. 배열의 layout은 [Array](#array)가 정한다.

### Boxed Primitives

Boxed primitives are the simplest heap objects — just the raw value:

| Type | Payload Size | Layout |
| ---- | ----------- | ------ |
| boxed i32 (Int/Nat/Bool) | 4 bytes | `[i32 value]` |
| boxed f64 (Float) | 8 bytes | `[f64 value]` |

### Bytes

`Bytes` 값은 RC object이고, 읽는 byte 범위와 그 byte의 소유자를 함께 가진다.

```text
Bytes: [ptr: 첫 byte의 주소] [len: u64] [owner: Bytes 또는 null] [cap: u64] [bytes...]
```

`ptr`와 `len`은 이 값이 보는 byte 범위다. 읽는 쪽은 이 둘만 쓰며 `owner`와 `cap`을
보지 않는다. `cap`은 객체 뒤에 붙은 byte 수이고, `owner`는 범위의 byte를 가진 다른
`Bytes`에 대한 owned 참조다.

| 값 | `ptr` | `owner` | `cap` |
| ---- | ---- | ---- | ---- |
| 새 byte를 만든 결과(이어 붙이기, 입력 읽기) | 자신의 뒤에 붙은 byte | null | 붙은 byte 수 |
| Byte를 공유하는 결과(잘라내기) | 다른 `Bytes`에 붙은 byte의 일부 | 그 `Bytes` | `0` |
| Literal | `clif.data` 같은 정적 저장소 | null | `0` |

- RC object는 할당된 뒤 옮겨지지 않으므로 자신의 뒤에 붙은 byte를 가리키는 `ptr`는
  객체가 살아 있는 동안 유효하다.
- 잘라내기의 결과는 byte를 실제로 가진 `Bytes`를 retain해 `owner`에 둔다. 원본이
  byte를 공유하는 값이면 원본이 아니라 원본의 `owner`를 retain하고, 원본이 정적 byte를
  보면 `owner`는 null이다. 따라서 `owner`는 다른 `owner`를 거치지 않으며, 결과는 원본이
  release된 뒤에도 byte를 읽을 수 있다.
- `Bytes`의 release는 null이 아닌 `owner`를 release한 뒤, 고정 부분에 `cap`을 더한
  크기로 자신을 해제한다.

Byte 하나를 읽는 compiler intrinsic은 native에서 runtime 함수 호출이 된다. `ptr`에서
읽어 낸 주소는 `Bytes`가 살아 있는 동안만 유효한데, 호출은 argument를 빌리므로 읽는
동안 `Bytes`가 release되지 않는다.

Runtime 함수는 `Bytes` argument를 호출 동안 빌리고, `Bytes` 결과는 호출자가 unit
하나를 갖는 새 값으로 돌려준다. Runtime이 다른 기록 안에 raw pointer로 담아 돌려준
`Bytes`도 같은 방식으로 unit 하나를 넘겨준다.

### Array

[`adt.array<E>`](ir.md#adt-배열) 값은 compiler가 할당하는 RC object다.

```text
Array: [len: i64] [element: E...]
```

- 원소는 payload offset 8부터 빈틈 없이 놓인다. 원소 하나의 크기(stride)는 `E`의
  native 표현 크기이며 struct field와 같다. 8비트 정수는 1, 16비트 정수는 2, 32비트
  정수, `f32`, native에서 `i32`로 표현하는 `Bool`은 4, 64비트 정수, `f64`, pointer와
  managed 참조는 8이다. 모든 stride가 8 이하의 2의 거듭제곱이므로 원소는 자연
  정렬된다.
- 할당 크기는 RC header, `len`, 원소를 더한 `16 + len × stride`다. 해제도 같은
  크기를 쓴다.
- Managed 참조 원소는 owned 참조다. `adt.ref_null`이 만든 null도 원소가 될 수
  있으며 retain과 release는 null에 대해 아무것도 하지 않는다.
- 배열 operation의 ownership 규칙은
  [typed ownership plan](#1-typed-ownership-and-rtti-plan), 해제는
  [type-specific release](#type-specific-release)가 정한다.

### Evidence

Evidence와 marker는
[compiler 소유 runtime 모듈](modules.md#compiler-소유-runtime-모듈)에 선언한
[source 타입](cps-effects.md#handle-evidence-extension--handler-closures)이다.
Evidence는 marker의 일반 [배열](#array)이고 marker는 일반 struct다. 다른 배열과
struct처럼 [typed ownership plan](#1-typed-ownership-and-rtti-plan)이 ownership을
계획하고, 할당 descriptor를 받으며, 예약 RTTI index나 별도의 release 함수를 갖지
않는다.

두 object는 만들어진 뒤 바뀌지 않는다. Evidence runtime은 `Slots`의 복사 연산으로
새 evidence를 만들 뿐 원소를 덮어쓰지 않는다. 참조는 먼저 만들어진 object만
가리키므로 순환하지 않는다.

Marker의 dispatch closure는 `Option(Dispatch)` 필드에 담긴 managed closure 값이다.
Handler를 설치하는 lowering은 그 closure를 다른 managed argument처럼 runtime 항목
호출에 넘기며 `core.ptr`로 바꾸지 않는다.

Evidence parameter는 다른 physical input과 같은 ownership contract를 따른다.

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

`tribute_rt` dialect는 두 RC operation과 두 ownership 경계 operation을 제공한다.

```text
tribute_rt.retain(ptr) -> ptr    // refcount++, return same pointer
tribute_rt.release(ptr)          // refcount--, free if zero
tribute_rt.into_raw(value) -> core.ptr // typed ownership unit 하나를 소비
tribute_rt.from_raw(ptr) -> value      // unit 하나를 가진 typed 값을 만든다
tribute_rt.release_moving(ptr, moved...) // owner를 release하며 field unit을 moved로 옮긴다
```

`tribute_rt.release_moving`은 owner의 unit 하나를 release하면서, owner의 managed field가
가진 unit을 그 field에서 읽은 `moved` operand로 옮긴다. Operation 뒤에 `moved`는 각자
unit 하나를 가진 owned 값이다. Owner가 유일하게 참조되면 옮기는 field는 release하지
않고, `release_offsets` 속성이 가리키는 나머지 managed field만 release한 뒤 owner를
`alloc_size`로 해제한다. 공유되어 있으면 `moved`마다 unit을 먼저 획득한 뒤 owner를
보통의 `tribute_rt.release`처럼 release한다. 다른 소유자가 그사이 release하면 이
release가 마지막일 수 있기 때문이다. Materialization은 owner의 exact struct layout에서
`release_offsets`와 header를 포함한 `alloc_size`를 계산한다. 이 operation은
[field move](#1-typed-ownership-and-rtti-plan)를 표현하며 typed plan만 만든다.

이 연산은 dialect 수준에서 다음 순서로 처리된다.

1. Native lowering이 필요한 `into_raw`와 `from_raw`를 먼저 만든다.
2. 검증된 type-erasure 전 ownership plan은 `retain`/`release`만 **materialize**하고
   검증한 `into_raw`와 `from_raw`를 보존한다.
3. Ownership lowering이 plan을 다 쓴 뒤, managed type을 physical type으로 바꾸기 전에
   `into_raw`와 `from_raw`를 explicit native representation conversion으로
   **lower**한다.
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

`tribute_rt.from_raw`는 그 역이다. RC object의 payload를 가리키는 `core.ptr`를 받아
unit 하나를 가진 typed managed 값을 만든다. 결과는 할당의 결과와 같은 새 owned 값이다.
이 operation은 pointer가 가리키는 object의 layout과 unit의 이전을 만든 쪽이 보장하는
자리에서만 쓴다. Native lowering이 직접 할당한 object, 그리고 runtime이 raw pointer로
담아 unit과 함께 넘겨준 object가 그런 자리다. RC header가 없는 memory를 managed 값으로
바꾸는 수단이 아니다.

두 operation의 타입은 선언이 고정한다. `into_raw`는 managed reference를 받아 `core.ptr`를
만들고, `from_raw`는 `core.ptr`를 받아 managed reference를 만든다. 두 operation은 managed
type이 physical type으로 바뀌기 전에 lower되므로, 존재하는 동안 이 선언을 항상 만족한다.

`core.ptr`와 managed reference 사이를 오가는 길은 이 두 operation뿐이다. 둘 사이의
`core.unrealized_conversion_cast`는 ownership의 의미를 말하지 못하므로 ownership
planning이 거부한다. Managed reference의 payload를 `core.ptr`로 빌려 제자리에서 읽는
operation은 없으며, 그런 읽기는 값을 빌리는 runtime 함수 호출로 한다.

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
Module 범위 분석이 function 정의, 검증된 managed nominal layout,
`adt.struct_set`이 쓰는 layout, closure 할당 layout(base와 capture layout)과 할당 순서의
RTTI descriptor를, function 범위 분석이 flat CFG, managed 값,
exact alias root, managed projection-owner 관계, 그중 borrow할 수 있는
projection과 liveness 입력을 제공한다. 이 사실은 `NativeOwnershipPlanOptions`와
무관하며 borrow elision과 entry ownership 정책은 사실을 소비하는 planner와
action planner가 적용한다.

Parameter-entry 계약은 module facts와 call graph에 의존하는 module 범위
`NativeEntryContracts` 분석이 제공한다. 분석은 정의된 함수의 managed
parameter가 borrow로만 쓰이는지를 항상 증명하고, 두 view로 결과를 낸다.
`elide_proven_borrowed_parameters`만 view를 선택한다. 증명 view는 증명된
parameter를 `Borrowed`로, 보수적 view는 같은 parameter를 `Retained`로 둔다.
Bodyless `extern "C"` 선언의 `Borrowed`는 증명이 아닌 선언 계약이므로 두
view에서 같다.

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
   last-use move, field move, return과 proper-tail action을 정한다.
3. 같은 typed predicate로 aggregate별 RTTI managed-field bitmap을 만든다.
4. 모든 identity, signature, CFG, action target과 ordering을 검증한 뒤 하나의
   deterministic plan을 반환한다.

Block argument로의 전달은 source의 unit을 destination으로 옮긴다. 옮길 unit은
source마다 하나뿐이다. Source가 선택된 liveness view에서 branch 뒤에도 살아 있으면
그 unit은 source에 남고, 같은 branch가 한 source를 여러 destination에 넘기면 첫
destination만 unit을 받는다. Unit을 받지 못한 destination마다 branch 직전에 새
unit을 획득한다.

Aggregate field로의 저장(`adt.struct_new`, `adt.variant_new`, `adt.struct_set`의
값, `adt.array_of`와 `adt.array_splice`의 `elements`, `adt.array_set`의 값)과
ordinary call의 consumed 매개변수는 독립된 unit을 받는 자리다. 기준 동작은 그
자리마다 직전에 새 unit을 획득하고 값의 unit은 값에 남긴다. Last-use move 정책을
켜면 그 operation이 값의 마지막 사용일 때 값의 unit을 그 자리로 옮기고, 획득도 뒤따르는
final release도 두지 않는다. 마지막 사용은 final release 위치를 정하는 것과 같은
판정이다: 값은 선택된 liveness view에서 block 끝에 살아 있지 않고, block 안의 뒤
operation이 값이나 값에서 파생된 borrow를 쓰지 않는다. 옮길 unit이 있어야 하므로
값은 owned root여야 하며, 파생 borrow와 borrowed 매개변수는 옮기지 않는다. 또한 그
operation이 값의 root에 닿는 모든 operand가 unit을 받는 자리여야 한다. Indirect
call의 callee, `adt.struct_set`과 `adt.array_set`의 대상, borrowed·retained
매개변수로의 argument, `adt.array_new`의 `initial`, `adt.array_splice`의 `source`,
값에서 파생된 borrow는 호출이나 저장 동안 caller의
unit에 기대므로, 이런 operand가
하나라도 있으면 모든 자리가 기준 동작대로 획득한다. 한 operation이 한 값을 여러
자리에 넘기면 첫 자리만 unit을 받고 나머지는 획득한다.

`adt.array_new`는 `initial`을, `adt.array_splice`는 `source`를 operation 동안
빌린다. `initial`을 담은 원소마다, 그리고 `source`에서 결과로 복사한 managed 원소마다
operation 자신이 unit을 하나씩 획득한다. 이 원소의 수는 실행 때 정해지므로 plan은
이 획득을 action으로 갖지 않으며, release 함수가 field를 release하는 것처럼
operation의 lowering이 소유한다. 결과는 unit 하나를 갖는 새 owned 값이다.

`adt.array_set`은 `adt.struct_set`과 같다. 대상 배열은 operation 동안 빌리고, 새
원소는 위의 저장 자리로 unit을 받으며, 바뀌기 전 원소의 unit은 replaced-field
release로 release한다.

Owner에서 읽은 borrow는 unit이 없으므로, unit이 필요한 사용(저장, consumed 매개변수,
return, block argument, proper tail)마다 기준 동작대로 새 unit을 획득한다. 그 owner가
같은 block에서 죽으면 owner의 final release가 같은 field를 다시 release하므로, owner가
유일하게 참조될 때 이 획득과 field의 release는 서로 상쇄된다. Field move 정책을 켜면
owner의 final release를 `tribute_rt.release_moving`으로 바꾸어 field의 unit을 그 borrow로
옮긴다. 옮긴 borrow는 그 지점부터 unit 하나를 가진 owned 값이며, 이후의 사용은 위의
last-use move와 final release 규칙을 그대로 따른다. Field move는 다음 조건을 모두
만족할 때만 둔다.

- Owner는 owned root다. Owner 자신이 옮겨진 borrow이면 옮긴 지점부터 owned root다. 이렇게
  nested projection은 단계마다 같은 규칙으로 옮긴다.
- Owner의 layout은 release가 정적 `alloc_size`로 해제하는 layout이다. 즉 exact struct,
  continuation frame layout이다. Variant마다 크기가 다른 enum 값, capture layout마다
  크기가 다른 closure 값, 길이마다 크기가 다른 배열과 `tribute_rt.anyref`/`intref`는
  RTTI로 해제하므로 대상이 아니다. 교체 가능한 layout에서
  읽은 값은 이미 자신의 unit을 가지므로 옮기지 않는다.
- Owner의 final release와 옮기는 borrow의 모든 사용이 한 block 안에 있다. Edge에서 죽는
  owner와 block 경계를 넘는 borrow는 옮기지 않는다.
- `release_moving`을 둘 지점 P가 있다. Owner 자신의 모든 사용과, 옮기지 않는 borrow의
  모든 사용은 P 앞에 있다. 옮기는 borrow의 모든 사용은 P 뒤에 있다. 기준 동작의 final
  release 지점을 R이라 하면 P는 R보다 앞설 수 있되, P와 R 사이의 사용은 모두 옮기는
  borrow의 사용이어야 한다. P 뒤의 사용만 unit을 받으므로, 공유된 owner의 경로에서
  unit을 획득하기 전에 그 field를 다른 객체가 가진 채 release되는 구간이 생기지 않는다.
- 옮기는 field는 owner에서 한 번만 읽는다. 같은 field를 두 번 이상 읽으면 owner를
  옮기지 않는다. Field마다 unit은 하나뿐이고, 같은 field의 여러 값 사이에서 unit의
  주인을 정하는 규칙을 두지 않는다.
- 옮길 borrow는 P 뒤에 unit이 필요한 사용이 있는 것만 고른다. 그런 사용이 없는 borrow를
  옮기면 획득 대신 final release 하나가 늘 뿐이다.

조건을 하나라도 만족하지 않으면 기준 동작대로 사용마다 획득하고 owner를 release한다.

Owner가 유일함을 plan이 정적으로 증명하면 lowering은 공유 경로를 생략할 수 있다.
증명은 다음뿐이다: owner는 같은 block의 할당 결과이고, owner에 대한 action은
call borrow, projection, final release뿐이다. Store·call·copy 획득, retained 매개변수로의
call(callee가 unit을 저장할 수 있다)과 branch 전달이 하나라도 있으면 증명이 아니다.

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
contract로 분류한다. `core.bytes`는 [`Bytes`](#bytes) 객체에 대한 managed 참조이고,
`adt.array`는 원소 타입과 관계없이 [array](#array) 객체에 대한 managed 참조다.
Unmanaged linear array인 `core.array`는 RC 객체가 아니다.
Function/code address, borrowed buffer, `core.ptr`는 unmanaged다. 변환 결과가
pointer라는 사실은 이 분류에 참여하지 않는다.

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
계산한 **payload size + RC header size**를 가진다. Replaced-field release도 field나
배열 원소의 exact declared layout으로 같은 size를 사용한다. `tribute_rt.anyref`와 `intref`처럼
static nominal layout이 없는 opaque dynamic reference와, variant마다 할당 크기가
다른 enum 값, capture layout마다 할당 크기가 다른 closure 값, 길이마다 할당 크기가
다른 배열의 `alloc_size = 0`은
deallocation size가 아니라 RTTI dispatch 전용 신호다. Closure 값은 base 타입으로
다뤄지고 native layout은 이 계산 뒤에 정해지므로, closure의 크기는 언제나 header
RTTI가 고른 layout의 release 함수가 정한다. backend는 header RTTI로 이를
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
  aliases. `tribute_rt.release_moving` releases its owner and acquires a unit
  for each moved operand at once, so the pass never matches it as an ordinary
  release and treats it as a barrier for its moved operands: no earlier
  `retain` of a moved value pairs with it.
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
  `adt.variant_get`이 읽은 exact declared managed field와 `adt.array_get`이 읽은
  managed 원소를 그 projection의 owner에서 파생된 temporary borrow로 분류할 수
  있다. Owner는 projection operand의 typed alias root이다. 변환된 `clif.load`
  result type, address provenance, `core.ptr` operand는 ownership evidence가
  아니다.

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
    쪽은 cast와 closure capture를 거쳐 같은 객체에 도달할 수 있으므로 판정은
    특정 SSA 값이 아니라 layout의 nominal identity로 한다.
  - 배열도 같다. Module 안의 어떤 `adt.array_set`이든 쓰는 `adt.array` 타입은 교체
    가능한 layout이며, 그 타입에서 `adt.array_get`이 읽은 managed 원소는 projection
    직후 자신의 unit을 획득한다. 그 밖의 배열 타입에서 읽은 원소는 borrow할 수 있는
    projection이다.

  이 최적화를 끈 기준 동작은 모든 managed projection이 projection 직후 자신의 unit을
  획득하는 것이다. Exact typed projection 계약을 만족하지 않는 IR은 borrow로도 기준
  동작으로도 계획하지 않고 ownership planning 오류로 거부한다.
- **Last-use moves:** Typed planning은 aggregate field로의 저장이나 consumed
  매개변수로의 ordinary call이 owned 값의 마지막 사용일 때 값의 unit을 그 자리로
  옮긴다. 조건과 operand 규칙은 [typed ownership plan](#1-typed-ownership-and-rtti-plan)이
  정한다. Return과 proper tail은 이 정책과 무관하게 항상 unit을 옮긴다.
- **Field moves:** Typed planning은 같은 block에서 죽는 owner에서 읽은 borrow에 unit이
  필요한 사용이 남아 있으면 owner의 final release를 `tribute_rt.release_moving`으로
  바꾸어 field의 unit을 그 borrow로 옮긴다. 조건과 배치는
  [typed ownership plan](#1-typed-ownership-and-rtti-plan)이 정한다.
- **Constant propagation (planned):** Elide RC for compile-time-known lifetimes

The paired-elimination, borrowed-parameter, temporary-borrow, last-use-move, and
field-move policies are
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
`retain` immediately before the call, leaving the caller's existing unit live,
unless the [last-use move](#1-typed-ownership-and-rtti-plan) transfers the
caller's unit because the call is the value's last use. A
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

tribute_rt.release_moving(ptr, moved...) {alloc_size, release_offsets} ->
    if ptr == null: jump continue_block
    %rc_addr = clif.iadd(ptr, clif.iconst(-8))
    %rc = clif.load(%rc_addr)
    %is_unique = clif.icmp(%rc, clif.iconst(1), cond="eq")
    clif.brif(%is_unique, then_dest=unique_block, else_dest=shared_block)

unique_block:
    // tribute_rt.release each field loaded at `release_offsets`
    %raw_ptr = clif.iadd(ptr, clif.iconst(-8))
    clif.call(@__tribute_dealloc, %raw_ptr, clif.iconst(<alloc_size>))
    clif.jump(continue_block)

shared_block:
    // tribute_rt.retain each moved operand, then tribute_rt.release(ptr)
    clif.jump(continue_block)
```

A count of one is the releasing unit itself, which no other holder can change,
so a plain load decides uniqueness. RC lowering emits this expansion once per
allocation size, released offsets and number of moved values, as a generated
`__tribute_release_moving_<n>(ptr, moved...)` function, and replaces each
moving release with a call to it. Expanding both paths at every owner's death
would grow the code it compiles to more than the moved units save. When the
plan proves the owner unique, only the unique path is emitted.

**Pipeline position:** After unrealized cast legalization and
`ReconcileUnrealizedCasts`, before `emit_module_to_native`.

---

## Ownership and Lowering Order

RC lowering follows a semantic-to-physical order:

1. Validate callable origins and managed-reference boundaries on typed IR.
2. Compute ownership actions and RTTI field information while `adt.typeref`
   identity is still available.
3. Native lowering이 이미 만든 `tribute_rt.into_raw` consuming transfer를 plan이
   검증·기록하고, `tribute_rt.retain`과 `tribute_rt.release`만 materialize한다.
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

배열의 release 함수는 원소 타입마다 하나다. 원소가 managed 참조이면 `len`개의
원소를 차례로 release하고, null 원소는 release가 건너뛴다. 그 뒤 `len`을 읽어
[할당 크기](#array)로 해제한다. 원소가 managed 참조가 아니면 원소를 읽지 않고 같은
크기로 해제한다.

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

### Closure captures

Native closure는 RC header, convention table 주소, capture를 담은 RC 객체 하나다
([closure 저장 layout](ir.md#closure-저장-layout),
[native 배치](cranelift-backend.md#메모리-관리-reference-counting)).

- `closure.new`는 capture layout의 `adt.struct_new`가 되므로 각 capture는 aggregate
  field로의 저장이다. Capture 자리는 독립된 unit을 받으며, capture가 값의 마지막
  사용이면 [last-use move](#1-typed-ownership-and-rtti-plan) 규칙을 그대로 따른다.
  Capture 없는 closure는 base layout 할당이며 unit을 받는 필드가 없다.
- Code 함수의 environment 입력은 closure 자신이다. 이 입력은 다른 물리 입력과 같은
  entry 계약을 따르며, closure 호출에서 closure 값을 넘기는 자리도 다른 argument와
  같은 규칙으로 계획한다.
- `closure.unpack`은 base에서 capture layout으로의 cast와 capture field projection이
  된다. Cast는 같은 객체의 alias이고, managed capture의 projection은 일반 managed
  projection 규칙을 따른다.
- Base와 각 capture layout은 서로 다른 할당 descriptor다. 이 layout들은
  `layout`과 `closure_base`로만 식별하며 struct 이름이나 필드 모양으로 판별하지
  않는다. 각 layout의 release 함수는 convention table 참조를 unmanaged 필드로 두고 plan이
  managed로 판정한 capture만 release한 뒤 객체를 해제한다. Convention table은 프로그램이
  끝날 때까지 남는 정적 data이고 managed 참조를 담지 않으므로 object graph에 간선을
  더하지 않는다. 한 closure의 모든 entry 함수는 같은 environment 입력 계약을 따른다.
- `closure.new`의 결과는 capture layout 할당에서 base로의 upcast다. Upcast가 그
  입력의 유일한 사용이면 할당의 unit은 경로가 하나이므로, `tribute_rt.into_raw`는
  이렇게 upcast된 값도 정확한 closure 소유 값으로 받는다.

### Continuation ownership

Tail-call CPS represents a continuation as a typed closure with an explicit
ContinuationFrame. Capturing a managed value into that frame creates an
independent owned reference and therefore materializes a retain. Destroying an
unresumed one-shot continuation releases every owned frame field through the
ordinary type-specific destructor. Resuming transfers the frame-owned values
according to the continuation callable contract. A resumption reads those
values from a frame that dies in the same block, so the
[field move](#1-typed-ownership-and-rtti-plan) applies: when the frame is
unique, its values keep their units instead of being acquired before the frame
is released.

Proper-tail lowering emits releases for dying,
non-transferred values before `func.tail_call` or
`func.tail_call_indirect`; no RC operation may follow the tail terminator.

---

## 비순환 object 그래프

Reference counting은 순환을 회수하지 못한다. Native RC는 별도의 순환 회수 수단을 두지
않으며, 대신 RC object의 참조 그래프가 순환하지 않음을 계약으로 둔다. 새로 저장되는
managed 참조는 항상 그 저장보다 먼저 만들어진 object를 가리킨다. 이 성질은 다음
규칙들이 함께 지킨다.

- **소스 데이터는 불변이다.** 소스 언어에는 가변 참조나 cell이 없다. Struct, variant,
  `List`, `Bytes`, closure capture는 만들 때 field를 채우고 그 뒤에 바뀌지 않는다.
  Field setter는 새 값을 만든다. 만들 때 채우는 field는 이미 있는 object만 가리킬 수
  있으므로 순환을 닫지 못한다.
- **Evidence와 marker도 만든 뒤 바뀌지 않는다.** [Evidence](#evidence)가 정한 대로
  먼저 만들어진 object만 가리킨다.
- **Managed field를 덮어쓰는 곳은 compiler가 소유한 completion cell뿐이다.**
  [값 delimiter](cps-effects.md#값-delimiter)의 completion cell은 region의 답을 한 번
  쓴다. 답에서 그 cell이나 cell을 가진 `Done`으로 가는 경로가 없다는 근거는 둘이다.
  - Resume token은 lambda capture로만 옮길 수 있고, token을 담은 closure를 struct나
    variant에 저장하면 whole-IR 검증이 거부한다. Handle의 답 타입이 그 handle을 재개하는
    closure를 담으려면 명목 타입을 거쳐야 하므로, 지금은 답이 token을 담지 못한다.
  - 답이 token을 담더라도 재개된 계산은 재개한 지점이 준 frame으로 handle 층을 다시
    만들고 이미 끝난 delimiter의 frame으로 돌아가지 않는다. 이 규칙은 token의 저장
    제한이 풀려도 유지되어야 한다.

  Root `main`의 delimiter는 `Nil`을 답으로 받으므로 managed 참조를 쓰지 않는다.
  One-shot resumption이 소비 여부를 표시하는 쓰기는 unmanaged flag를 쓴다. 두 cell의
  layout은 `adt.struct_set`이 쓰는 [교체 가능한 layout](#3-rc-optimization-pass)이므로
  그 projection은 borrow하지 않는다.
- **배열도 만든 뒤 managed 원소가 바뀌지 않는다.** `adt.array_new`,
  `adt.array_of`, `adt.array_splice`는 operand와 `source`의 원소로 새 배열을 채우므로
  struct처럼 이미 있는 object만 담는다. Managed 원소를 덮어쓰는 `adt.array_set`을
  내는 producer는 없다. 공유된 배열에 그 배열을 capture한 closure를 저장하면 순환이
  닫히므로, 그런 쓰기를 들이는 producer는 completion cell처럼 순환이 생기지 않는
  근거를 이 절에 둔다. 원소를 바꾼 배열이 필요하면 `adt.array_splice`로 새로 만든다.
- **Closure는 자신을 capture하지 않는다.** Code 함수는 closure 자신을 environment
  입력으로 받는다([Closure captures](#closure-captures)). 지역 함수가 자기 자신을
  재귀 호출한다면 그 참조는 environment 입력으로 해결하며 capture로 만들지 않는다.
  서로 호출하는 지역 함수들도 capture만 담은 environment를 공유하고, 형제 closure를
  environment에 저장하지 않는다.

이 계약을 깨는 기능은 아직 없다. 들인다면 다음 순환 대책과 함께 들인다.

- **범위에 묶인 가변 cell.** Handler가 소유하는 상태처럼 dynamic extent가 정해진
  cell을 들인다면, 저장한 값에서 cell로 돌아오는 경로가 생길 수 있다. 예를 들어 cell을
  읽는 closure를 cell에 저장하면, closure가 가진 evidence와 marker를 거쳐 cell에 닿을
  수 있다. 이런 cell은 범위가 끝날 때 내용을 release하고 비운다. 범위 밖에서 cell을
  쓰는 일은 effect 타입이 막으므로, 비우는 것은 관찰되지 않으며 그 시점에 순환이
  끊긴다.
- **범위 없는 가변 참조.** 소스 언어가 범위 없이 쓰는 가변 참조를 들인다면 순환이 남을
  수 있다. 그 회수 수단은 trial deletion 방식의 순환 collector다. 순환에 낄 수 있는
  타입, 곧 가변 참조를 담을 수 있는 타입의 object만 후보로 기록한다. 후보에서 닿는
  object 사이의 내부 참조 수를 refcount에서 빼고, 바깥 참조가 남지 않은 묶음을
  해제한다. 이 방식은 stack root를 스캔하지 않으므로 stack map이 필요 없고, RTTI의
  field 분류로 자식을 따라간다. 사용자 finalizer가 없으므로 해제 중 부활을 다루지
  않는다. Weak 참조는 순환을 끊는 책임을 프로그래머에게 넘기므로 기본 수단으로 두지
  않는다.
- **재귀 값과 lazy 값.** `let`으로 자기 자신을 참조하는 데이터나 지연 계산은 위
  가변 참조와 같은 순환을 만든다. 들인다면 같은 대책을 따른다.

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
| `4` | [배열](#array). `field_count`는 1이고 그 필드가 원소의 종류다. 원소 수는 값의 `len`이다 |

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
| `0` | [`Bytes`](#bytes). `owner`를 release한 뒤 `cap`을 읽어 해제 |
| `1`–`4` | boxing된 `Bool`, `Nat`, `Int`, `Float`. 고정 크기 release |
| 예약 범위 다음 | ownership planning이 할당 순서대로 정한 struct, variant, 배열의 descriptor |

예약 범위는 compiler가 생성하는 할당 operation 없이 runtime이나 boxing lowering이
만드는 값에만 쓴다. 할당 operation으로 만드는 값은 모두 예약 범위 다음 번호를 받는다.
소스 struct와 variant뿐 아니라 closure layout처럼 compiler가 소유한 layout의 할당도
여기에 속하며, 그 descriptor는 compiler 소유 이름과 필드 종류를 가진다. Closure는
base와 capture layout마다 descriptor를 하나씩 가지며, 배열의 descriptor 단위는
[runtime-types.md](runtime-types.md#descriptor-단위)가 정한다. RC header
없이 unmanaged로 다루는 값은 RC 객체가 아니므로 RTTI index와 descriptor를 갖지 않는다.

RTTI index는 전체 프로그램 컴파일을 전제로 한 프로그램 내부 번호다. Table과
`__tribute_deep_release`는 그 프로그램의 모듈 안에서만 index를 해석한다. Runtime과
공유하는 번호는 runtime이 직접 할당하는 `Bytes`의 `0`뿐이며
고정이다. 그 밖의 예약 범위를 늘릴 때는 호환 단계가 필요 없고, 사용자 layout index는
예약 범위 바로 다음부터 시작한다. 따로 컴파일한 단위 사이에서 객체가
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
- **Acyclicity:** The AddressSanitizer exit-time leak check fails a program
  that leaves an RC object live, including one kept alive by a cycle

---

## Deferred Decisions

- **Thread-safety:** Atomic refcount for multi-threaded code?
- **FFI boundaries:** How to handle RC objects at C FFI boundaries?
- **Optimization:** Compile-time escape analysis to elide RC?
