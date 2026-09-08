# Cranelift Backend Architecture

> 이 문서는 Tribute → Native 컴파일 백엔드의 아키텍처를 정의한다.
> WASM 백엔드 문서는 [wasm-backend.md](wasm-backend.md)를 참조.

## Overview

Cranelift 백엔드는 WASM 백엔드와 동일한 **2-layer 패턴**을 따른다:

1. **타겟 독립적 IR 유지**: trunk-ir는 특정 타겟에 종속되지 않음
2. **Backend-specific lowering**: `clif.*` dialect은 Cranelift IR과 1:1 대응
3. **관심사 분리**: lowering (tribute-passes/native)과
   emission (trunk-ir-cranelift-backend) 분리

---

## 크레이트 구조

```mermaid
graph TD
    subgraph "trunk-ir (언어 독립적)"
        dialects["dialect/\nfunc, arith, scf, cont, ...\nwasm.rs | clif.rs"]
    end

    subgraph "trunk-ir-cranelift-backend (trunk-ir만 의존)"
        translate["translate.rs — ObjectModule → object file"]
        function["function.rs — clif.* → Cranelift FunctionBuilder"]
        validation["validation.rs — pre-emit 검증"]
        passes["passes/\nfunc_to_clif, arith_to_clif, ..."]
    end

    subgraph "tribute-passes/src/native/ (tribute-ir 의존)"
        lower["lower.rs — 오케스트레이션"]
        type_conv["type_converter.rs — 네이티브 타입 변환"]
        cps["CPS effect lowering\nlower_ability_perform + lower_handle_dispatch"]
        rc["rc.rs — RC 삽입 (future)"]
    end

    subgraph "tribute (main crate)"
        pipeline["pipeline.rs — --target native"]
    end

    dialects --> passes
    dialects --> cps
    passes --> translate
    translate --> function
    translate --> validation
    lower --> passes
    lower --> cps
    pipeline --> lower
    pipeline --> translate
```

---

## Lowering 경로

String constant lowering reads the exact prelude `String` `TypeRef` from the
root module's `tribute.type.string` metadata. It does not search interned
ADT types by `String`, `Leaf`/`Branch`, or representation layout. Missing
metadata is a lowering error whenever `adt.string_const` is present; this keeps
hand-written or text-round-tripped IR from silently selecting a user lookalike.

### Private List layout

The native backend must eliminate shared `list.*` operations before the
backend-ready boundary. It lowers them to a private immutable RC-managed RRB
tree. A representative physical shape is:

```text
root          = RC object [length, depth, root_node, tail_leaf]
internal node = RC object [child_count, cumulative_sizes, children...]
leaf node     = RC object [element_count, elements...]
```

The exact branching factor, packing, and empty sentinel are target-private and
are not an `adt.enum List` contract. Frontend/shared IR never names RRB node
fields. Node allocation uses the standard RC header, RTTI, retain/release
insertion, and deep-release path. Internal nodes own their children and leaves
own reference-typed elements. Sequence observations preserve order and
persistence without exposing this layout.

### Native 타겟

```mermaid
flowchart TB
    input["TrunkIR Module\n(func.*, arith.*, scf.*, adt.*, evidence runtime calls)"]

    subgraph native_passes["tribute-passes/src/native/"]
        cont["CPS effect lowering\nlower_ability_perform + lower_handle_dispatch"]
        list_lower["opaque List lowering\nnative::list::lower\nlist.* → private RC nodes"]
        cfg["structured control normalization\nscf_to_cf"]
        rc_plan["typed ownership/RTTI plan\nsemantic type + CFG"]
        rc_pass["explicit RC materialization\nretain/release 삽입"]
    end

    subgraph clif_passes["trunk-ir-cranelift-backend/passes/"]
        arith["arith_to_clif\narith.* → clif.iadd, clif.fadd, ..."]
        cf["cf_to_clif\ncf.* → clif.brif/jump + blocks"]
        adt["adt_to_clif\nadt.* → clif.load/store + malloc"]
        func["func_to_clif\nfunc.* → clif.func, clif.call, ..."]
        intrinsic["intrinsic_to_posix\nstd::intrinsics::posix → clif.call"]
        const_pass["const_to_clif\nfunc.constant → clif.iconst, ..."]
    end

    subgraph emit["trunk-ir-cranelift-backend/"]
        validate["validation — 모든 ops가 clif.*인지 검증"]
        codegen["function.rs — clif.* → Cranelift IR"]
        obj["translate.rs — ObjectModule → .o"]
    end

    output[".o (object file)\n→ cc 링크 → 실행 파일"]

    input --> cont --> list_lower --> cfg --> rc_plan --> rc_pass
    rc_pass --> arith --> cf --> adt --> func --> intrinsic --> const_pass
    const_pass --> validate --> codegen --> obj --> output
```

### WASM 타겟과 비교

| 측면 | WASM | Native |
| ---- | ---- | ------ |
| Effect | CPS tail-call handling | CPS tail-call handling |
| 메모리 | WasmGC (런타임 GC) | Reference Counting |
| ADT | GC struct/array | 포인터 + load/store |
| 제어 흐름 | Structured (block/loop/if) | CFG (brif/jump/br_table) |
| 함수 참조 | funcref + table | 함수 포인터 |
| 출력 | `.wasm` binary | `.o` object file |

---

## `clif.*` Dialect

Cranelift IR과 1:1 대응하는 저수준 연산. 전체 연산 목록은 [ir.md](ir.md#clif-dialect)를 참조.

핵심 차이점 (`wasm.*` 대비):

- **GC 타입 없음**: struct/array 대신 포인터 + load/store
- **CFG 기반**: structured control flow 대신 brif/jump/br_table
- **스택 할당**: stack_slot으로 로컬 메모리 할당 가능
- **함수 포인터**: funcref 대신 symbol_addr로 함수 주소 획득

### `clif.func_sig` 네이티브 호출 계약

네이티브 lowering은 공통 callable type을 유지하지 않고 `clif.func_sig`를 소유한다.
계약은 순서 있는 입력 뒤에 순서 있는 결과를 평탄한 벡터에 저장하고, 필수 `u32`
`num_inputs`와 `num_results` 속성으로 경계를 구분한다. 두 delimiter와 예약되지 않은
타입 속성은 타입 동일성에 참여하며, delimiter는 ABI나 CPS marker가 아니다. 네이티브
함수 정의와 선언, 직접·exact indirect call, return, tail transfer, emission은 이
target-owned contract를 소비한다. 호출 계약 변환은 그 내부의 중첩 타입 메타데이터를
보존하고, 타입이 지워진 함수 포인터, 심볼, ABI 문자열, 저장 형태에서 exact contract를
추론하지 않는다.

네이티브 최종 호출 계약의 각 operand와 result slot은 `clif.func_sig`의 같은 순서
slot과 정확히 같은 TrunkIR type이어야 한다. semantic reference SSA 값은 native
lowering이 그 producer 또는 block argument를 `core.ptr`로 명시적으로 낮춘 뒤에만
`core.ptr` slot을 채울 수 있다. 검증과 emission은 dialect 이름, type attribute, ABI
문자열, symbol, erased representation에서 pointer 동치를 추론하지 않는다. 이 규칙은
`core.nil`의 정해진 zero-width projection과 별개이며, 다른 contract type 사이의
호환성 규칙을 만들지 않는다.

### Zero-width `core.nil`

`core.nil`은 논리 TrunkIR `Unit` SSA 값이지만 Cranelift runtime representation은 없다.
따라서 native emitter는 함수 시그니처, entry와 non-entry block parameter, CFG edge
operand, 직접·간접 call operand, 직접·간접 tail-call operand에서 nil을 생략한다.
논리 TrunkIR는 바꾸지 않은 채 non-nil parameter와 operand의 순서(간접 호출의 non-nil
callee 포함)를 보존한다. Nil return, constant, return operand에도 같은 zero-width
규칙이 적용된다.

---

## Effect 구현: CPS Tail-Call

Effect handling은 tail-call CPS 방식으로 처리된다.
`lower_ability_perform`과 `lower_handle_dispatch` pass가
ability 연산을 handler_dispatch 클로저 호출로 변환한다.

물리 CPS 판정은 exact `Cps` convention과 빈 결과 목록의 조합이다. 실제
Direct/EvidenceDirect Unit 결과와 살아 있는 nil SSA 값의 zero-width 처리는 유지한다.
최종 dispatch는 operand와 독립적인 compiler-owned canonical shared signature를
기존 Native 변환으로 낮추며 machine 입력은 `ptr, ptr, ptr, i32, i32, i32, ptr`,
결과는 빈 목록이다. Operand는 이 고정 계약과 정확히 대조한다.

상세 내용은 [cps-effects.md](cps-effects.md)를 참조.

---

## 메모리 관리: Reference Counting

상세 내용은 [implementation.md](implementation.md#cranelift-reference-counting)를 참조.

Private native runtime helper는 target stage에서 exact physical callable ABI로만
선언한다. 이 helper의 parameter와 result는 `core.ptr` 또는 scalar physical type이며
`adt.typeref`를 사용할 수 없다. Source-logical managed reference와 private runtime
pointer 사이의 경계는 이름, `abi` 문자열 또는 symbol 위치로 추론하지 않는다.

Native ownership/RTTI plan은 `scf_to_cf` 뒤와 `func_to_clif` 앞에서 한 번 만들며
IR을 변경하지 않는다. `adt.typeref`와 compiler-generated managed layout의 semantic
type, exact callable contract와 CFG liveness만 사용한다. RTTI deep-release field와
entry/call/store/load/final-use/tail action은 이 plan에 함께 들어간다. 이후
`core.ptr`는 이미 선택된 explicit RC operation의 physical operand일 뿐이다.

Native RC materialization은 같은 type-erasure 전 경계에서 검증된 plan을 즉시
소비한다. Evidence runtime에 저장할 managed `_closure`는 이 경계에서만
`tribute_rt.into_raw`로 ownership unit 하나를 transfer하며, 이후 명시적 conversion이
이를 unmanaged `core.ptr`로 낮춘다. IR을 바꾸기 전에 전체 insertion schedule을 검증하며,
`func_to_clif` 뒤에는 ownership, liveness, pointer provenance를 다시 발견하는 pass를
실행하지 않는다.

### ADT 메모리 레이아웃

```text
Struct: [fields in order, naturally aligned]
Enum:   [tag: i32] [padding] [payload: max(variant sizes)]
Array:  [length: i64] [elements...]
```

### RC Object 헤더

```text
[-8 bytes] refcount: u32 + type_id: u32
[ 0 bytes] first field
```

---

## 구현 단계

### Phase 1: 기본 함수 컴파일

- `clif` dialect 정의
- `trunk-ir-cranelift-backend` 크레이트 스캐폴딩
- `func_to_clif` + `arith_to_clif` passes
- Cranelift codegen (function.rs + translate.rs)
- `fn main() -> Int { 42 }` → object file

### Phase 2: 제어 흐름 + ADT + 클로저

- `scf_to_clif` pass (CFG 변환)
- `adt_to_clif` pass (malloc/free 기반)
- 간접 호출 (call_indirect)
- if/case/loop, struct/enum 지원

### Phase 3: Reference Counting

- RC retain/release 삽입 pass
- Valgrind / AddressSanitizer 검증

### Phase 4: CPS Tail-Call Effect Handling

- `lower_ability_perform` + `lower_handle_dispatch` passes
- Evidence 런타임 (native): `new-plans/cps-effects.md`의 Marker layout과
  `__tribute_evidence_*` C ABI를 따른다.

### Phase 5: E2E 파이프라인

- `tribute compile --target native file.trb` → 실행 파일
- E2E 테스트 (ability 포함)

### Phase 6: Native Basic I/O

- shared `tribute_io.write`와 `tribute_io.read_line`을
  [io.md](io.md#native-runtime-abi)의 private runtime ABI로 낮춘다.
- Runtime descriptor를 high-level `ReadLineResult` ADT로 바꾼 뒤 기존 SCF, ADT,
  memory lowering을 적용한다.
- Native runtime은 Tribute enum/RTTI layout에 의존하지 않는다.
- E2E 테스트는 subprocess stdin에 raw bytes를 주입하여 빈 줄, partial EOF, EOF,
  invalid UTF-8을 검증한다.

---

## References

- [Cranelift](https://cranelift.dev/) — Rust로 작성된 코드 생성기
- [wasm-backend.md](wasm-backend.md) — WASM 백엔드 아키텍처 (대칭 구조)
