# Wasm Backend Architecture

> 이 문서는 Tribute → WebAssembly 컴파일 백엔드의 아키텍처를 정의한다.

## Overview

Tribute의 Wasm backend는 WasmGC (Wasm 3.0) 표현을 emit하고 native와 같은 shared
middle-end와 effect ABI를 입력으로 받는다. 백엔드는 다음 원칙을 따른다:

1. **타겟 독립적 IR 유지**: trunk-ir는 특정 타겟에 종속되지 않음
2. **Backend-specific 타입 처리**: WasmGC 타입 정의는 백엔드에서 처리
3. **관심사 분리**: lowering (tribute-passes)과 emission (trunk-ir-wasm-backend) 분리

---

## 크레이트 구조

```text
trunk-ir/
├── dialect/
│   ├── wasm.rs           # wasm ops (struct_new, array_new, call, ...)
│   └── ...               # target-independent dialects only

trunk-ir-wasm-backend/    # trunk-ir만 의존
├── translate.rs          # IR → WebAssembly binary entrypoint
├── emit.rs               # instruction/function/module emission
├── gc_types.rs           # builtin WasmGC type index layout
├── emit/gc_types_collection.rs
│                         # wasm.* ops에서 타입 정보 수집
├── passes/func_to_wasm.rs
├── passes/arith_to_wasm.rs
├── passes/scf_to_wasm.rs
├── passes/adt_to_wasm.rs
└── ...

tribute-passes/           # tribute-ir 의존
├── wasm/lower.rs         # Wasm lowering pipeline orchestration
├── wasm/evidence_to_wasm.rs
│                         # effect.* → evidence helpers + call/return_call
├── wasm/tribute_rt_to_wasm.rs
├── wasm/const_to_wasm.rs
├── wasm/intrinsic_to_wasm.rs
├── wasm/normalize_primitive_types.rs
└── ...

tribute/                  # main crate - 파이프라인 조율
└── pipeline.rs
```

Native (Cranelift) 백엔드는
[cranelift-backend.md](cranelift-backend.md)를 참조.

---

## Lowering 경로

### WasmGC 타겟

```text
tribute-ir (High-level)
├── adt.struct_new
├── adt.variant_new
├── closure.new
├── effect.extend
├── effect.dispatch_tail / effect.dispatch_cps
├── func.tail_call / func.tail_call_indirect
│
▼ tribute-passes/wasm/lower.rs
│
trunk-ir (Mid-level)
├── wasm.struct_new       # 인스턴스 생성
├── wasm.struct_get/set   # 필드 접근
├── wasm.array_new        # 배열 생성
├── wasm.call_indirect    # ordinary indirect 호출
├── wasm.return_call      # direct proper tail transfer
├── wasm.return_call_indirect
│                         # indirect proper tail transfer
├── wasm.func             # evidence lookup/extend helpers
│
▼ trunk-ir-wasm-backend
│   (gc_types_collection: wasm.* ops에서 타입 수집)
│   (builtin GC type layout + user type collection)
│
WebAssembly Binary
```

Effect lowering은 target별로 수행한다. Shared ability lowering은 `effect.*`를
만들며 Marker field 번호나 closure layout을 검사하지 않는다.
`wasm/evidence_to_wasm`은 evidence lookup/extend helper를 만들고 closure struct
`(table_idx, env)`를 풀어 semantic role에 맞는 일반 호출 또는 proper-tail call을
emit하여 해당 operation을 제거하는 Wasm 경계다.

### Wasm 결과 슬롯

`wasm.func_sig`의 저장 형식과 간접 호출의 명시적 시그니처는
[IR 호출 계약](ir.md#wasmfunc_sig-wasm-호출-계약)을 따른다.

바이너리 코드 생성은 결과 위치의 `core.nil`만 생략하고 나머지 결과의 선언 순서를
보존한다. 생략되지 않는 각 결과에는 SSA 값과 지역 변수를 할당한다. 다중 결과
호출 후에는 마지막 결과가 스택 맨 위에 있으므로 지역 변수에 역순으로 저장한다.
입력이나 값 위치의 `core.nil`은 생략하지 않고 널을 허용하는 참조로 표현한다.
Wasm 코드 생성기는 함수 본문이나 테이블 인덱스에서 CPS 여부를 추론하지 않는다.

### Terminal structured-control 결과

`scf_to_wasm`은 결과가 정확히 하나의 `core.never`이고 사용되지 않으며 블록의
마지막 operation인 `scf.if`와 `scf.loop`를 zero-result Wasm 제어 연산으로
낮춘다. 모든 진입 successor의 region은 단일 블록이고, 마지막 operation이
검증된 `CallableExit` 또는 같은 terminal 판정을 만족하는 중첩 if/switch여야
한다. Native structured-to-CFG lowering과 같은 `RegionBranch`/`CallableExit`
판정을 사용하며, parent 복귀, 불완전한 interface 응답, 오류는 terminal 증거가
아니다. Loop의 continue 순환이나 임의의 다중 블록 CFG는 분석하지 않는다.

Terminal if는 결과 없는 `wasm.if`, terminal loop는 결과 없는 `wasm.block`과
`wasm.loop`가 된다. Resultless switch도 블록 마지막에 있고 explicit default를
포함한 모든 arm이 terminal이면 결과 없는 Wasm 비교 분기들을 만든다. Source
switch에 결과를 추가하지 않으며 일반 fallthrough switch의 계약은 유지한다.

중첩 pattern이 source operation을 바꾸기 전에 전체 입력을 검사하고 terminal
판정을 수집한다. 사용 중이거나 terminal 증명이 안 되는 `Never` 결과는 mutation
전에 거부한다. 결과 제거 API는 모든 기존 결과가 미사용이고 새 결과가 비어
있음을 확인하며, 일반 rewrite의 결과 개수 일치 조건을 완화하지 않는다.

이 규칙은 [CPS 적법화 경계](cps-effects.md#적법화-경계)의 callable 결과 물리화와
별개다. 일반 `never`/`nil` 치환이 아니며 기존 `core.nil`과 다른 값 결과는
보존한다. `Never` 값의 지역 변수나 block argument를 만들거나 emitter에
`core.never` 표현을 추가하지 않는다.

### 생성 operation의 타입 출처

Wasm 경계를 넘어 새로 생성되는 모든 `wasm.*` operation의 result type과 block
argument type은 Wasm target `TypeConverter`가 만든 값이어야 한다. Shared IR
spelling을 그대로 복사하면 `adt.typeref`나 logical `core.array`가 backend-ready
경계까지 남아, physical assignability와 GC layout 검증이 정상 producer를
거부한다. 이것은 구현 편의가 아니라 [ir.md](ir.md)의 backend-ready 소거
요구를 Wasm 쪽에서 만족시키는 조건이다.

Pass가 새 operation을 만들며 결과 타입을 정할 때는 identity converter 대신
Wasm target converter를 사용하고 `PatternRewriter::result_type` /
`result_types`로 result type을 읽는다. `scf.loop` body처럼 새 operation이
detached region을 소유하면 그 region의 block argument도 같은 converter가 만든
타입으로 선언한다. Operand type은 producer가 정한 값
타입이므로 여기서 다시 변환하지 않는다. 이미 존재하는 `wasm.*` operation의
결과를 검증할 때는 candidate의 변환된 result 목록과 비교하며, 변환된
signature를 still-unconverted operation과 맞대지 않는다.

### Entrypoint contract

The frontend accepts `main` only when its declared result is `Nil`. A frontend
error is terminal, so the Wasm backend never receives a valid program whose
`main` returns an `Int`, `Nat`, or another user value. The generated `_start`
function therefore calls `main` for its side effects. A pure `main` is called
directly; a `main ->{Io} Nil` receives target-provided initial evidence through
the `EvidenceDirect` ABI. Other residual effects are rejected by the frontend.
Printing program results belongs in explicit standard I/O calls such as
`std::io::print_line`, which shared lowering maps to the target-independent I/O
boundary described in [io.md](io.md), not in backend entrypoint lowering.

CPS root가 필요한 export는
[직접형 wrapper와 completion cell](cps-effects.md#root-main-delimiter)을 사용한다.
Shared IR은 completion cell과 `core.never` root `done_k`의 추상 조합 계약만
보존한다. Wasm signature lowering이 CPS signature를 empty-result signature로
내린 뒤 nil/void machinery에 맞는 wrapper와 ordinary call을 합성한다. Wrapper는 root
`done_k`가 typed cell을 쓴 뒤 call이 돌아오면 이를 읽는다. Shared `func.call`은
zero-result 형상을 지원하며, 이 bridge는 trampoline이나 `anyref` control
carrier가 아니다.

### 올바른 꼬리 호출 계약

[WebAssembly 3.0 validation](https://webassembly.github.io/spec/core/valid/instructions.html#valid-return-call-indirect)은
`return_call`, `return_call_indirect`, `return_call_ref`의 tail-call 형식과 caller
result matching을 정의한다. Tribute는 다음 경로를 구현한다:

| Shared IR | Wasm IR | Encoder |
| ---- | ---- | ---- |
| `func.tail_call` | `wasm.return_call` | `wasm_encoder::Instruction::ReturnCall` |
| `func.tail_call_indirect` | `wasm.return_call_indirect` | `wasm_encoder::Instruction::ReturnCallIndirect { type_index, table_index }` |

`func.tail_call_indirect` lowering은 callee table index와 argument를 기존
`call_indirect`와 같은 순서로 평가하고, callee `func.func_sig`에서 `type_index`를
결정한다. CPS caller/callee의 result vector는 모두 비어 있어야 하며
`wasm.return_call_indirect`는 result local을 만들지 않는다. 일반 source-data
indirect call만 `wasm.call_indirect`를 유지한다.

### Dynamic basic output

WASI preview1 `fd_write` cannot read a WasmGC `Bytes` array directly because its
iovec points into linear memory. The initial `tribute_io.write` lowering copies
the dynamic `Bytes` slice into an instance-local linear scratch buffer, appends
the optional newline there, and invokes `fd_write` with compiler-owned iovec and
`nwritten` cells. The lowering grows memory when required and retries partial or
interrupted writes. See [io.md](io.md#wasm-runtime-boundary) for lifetime and
failure rules.

Wasm output uses only `tribute_io.write`. The former `__print_line` literal
analysis and its `i32` pointer plus `literal_len` representation are not part of
the backend boundary; string literals remain canonical `String` values until
the standard-library I/O wrapper explicitly converts them to `Bytes`.

### Private List layout

WasmGC must lower the same representation-independent `list.*` sequence
operations to a target-private GC layout and eliminate them before the
backend-ready boundary. It is not required to share native's linked-node/null
layout. Native implementation and shared frontend evidence do not by themselves
establish Wasm compilation or execution support; capability claims require
focused Wasm evidence.

---

## Emission 경계

### Bodyless 선언의 처분

Backend-ready 경계에는 본문 없는 `wasm.func` 선언이 남을 수 있다. 이는 target call
rewriting 이후에도 유지되는 ordinary C helper 선언과, 호출이 제거된 등록 compiler
intrinsic 선언이다. Wasm module은 import가 아닌 모든 function에 code entry와 body를
요구하므로 bodyless 선언은 explicit import로만 emit될 수 있다.

Emitter는 IR을 수정하지 않는 read-only 처분으로 이 상황을 해결한다.

- 함수 심볼 참조는 마지막 helper rewrite 이후의 최종 IR에서 새로 수집한다. 이전
  단계의 resolved-reference 사실이나 cached 분석을 재사용하지 않으므로, 오래된
  사실이 살아있는 참조를 가리거나 무참조 선언을 잘못 남기지 않는다.
- 참조 모델은 최종 경계에 허용된 연산만 다룬다. `wasm.call`과
  `wasm.return_call`의 `callee`, `wasm.ref_func`의 `func_name`,
  `wasm.export_func`의 `func`가 함수 심볼을 지칭한다. `wasm.elem`은 자식 `funcs`
  region을 재귀 순회해 그 안의 `wasm.ref_func`를 element segment 참조로 분류한다.
  Container 연산이 스스로 심볼 속성을 갖지 않는다는 사실은 자식 참조 사용을
  부정하지 않는다.
- 모델에 없는 연산이 함수 심볼 속성을 소유하거나, 모델에 있는 연산의 심볼 속성이
  없거나 malformed면 silent non-user로 넘기지 않고 거부한다.
- 참조가 없는 well-formed bodyless 선언만 emitted definition/type-index/code 목록에서
  제외한다. 본문 없는 선언을 남기는 것 자체는 오류가 아니다.
- 살아남은 참조의 목적지는 import-first 인덱싱을 유지한 명시적 `wasm.import_func`의
  `sym_name` 또는 본문 보유 `wasm.func`여야 한다. 그렇지 않으면 bodyless 선언과
  미해결 참조를 각각 정확한 진단으로 구분해 보고한다.
- 본문이 있는 C 함수와 명시적 import는 그대로 emit된다. body region은 있으나 entry
  block이 없는 malformed `wasm.func`는 bodyless 선언이 아니므로 계속 거부한다.

이 처분은 공통 DCE reachability나 인증된 intrinsic 삭제와 별개다. IR을 변경하지
않고 emission 목록만 좁히며, 대상 심볼 삭제를 다른 pass에 위임하지 않는다.

---

## WasmGC 타입 처리

### Backend에서 타입 수집

`trunk-ir-wasm-backend`는 builtin GC type layout을 먼저 예약하고, `wasm.*`
연산들에서 user type 정보를 수집한다:

```rust
// wasm.struct_new 연산에서 타입 정보 추출
// @Point 타입과 필드 타입들을 수집
%p = wasm.struct_new @Point (%x: f64, %y: f64) : ref<@Point>

// wasm.array_new에서 배열 타입 정보 추출
%arr = wasm.array_new @IntArray (%len) : ref<@IntArray>
```

### Type Section 생성

수집된 타입 정보로 WasmGC type section을 생성한다. Builtin layout은 다음과 같고
user-defined type은 그 뒤에 배치된다:

| Index | Type |
| ---: | --- |
| 0 | `BoxedF64` |
| 1 | `BytesArray` |
| 2 | `BytesStruct` |
| 3 | `_closure { table_idx: i32, env: anyref }` |
| 4 | `_Marker { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref }` |
| 5 | `Evidence` array |
| 6+ | user-defined structs, arrays, variants, closures |

이 표는 backend-ready builtin layout의 규범적 최종 계약이다. Emitter와 layout
verifier는 `_closure` 3, `_Marker` 4, `Evidence` 5, user-defined type 6+를 정확히
사용하며 CPS control carrier나 trampoline placeholder index를 예약하지 않는다.
`_closure` environment와 Marker의 dispatch closure field는 일반 reference
erasure이므로 계속 `anyref`를 사용할 수 있다.

```wasm
;; 생성된 type section 예시
(rec
  (type $Node (struct (field i32) (field (ref null $Node)))))
(type $Point (struct (field f64) (field f64)))
```

### 물리적 참조 할당 가능성

WasmGC의 서브타이핑은 non-coercive이고 concrete struct 타입은 `struct`의
서브타입이므로, concrete struct 참조는 추상 참조 슬롯에 런타임 캐스트 없이
대입된다. 인자, 결과, 정확 indirect/tail 시그니처 경계는 모두 하나의 물리적
할당 가능성 관계를 공유하며, 그 관계는 다음 확장만 인정한다.

| 값 타입 | 슬롯 | 판정 |
| --- | --- | --- |
| builtin 레이아웃 인덱스를 갖는 struct (`core.bytes`, `_closure`, `_Marker` 등) | `structref`, `anyref` | 허용 |
| `adt.typeref` | `structref`, `anyref` | 허용 |
| `base_enum`을 가진 concrete variant instance | `structref`, `anyref` | 허용 |
| builtin 배열 레이아웃 (Bytes backing array, Evidence array) | `arrayref`, `anyref` | 허용 |
| `core.array` | `arrayref`, `anyref` | 허용 |
| 등록 근거가 없는 ADT 표기 (선언 타입 등) | `structref` | 거부 |
| `anyref` | `structref` | 거부 (downcast) |
| `arrayref`, `funcref`, `externref`, `i31ref` | `structref` | 거부 |
| `structref` 또는 등록된 struct | `arrayref` | 거부 |

여기서 "등록"은 backend-ready 경계에서 해당 타입이 concrete GC 인덱스를
받는지를 뜻한다. 근거가 되는 것은 builtin 레이아웃 인덱스, `adt.typeref`,
그리고 `base_enum`을 가진 variant instance뿐이며, ADT 이름이나 레이아웃
모양만으로는 등록을 추론하지 않는다. 추상 참조에서 concrete 타입으로
좁히는 방향은 `wasm.ref_cast`가 필요하므로 검증에서 거부한다.

---

## 설계 결정 배경

### GC 관련 타입을 trunk-ir에 추가하지 않는 이유

Cranelift 팀의 교훈 참고 ([Stack Maps 문서](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)):

> IR 코어에 GC 참조 타입을 넣으면 복잡해진다. Frontend가 처리하는 게 낫다.

Cranelift는 초기에 GC 참조를 IR 전체에서 추적했으나, 다음 문제 발생:

- 전용 참조 타입이 최적화 방해
- Mid-end에서 safepoint spill/reload가 보이지 않아 버그 발생
- 복잡성 증가

해결책: "User Stack Maps" - frontend가 GC 관련 처리를 담당

**Tribute에서의 적용:**

- trunk-ir에 GC 관련 dialect 추가하지 않음 (gc, gc_type 등)
- WasmGC-specific 개념 (type indices, builtin type layout, ref/nullability)은
  백엔드에서 처리
- trunk-ir는 target-independent하게 유지

### wasm dialect의 역할

wasm dialect는 WasmGC 인스턴스 연산만 포함:

- `wasm.struct_new`, `wasm.struct_get`, `wasm.struct_set`
- `wasm.array_new`, `wasm.array_get`, `wasm.array_set`
- 기타 Wasm 명령어들

타입 정의 (type section)는 백엔드가 이 연산들에서 추론하여 생성한다.

### 크레이트 역할 분담

역할 분담은 다음과 같다:

- Lowering → tribute-passes
- Emission → trunk-ir-wasm-backend
- 조율 → tribute main crate

최종 Wasm emission boundary는 residual `tribute_control.*`, `ability.*`,
`effect.*`, CPS control carrier, trampoline과 result-producing CPS transfer를
거부한다.

---

## References

- [Wasm 3.0 Release](https://webassembly.org/news/2025-09-17-wasm-3.0/)
- [WebAssembly 3.0 tail-call validation](https://webassembly.github.io/spec/core/valid/instructions.html#valid-return-call-indirect)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md)
- [Cranelift Stack Maps](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
