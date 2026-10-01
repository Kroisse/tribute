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
│                         # 경계 안: effect.* → helper 호출 + call/tail_call
│                         # 출구 뒤: helper 선언을 GC 배열 구현과
│                         #          prompt tag 카운터로 바인딩
├── wasm/tribute_rt_to_wasm.rs
├── wasm/const_to_wasm.rs
├── wasm/bytes.rs         # bytes layout 타입과 경계 안 bytes 읽기 intrinsic lowering
├── wasm/intrinsic_to_wasm.rs
│                         # 출구 뒤: extern "C" bytes helper를 GC 연산으로 바인딩
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
├── wasm.func             # evidence runtime helper 구현
│
▼ trunk-ir-wasm-backend
│   (gc_types_collection: wasm.* ops에서 타입 수집)
│   (builtin GC type layout + user type collection)
│
WebAssembly Binary
```

Effect lowering은 target별로 수행한다. Shared ability lowering은 `effect.*`를
만들며 Marker field 번호나 closure layout을 검사하지 않는다.
`wasm/evidence_to_wasm`은 [representation/ABI 경계](ir.md#representationabi-경계)
안에서 `effect.*`를 공유 value/control dialect로 낮춘다. Evidence 조회와 확장은
native와 같은
[evidence runtime ABI](cps-effects.md#handle-evidence-extension--handler-closures)의
몸체 없는 helper 선언을 `func.call`로 호출한다. Canonical closure layout은 `adt.struct_get`으로
풀고, semantic role에 맞게 `func.call_indirect` 또는 `call_conv = @tail`
signature의 proper-tail `func.tail_call_indirect`를 만든다. 이 lowering은 `wasm.*`
operation을 만들지 않고 `tribute.calling_convention`을 읽거나 쓰지 않는다.

Helper 구현은 target이 제공하는 runtime이다. 출구 뒤 Wasm lowering은 참조된
몸체 없는 helper 선언을 GC 배열 위의 이진 탐색 구현으로 바꾼다. Marker 생성과
field 접근은 이 구현 안에만 있다. Wasm dialect lowering은 `effect.*`를 보지 않는다.

### Wasm 결과 슬롯

`wasm.func_sig`의 저장 형식과 간접 호출의 명시적 시그니처는
[IR 호출 계약](ir.md#wasmfunc_sig-wasm-호출-계약)을 따른다.

바이너리 코드 생성은 결과 위치의 `core.nil`만 생략하고 나머지 결과의 선언 순서를
보존한다. 생략되지 않는 각 결과에는 SSA 값과 지역 변수를 할당한다. 다중 결과
호출 후에는 마지막 결과가 스택 맨 위에 있으므로 지역 변수에 역순으로 저장한다.
입력이나 값 위치의 `core.nil`은 생략하지 않고 널을 허용하는 참조로 표현한다.
모든 `wasm.call_indirect`와 `wasm.return_call_indirect`는 exact `wasm.func_sig`를
명시한다. Signature가 없으면 lowering과 emission이 실패하며 operand/result 타입,
상위 함수의 return 또는 `type_idx`로 대체 시그니처를 추론하지 않는다.
Wasm 코드 생성기는 함수 본문이나 테이블 인덱스에서 CPS 여부를 추론하지 않는다.

### Terminal structured-control 결과

`scf_to_wasm`은 결과가 정확히 하나의 `core.never`이고 사용되지 않으며 블록의
마지막 operation인 `scf.if`와 `scf.loop`를 zero-result Wasm 제어 연산으로
낮춘다. 모든 진입 successor의 region은 단일 블록이고, 마지막 operation이
검증된 `CallableExit` 또는 같은 terminal 판정을 만족하는 중첩 if/loop/switch여야
한다. Native structured-to-CFG lowering과 같은 `RegionBranch`/`CallableExit`
판정을 사용하며, parent 복귀, 불완전한 interface 응답, 오류는 terminal 증거가
아니다. Loop의 continue 순환이나 임의의 다중 블록 CFG는 분석하지 않는다.

Terminal if는 결과 없는 `wasm.if`, terminal loop는 결과 없는 `wasm.block`과
`wasm.loop`가 된다. Resultless switch도 블록 마지막에 있고 explicit default를
포함한 모든 arm이 terminal이면 결과 없는 Wasm 비교 분기들을 만든다. Source
switch에 결과를 추가하지 않으며 일반 fallthrough switch의 계약은 유지한다.
Wasm 검증기는 각 structured block의 `end` 뒤에 branch 내부의 도달 불가능성을
전파하지 않는다. 따라서 terminal로 증명된 zero-result if/loop/switch 뒤에는
명시적인 `wasm.unreachable`을 둔다. 결과가 처음부터 없는 terminal 제어 연산에도
같은 규칙을 적용하며, 정상 successor가 있는 제어 연산에는 적용하지 않는다.

중첩 pattern이 source operation을 바꾸기 전에 공통 `StructuredControlAnalysis`를
조회한다. Wasm 소비자는 switch 지원 조건과 `Never` 결과의 적법성을 검사하고,
`Never` 결과 제거와 terminal switch의 결과 없는 분기 생성을 별도 변환 결정으로
기록한다. 분석 자체는 Wasm 지원 조건을 검사하거나 conversion 오류를 만들지
않는다. Pipeline 진입점의 사전 검사와 SCF lowering 사이에 다른 pass가 실행되면
분석을 다시 계산하고, 변환 직전에는 캐시를 무효화한 뒤 결정만 pattern에 전달한다.
사용 중이거나 terminal 증명이 안 되는 `Never` 결과는 mutation 전에 거부한다.
결과 제거 API는 모든 기존 결과가 미사용이고 새 결과가 비어
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
`main` returns an `Int`, `Nat`, or another user value. Entry bridge composition
inside the [representation/ABI boundary](ir.md#representationabi-경계) turns
every root `main` into a worker and synthesizes a parameterless, unreferenced
wrapper `main` that supplies the worker's convention-specific inputs, such as
target-provided initial evidence. Wasm lowering exports that wrapper directly
as the WASI command entry `_start`; it builds no separate start function, never
reads a semantic calling convention, and exports no `main`. Other residual
effects are rejected by the frontend.
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

### Fresh prompt 바인딩

새 handler delimiter의 prompt 생성은 `__tribute_next_tag` 호출을 요구한다.
Wasm target은 경계 출구 뒤에 이 C helper 선언을 함수 본문으로 바인딩한다. 본문은
module-level mutable `i32` global 하나를 카운터로 쓰며, 현재 값을 반환하고 1을
더해 저장한다. 따라서 tag는 native runtime과 같이 0부터 호출 순서대로 발급된다.
카운터 global은 기존 global 뒤에 추가하므로 기존 global index를 바꾸지 않는다.
Wasm target이 구현을 제공하지 않는 C helper를 참조하면
[representation/ABI 경계](ir.md#representationabi-경계)의 출구 검증이 모듈을
거부한다. 아래의 bodyless 선언 규칙은 경계를 우회한 입력에 대한 emission의 마지막
방어선이다.
Wasm source handler 지원은 shared/native handler 실행 지원이 아니라 Wasm 실행
증거로 판정한다.

### Tail-resumptive dispatch의 함수 시그니처

`effect.dispatch_tail`의 Wasm lowering은 `(Evidence, env: anyref, op_idx: i32,
payload: anyref) -> anyref` 시그니처를 명시한다. Evidence와 반환 표현은 정확히
일치해야 하며 packed payload는 검증된 GC widening만 허용한다. Lowering은
operand로 ABI를 재추론하지 않고 이 고정 시그니처를 `wasm.call_indirect`에
보존한다. Emitter는 최종 모듈 등록 결과에서 함수 타입 인덱스를 얻으며,
lowering의 placeholder `type_idx`가 그 결과를 덮어쓰지 않는다.

### Nil 값과 callable result의 구분

`Nil`을 값으로 생성하는 Wasm 연산은 `ref.null none`을 포함한 실제 stack 값을
만들며, emitter는 그 값을 해당 SSA local에 저장한다. 사용되지 않은 Nil 값도
stack에 남겨 두지 않는다. 반면 callable의 Nil result slot은 기존 ABI에서
생략하므로 call 결과를 저장할 때만 그 slot을 건너뛴다. 이 구분은 Nil을 인자나
필드로 전달하는 것과 zero-result call을 모두 보존한다.

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
결정한다. Tail transfer의 검증은 callee signature의 결과 목록이 둘러싼 함수의
결과 목록과 호환되는지만 본다. 의미적 호출 규약이나 signature의 `call_conv`로
판정하지 않는다. Wasm signature는 공통 signature의 입력과 결과 타입만으로 만들어
지므로 `call_conv`는 `type_index`에 영향을 주지 않는다.
`wasm.return_call_indirect`는 result local을 만들지 않는다. 일반 source-data
indirect call만 `wasm.call_indirect`를 유지한다.

### 모듈 수준 자원의 선언

Operation을 낮추는 단계는 그 결과가 의존하는 모듈 수준 자원을 같은 단계에서 IR에
선언한다. 자원은 passive data segment, host import, linear memory이며, 같은 자원이 이미
선언되어 있으면 재사용한다. 뒤의 단계와 모듈 조립은 IR에 선언된 자원만 읽는다. 앞
단계의 입력에서 정한 배치 결정을 IR 밖으로 넘겨받지 않는다. Data index는 모듈 안
`wasm.data` operation의 순서이다. 모듈 조립은 선언된 memory를 export하고 entrypoint
export를 추가할 뿐, 자원을 새로 계획하지 않는다.

### Dynamic basic output

WASI preview1 `fd_write` cannot read a WasmGC `Bytes` array directly because its
iovec points into linear memory. The initial `tribute_io.write` lowering copies
the dynamic `Bytes` slice into an instance-local linear scratch buffer, appends
the optional newline there, and invokes `fd_write` with compiler-owned iovec and
`nwritten` cells. The lowering declares the `fd_write` import and the linear
memory that holds these cells, grows memory when required, and retries partial or
interrupted writes. See [io.md](io.md#wasm-runtime-boundary) for lifetime and
failure rules.

Wasm output uses `tribute_io.write`. String literals remain canonical `String`
values until the standard-library I/O wrapper explicitly converts them to
`Bytes`.

### Private List layout

WasmGC must lower the same representation-independent `list.*` sequence
operations to a target-private GC layout and eliminate them before the
backend-ready boundary. Concrete layout과 메모리 관리는 Wasm target이 소유하며
native의 RC RRB tree 표현과 독립적이다.

---

## Emission 경계

### Bodyless 선언의 처분

Backend-ready 경계에는 본문 없는 `wasm.func` 선언이 남을 수 있다. 이는 target call
rewriting 이후에도 유지되는 ordinary C helper 선언과, 호출이 제거된 등록 compiler
intrinsic 선언이다. Wasm module은 import가 아닌 모든 function에 code entry와 body를
요구하므로 bodyless 선언은 explicit import로만 emit될 수 있다.

Shared pre-CPS 단계는 managed logical parameter/result를 가진 `extern "C"`도
trusted/unsafe FFI 선언으로 허용한다. 이는 native ABI나 C linkage를 Wasm에 제공한다는
뜻이 아니다. Wasm target은 logical signature를 Wasm 타입으로 변환하고, 남은 참조에는
그 target signature와 호환되는 명시적 import 또는 본문을 요구한다. `abi = "C"`만으로
import나 managed reference adapter를 합성하지 않는다. 바인딩 없는 선언은 미참조일 때만
emission에서 제외하며, 참조가 남으면 오류다. Native의 borrowed/fresh-owned RC 계약을
Wasm host ABI로 그대로 적용하지 않으며, 명시적 host 바인딩은 해당 Wasm reference
representation을 지켜야 한다.

선언·정의·malformed 분류는 [공통 본문 구조](ir.md#callable-본문-구조)를 따른다.
Emitter는 malformed를 먼저 거부한 뒤 IR을 수정하지 않는 read-only 처분을 수행한다.

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
- 본문이 있는 C 함수와 명시적 import는 그대로 emit된다. `wasm.func`의 `abi`는
  `wasm.import_func` 바인딩을 대신하지 않으며, 본문 구조 판정에도 참여하지 않는다.

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
verifier는 closure 3, marker 4, evidence 5, user-defined type 6+를 정확히
사용하며 CPS control carrier나 trampoline placeholder index를 예약하지 않는다.
Index 1은 `@bytes_data`, index 2는 `@bytes`, index 3-5는 `@closure`,
`@evidence_marker`, `@evidence`
[runtime layout 식별자](ir.md#runtime-layout-식별자)로만 정해진다. 경계 출구의
명목 타입 `core.bytes`는 Wasm lowering의 첫 단계에서 `@bytes` layout struct로
바뀐다. 이 변환은 alias, 연산 속성, 결과와 block 인자뿐 아니라 ADT field, variant
payload, signature처럼 다른 타입 안에 들어 있는 `core.bytes`까지 구조적으로 바꾼다.
이후 단계가 만드는 bytes 값도 `@bytes` struct 타입을 가지므로, 그 뒤의 Wasm IR과
backend에는 `core.bytes`가 나타나지 않는다.
Struct 이름이나 원소 타입이 같더라도 식별자가 없는 타입은 builtin layout이 아니다.
원소가 `core.i8`인 배열도 `@bytes_data`가 없으면 bytes 배열이 아니다.
`_closure` environment와 Marker의 dispatch closure field는 일반 reference
erasure이므로 계속 `anyref`를 사용할 수 있다.

```wasm
;; 생성된 type section 예시
(rec
  (type $Node (struct (field i32) (field (ref null $Node)))))
(type $Point (struct (field f64) (field f64)))
```

### GC 인덱스 등록의 소유권

GC 연산의 concrete `type_idx`는 해당 연산이 접근하는 레이아웃을 지정한다.
수집기는 이를 근거로 입력·결과의 추상 Wasm 참조 타입을 concrete 타입으로
전역 등록하지 않는다. `structref`·`anyref` 인자와 필드는 다른 함수의
projection, 생성 또는 참조 연산의 등장 순서와 무관하게 추상 타입을 유지한다.
Concrete nominal 타입의 등록과 연산별 narrowing cast는 그대로 유지한다.
Evidence 배열은 type 변환 뒤에도 `layout = @evidence`를 가진 타입으로 emission까지
남고, 그 식별자로 index 5를 받는다. Erased `arrayref`는 evidence로 간주하지 않는다.

### GC struct 필드의 scalar 표현

GC struct 필드 수집은 타입 비교와 저장 전에 Wasm의 물리적 scalar 표현을
정규화한다. `core.i1`과 `core.i32`는 모두 Wasm `i32` 필드로 표현하므로,
생성·읽기·쓰기에서 두 타입이 섞이더라도 수집 순서와 무관하게 `i32`를 사용한다.
이 동등성은 Bool의 target 표현에 한정한다. `i64`, 부동소수점, 참조 타입의
불일치를 같은 크기나 비슷한 레이아웃으로 허용하지 않는다.

### 좁은 정수 표현

Wasm에는 `i8`과 `i16` 값 타입이 없다. `core.i8`과 `core.i16` 값은 `i32`에 담고
상위 비트는 정하지 않는다. 값을 읽는 쪽이 필요한 만큼 정규화한다.

- 상위 비트가 결과의 하위 비트에 영향을 주지 않는 연산(`addi`, `subi`, `muli`,
  `and`/`or`/`xor`, `shl`, 상수)은 정규화 없이 `i32` 명령으로 낮출 수 있다.
- `extui`는 폭만큼의 mask(`i32.and`)로, `extsi`는 `i32.extend8_s`/`i32.extend16_s`로
  정규화한다. 결과가 `i64`면 그 뒤에 `i64.extend_i32_u`/`i64.extend_i32_s`를 둔다.
  `trunci`는 `i64` 입력이면 `i32.wrap_i64`로, `i32` 입력이면 결과 폭의 mask로
  낮춘다.
- 상위 비트가 결과에 영향을 주는 연산(`cmpi`, `divsi`/`divui`, `remsi`/`remui`,
  `shr`/`shru`, `sitofp`/`uitofp`)은 좁은 정수에서 낮추지 않는다. 이 연산은
  target 변환 경계에서 거부된다.
- Packed 배열(`core.array<core.i8>`, `core.array<core.i16>`)의 원소 읽기는
  `array.get_u`로 낮춘다. 상위 비트를 정하지 않으므로 `array.get_s`도 맞지만 하나로
  고정한다. 쓰기는 `array.set`이 하위 비트만 저장한다.

`core.i1`은 이 규칙의 대상이 아니다. Bool은 `i32`의 0 또는 1로 정규화된 값이다.

### 물리적 참조 할당 가능성

WasmGC의 서브타이핑은 non-coercive이고 concrete struct 타입은 `struct`의
서브타입이므로, concrete struct 참조는 추상 참조 슬롯에 런타임 캐스트 없이
대입된다. 인자, 결과, 정확 indirect/tail 시그니처 경계는 모두 하나의 물리적
할당 가능성 관계를 공유하며, 그 관계는 다음 확장만 인정한다.

| 값 타입 | 슬롯 | 판정 |
| --- | --- | --- |
| builtin 레이아웃 인덱스를 갖는 타입 | 같은 인덱스를 갖는 다른 표기 | 허용 |
| builtin 레이아웃 인덱스를 갖는 struct (`@bytes`, closure, marker 등) | `structref`, `anyref` | 허용 |
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
