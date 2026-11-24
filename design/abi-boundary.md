# Representation/ABI 경계

Compiler pipeline 단계의 불변 조건과 target representation/ABI 경계의 계약이다.

## 단계 불변 조건

```text
source
  -> parse / AST
  -> name resolution
  -> type checking
  -> AST-to-IR (tribute_control callable/control + ordinary value IR)
  -> shared CPS legalization
  -> shared lowering and optimization
  -> Wasm or native lowering
  -> backend-ready full conversion target
  -> emit
```

이 도식은 단계의 큰 흐름만 보인다. 구체적인 pass 순서는
[lowering.md](lowering.md#pipeline-order)가 정한다.

Important stage invariants:

| Stage | Required invariant |
| ---- | ---- |
| Resolution | Names, constructors, and variable references are resolved |
| Type check | Type variables and ability rows are solved; method-style calls are converted to resolved calls |
| AST-to-IR | `tribute_control.func_sig`과 callable/control operation, valid SSA use chain |
| Shared CPS legalization | 전체 callable graph가 physical `CallableAbi`와 direct/indirect tail transfer를 사용하며 `tribute_control` operation/type이 남지 않음 |
| Shared lowering | 명시된 경계에서 high-level ability dispatch operation이 제거됨 |
| Effect ABI | `effect.*` operations preserve dispatch semantics without backend layout details |
| Native ownership | Typed managed value와 `core.ptr` 사이의 이전은 ownership unit 하나를 소비하는 `tribute_rt.into_raw`와 unit 하나를 만드는 `tribute_rt.from_raw`로만 허용되며 `core.ptr`는 항상 unmanaged |
| Native RTTI | Ownership planning이 선언한 `tribute_rtti.layout`이 RC header lowering까지 모든 할당 layout을 정확히 한 번씩 선언하며, RC header lowering 뒤에는 남지 않음 |
| Representation/ABI 경계 | 아래 [경계 계약](#representationabi-경계)의 출구 적법성을 만족함 |
| Backend lowering | Backend-ready 검증이 성공하고 `effect.*`, `tribute_rt.into_raw`, `tribute_rt.from_raw`, CPS control carrier, trampoline이 남지 않음 |

<!-- markdownlint-disable-next-line MD033 -->
<a id="representationabi-경계"></a>

## Representation/ABI 경계

Representation/ABI 경계는 상위 제어 의미를 물리적 호출·저장·runtime 계약으로
소비하는 target별 단계다. 이 단계는 shared middle-end가 끝난 뒤 시작하고 target
dialect lowering(`func_to_clif`, `func_to_wasm` 등) 앞에서 끝난다. 경계 이후의 모든
pass — native ownership/RTTI 계획, target dialect lowering, backend 검증기, emitter —
는 출구가 명시한 물리 계약만 소비한다.

### 단계 구성

경계는 target별로 여러 pass로 이루어질 수 있으며 새 dialect, crate 또는 모든 target에
같은 layout을 요구하지 않는다. 경계 안에서 다음을 순서대로 완결한다.

1. Exact callable/dispatch/frame 계약 검증과 CPS signature 물리화
2. Root/export 및 target 진입점 bridge 합성
3. Closure operation lowering
4. Target evidence lowering: `effect.*` dispatch, evidence 조회·확장, fresh prompt,
   one-shot 검사를 runtime 호출과 ordinary/proper-tail transfer로 바꾼다. 이
   lowering은 공유 value/control dialect 수준에서 수행하고 target dialect operation을
   직접 만들지 않는다.
5. Closure storage layout 확정
6. [표현 선택](#표현-선택): boxed 타입의 `adt` 값을 `heap.box`로 바꾸고, unboxed 값은
   `adt` 값과 inline 필드로 둔다.
7. [Object 접근 fusion](tribute-dialects.md#object-접근-fusion): 값 수준으로 남은
   box 접근을 target이 고른 `heap` object operation으로 합친다.
8. 경계 출구 검증

지원되는 FFI/intrinsic bridge의 의미도 이 단계 안에서 완결한다.

### 표현 선택

[값의 표현은 의미에 드러나지 않으므로](runtime-types.md#표현은-의미에-드러나지-않는다)
compiler가 값마다 표현을 고른다. 표현 선택은 값 의미론인 [`adt`](tribute-dialects.md#adt)
값을 target에 공통인 저장 표현으로 바꾸는 단계이며, MLIR이 tensor 값을 buffer로
바꾸는 bufferization과 같은 자리다. 두 target이 같은 규칙으로 고른다.

- **표현은 타입마다 고른다.** Struct 타입(compiler가 만든 tuple layout 포함)마다
  unboxed와 boxed 중 하나를 고르며, 한 타입의 값은 아래 강제 위치를 빼면 어디서나
  같은 표현을 쓴다. 이것은 선택 정책이며 계약이 아니다. 계약은 표현이 의미에 드러나지
  않는다는 것과, 어느 쪽이든 layout이 그 타입의 layout 하나라는 것뿐이다
  ([같은 소스 타입을 서로 다른 layout으로 표현하는 일은 없다](runtime-types.md#descriptor-단위)).
  사용 위치마다 표현을 고르는 일은 이 계약을 바꾸지 않고 더할 수 있다.
- **Unboxed 값은 `adt` 값으로 남는다.** 표현 선택은 unboxed 타입의 값과 그 operation을
  바꾸지 않는다. 다른 object의 필드로 저장할 때는 그 object의 layout 안에 inline으로
  놓이며, `heap` operation은 그 필드를 `adt` 값 하나로 읽고 쓴다. 값을 leaf로 쪼개는
  일은 [target lowering](tribute-dialects.md#target-lowering의-unboxed-값)이 한다.
- **Boxed 값은 `heap.box`에 담는다.** 표현 선택은 boxed 값의 operation을 값 수준으로
  남긴다. 생성과 갱신의 결과는 `heap.box_value`로 box에 담고, box를 읽거나 갱신의
  source로 쓸 때는 `heap.unbox`로 꺼낸다. 이 접근을 box를 빌린 읽기, `heap` 할당,
  source box를 소비하는 `heap` 갱신으로 합치는 일은
  [object 접근 fusion](tribute-dialects.md#object-접근-fusion)이 한다.
- **Cell은 표현 선택의 대상이 아니다.** Compiler가 쓰는 [`heap.cell`](tribute-dialects.md#heapcell)은
  표현 선택 전부터 object다. Cell이 담은 값은 다른 object의 필드처럼 저장 표현을
  따르므로, unboxed 값은 cell 안에 inline으로 놓인다.
- **강제 위치에서는 언제나 box에 담는다.** Unboxed로 고른 타입도 이 위치를 지날 때는
  필드 값으로 object를 할당하고, 거기서 나올 때는 필드를 읽는다. 단, box에서 꺼낸
  값을 바꾸지 않고 같은 layout의 box에 다시 담으면 새로 할당하지 않고 원래 box를
  retain해 쓴다(box forwarding). 값 수준에서는 `heap.unbox` 결과를 그대로 담는
  `heap.box_value`가 그 box가 된다. Retain으로 공유된 object는 유일하지 않으므로 이후의
  `heap.struct_update`와 `heap.array_set`은 복사 경로를 타고, 공유된 box가 담은 값은
  바뀌지 않는다. 표현 선택은 값이 box에서 나온 출처를 알고 있는 동안
  이 규칙을 적용한다.
  - 배열의 원소.
  - `tribute_rt.anyref`로 지워지는 자리. Effect payload, 제네릭으로 지운 값, Cps
    convention의 결과가 여기에 든다. Cps 결과는 continuation을 지나며 지워지므로, 결과를
    쪼개는 일은 Direct와 EvidenceDirect callable에만 해당한다.
- **언제나 boxed인 값.** Enum 값, 배열 값, closure 값, continuation frame은 box에
  담는다. 재귀하는 struct도 그렇다. 자기 필드를 통해 자기에게 닿는 layout은 크기가 정해지지 않기 때문이다.
- **선택 정책.** 강제로 boxed가 아닌 struct 타입 중에서 다음을 모두 만족하는 타입을
  unboxed로 고른다.
  - 펼친 뒤의 managed leaf가 하나 이하다.
  - 펼친 뒤의 크기가 compiler가 정한 상한 이하다.

  Managed leaf가 하나 이하이므로 unboxed 값을 복사할 때의 retain 수는 box를 복사할
  때와 같고, 할당만 사라진다. Ownership 계획은 unboxed 값을 unit 하나 이하인 값으로
  보므로, 선택 정책이 바뀌어도 첫 조건은 남는다. 상한의 값은 compiler 내부 상수이며 이 계약에 들지 않는다.
  Compiler는 값이 대부분 강제 위치를 지나는 타입처럼 boxed가 나은 타입을 boxed로
  남길 수 있다.
- **할당 제거는 별도의 최적화다.** Boxed 타입이라도 함수 밖으로 나가지 않는 box는
  표현 선택과 fusion 사이의 최적화가 필드 값으로 바꿀 수 있다. 이 구간의 box 접근은
  값 수준이므로, 이 최적화는 `heap.box_value`와 `heap.unbox`의 쌍을 지우는 일이다.
  타입의 표현을 바꾸지 않으며, 그 box가 가리킬 object가 관찰되지 않는다는 사실만 쓴다.
- **Ownership 계획보다 앞선다.** 표현 선택과 fusion은 경계 안에서 끝나므로 ownership 계획이
  보는 managed 값은 `heap.box`, `heap.cell`, `tribute_rt.anyref`, `tribute_rt.intref`와
  managed leaf를 가진 unboxed `adt` 값이다. 선택 정책상 unboxed 값의 managed leaf는
  하나 이하이므로, 그런 값은 unit 하나를 가진 managed 값이다.
- **필드 타입 변환.** 표현 선택은 layout의 필드 타입도
  [저장 표현으로 바꾼다](tribute-dialects.md#heapbox).

### 출구에서 확정되는 물리 계약

- **Exact signature:** 모든 정의·선언·직접 호출·간접 호출은 exact `func.func_sig`를
  가진다. Environment, evidence, continuation 인자는 이미 signature의 순서 있는 입력이며
  별도 slot 속성으로 위치를 기록하지 않는다. 타입이 있는 `func.constant`의 결과는
  참조 대상 함수의 signature와 정확히 같다. Closure의 호출자에게 보이는 signature처럼
  environment를 뺀 타입은 경계 안에서만 쓰며, 함수 참조의 타입으로 경계를 넘지 않는다.
- **기계 호출 규약:** 기계 호출 규약은 physical `func.func_sig`의 `call_conv` type
  속성이 소유한다. 경계는 CPS signature를 물리화할 때 target과 무관하게 그 signature에
  `call_conv = "tail"`을 일괄 부여하며, 속성이 없으면 platform 규약이다. `call_conv`는
  type identity에 참여하고 함수 정의, 직접 호출의 피호출자, 간접 호출 signature가 모두
  같은 signature에서 읽는다. 기계 규약을 구별하는 target은 이 값을 target signature로
  옮기고, 기계 규약이 없는 target은 이를 무시하고 target signature에서 버린다.
- **제어 이전의 세 가지 구별:** 빈 결과 목록은 machine stack에 결과가 없다는 뜻일
  뿐이다. Proper tail transfer는 `func.tail_call`/`func.tail_call_indirect` operation
  자체로 표현한다. 기계 수준 noreturn은 `func.unreachable` 같은 명시적 terminator로
  표현한다. 어느 하나에서 다른 것을 추론하지 않는다.
- **매개변수 ownership:** RC target에서 managed 매개변수의 entry mode
  (`borrowed`/`retained`/`consumed`,
  [rc.md](rc.md#proper-tail-ownership-transfer)) 중 callable 계약이 요구하는
  것은 exact physical signature의 일부로 표현한다. 그래서 직접 정의와
  exact indirect signature 모두에서 같은 계약을 읽을 수 있다. 인코딩은 signature의
  [타입 매개변수 속성](ir.md#타입-매개변수-속성) `tribute.ownership = "consumed"`이며,
  경계가 module 내부 물리 callable의 모든 입력에 붙인다. Platform `abi`를 가진
  callable은 표시를 받지 않는다. 표시가 없는 managed 매개변수는 retained 계약을
  가진다.
- **값의 표현:** 출구의 값은 scalar, `heap.box`, `heap.cell`, `tribute_rt.anyref`·`intref`,
  unmanaged 포인터 같은 기계 값이거나 unboxed 타입의 `adt` struct 값이다. Boxed 타입의
  `adt` 값은 없다. Unboxed 값은 object의 필드로는 inline으로 놓이며, target lowering이
  이를 leaf로 쪼갤 때 매개변수, 결과, block 인자에 나란히 놓인다. 그
  [scalarization](tribute-dialects.md#target-lowering의-unboxed-값)은 출구가 확정한
  exact signature와 매개변수 ownership 계약을 모든 정의와 호출에 똑같이 leaf 단위로
  옮길 뿐이며, 새 계약을 만들지 않는다.
- **Closure/frame 저장:** Compiler가 소유하는 runtime layout은 명시적
  [runtime layout 식별자](ir.md#runtime-layout-식별자)로 구별한다. Struct 이름, field
  모양, `arrayref` 같은 erased heap 형상을 provenance로 쓰지 않는다.
- **진입점:** 출구의 root `main`은 bridge 합성이 만든 wrapper다. Hidden 매개변수가
  없고 결과는 `Nil`이며 platform 규약을 따른다. Body는 root worker 호출 하나로 끝나고,
  모듈 안에서 이 `main`을 참조하는 곳은 없다. 원래 source `main`은 모든 calling
  convention에서 root worker가 되고, 모듈 안의 참조도 worker로 옮겨 간다. 초기
  evidence와 값 delimiter처럼 source calling convention에 따라 달라지는 부분은
  bridge 합성이 소비한다. Target은 이 wrapper를 그 자리에서 target 진입점으로 바꾼다.
  이때 runtime 초기화, 종료 코드, sanitizer 초기화처럼 platform 고유 작업만 더하고,
  별도 진입 함수를 만들거나 `main`의 이름을 바꾸지 않는다.
- **Runtime helper 바인딩:** 출구에서 참조가 남은 bodyless 선언은 명시적 바인딩
  의도를 가진다. `abi = "C"` 선언은 C linkage이므로 선언의 링크 이름이 곧 바인딩
  정체성이다. 이는 symbol 철자에서 의미를 추론하는 것이 아니라 FFI 선언이 명시한
  링크 이름이다. 각 target은 자신이 충족하는 C 바인딩을 명시적으로 선언한다. Native는
  runtime library와 링커가 C 이름을 해석하므로 모든 C 선언을 충족한다고 본다. Wasm은
  target이 구현을 제공하는 runtime helper 목록만 충족하며, 그 목록은 구현과 같은
  곳에서 정의한다. 참조되는 C 선언을 target이 충족하지 못하면 emission이 아니라 경계
  출구 검증이 거부한다. 최종 import 등록과 미참조 선언의 처분은 target emission이
  정한다.

### 출구 적법성

출구에 남으면 안 되는 것:

- `tribute_control.*`, `ability.*`, `effect.*`, `closure.*` operation과
  `closure.closure` type
- Callable 결과로서의 `core.never`. 논리 CPS 결과는 물리 결과 목록 `[]`로 바뀐다.
- 의미적 호출 규약과 제어 metadata: `tribute.calling_convention`,
  `tribute.cps_continuation_frame_result`
- 물리 계약으로 옮기지 않은 채 남은 handler/resume/prompt 정체성과 ability row
- Boxed 타입의 `adt` 값, `adt` enum·배열 값, 그 값 operation. 출구 뒤에 값을
  object에 담는 일이 생기면 ownership 계획이 보지 못한 managed 값이 생기므로, 표현
  선택과 fusion이 끝나지 않은 IR은 출구를 넘지 않는다. `heap.box_value`와
  `heap.unbox`도 남지 않는다. Literal operation도 표현 선택과 fusion이
  [`heap` literal](tribute-dialects.md#위치)로 바꾼다. Unboxed struct 값과 그
  operation(`adt.struct_new`, `adt.struct_get`, `adt.struct_update`)은 남는다.

출구 이후에도 보존하는 것:

- 기계 호출 규약, exact signature, 외부 바인딩(`abi`)
- `heap` operation이 layout을 읽는 `heap.box`의 layout, [nominal 선언](tribute-dialects.md#nominal-선언),
  명시적 layout 식별자, ownership/RTTI 입력(매개변수 속성 `tribute.ownership` 포함)과
  `tribute.types` 같은 타입 식별 metadata
- Location과 선언 위치 속성 `tribute.definition` 같은 실행에 관여하지 않는
  source/debug 정보

경계 이후 pass는 금지된 metadata를 조회하거나 다시 만들지 않는다. 이름, 포인터 형태,
arity, 빈 결과 목록에서 소실된 의미를 복원하지 않는다. 의미적 분류를 이름만 바꾼
물리 속성으로 복제하는 것도 허용하지 않는다.

출구 검증기는 operation, callable type, alias, 중첩 type 속성, block 인자와 그
속성을 재귀적으로 검사한다. 속성 값 안의 dictionary key도 깊이와 관계없이 같은
규칙으로 분류한다. 금지 목록뿐 아니라 보존 목록에도 없는 언어 전용
`tribute.*` 속성은 분류되지 않은 metadata로 보고한다. 새 언어 전용 속성은 금지 또는
보존 중 하나로 분류된 뒤에만 출구를 넘을 수 있다. 입력과 결과 타입이 같은
unrealized cast, scalar(`core.i32`·`core.i1`·`core.f64`, `tribute_rt.int`·`nat`·
`bool`·`float`)와 `tribute_rt.anyref` 사이의 unrealized cast, 대상 함수의
signature와 다른 타입의 `func.constant`, 위의 금지 목록에 든 `adt` 값과 operation도
위반으로 보고한다. 매개변수 ownership
계약도 검사한다. `call_conv = "tail"`인 signature의
입력에 `tribute.ownership = "consumed"`가 없거나, `tribute.ownership` 값이
`"consumed"`가 아니면 위반이다. Root `main`이 있으면 그것이 매개변수가 없고 결과가
`Nil`이며 platform 규약을 따르는 정의인지도 검사한다. 위반이 하나라도 있으면
target dialect lowering에 들어가기 전에 컴파일이 실패한다. 이 검증은 build 구성과 관계없이 항상 수행한다.

### Unrealized cast

Boxing과 unboxing은 변환을 만든 pass가 이미 만들었으므로, 출구에 남는
`core.unrealized_conversion_cast`는 타입만 바꾸며 target type 변환을 기다린다.
Scalar와 `tribute_rt.anyref` 사이의 cast가 출구에 남았다면 그 cast를 만든 pass가
boxing·unboxing을 직접 만들지 않은 것이다. 그대로 두면 target type 변환이 native
ownership/RTTI 계획 뒤에 box를 만들어 그 계획이 보지 못한 managed 값이 생기므로,
출구 검증이 이를 거부한다.
이런 cast의 적법성은 target 타입에 따라 정해지므로 target type 변환이 결과
타입을 변환하고 필요한 representation 변경을 materialize한다. 그 끝에서 converter
없는 `ReconcileUnrealizedCasts` pass가 cast를 닫으며, 그 뒤 남은 cast는 target emission
경계가 거부한다.

### 직렬화

경계 출력은 textual IR로 출력한 뒤 새 `IrContext`에서 파싱해도 frontend나 control
보조 테이블 없이 target pipeline을 계속 진행할 수 있어야 한다. 출구 계약에 필요한
정보는 모두 IR의 operation, type, attribute로 표현한다.
