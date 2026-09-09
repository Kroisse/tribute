# 직접형 제어와 CPS 효과 처리 파이프라인

이 문서는 직접형 IR을 CPS로 바꾸는 conversion source of truth다. Operation
구조와 verifier 계약은 [ir.md](ir.md#direct-style-control), 계층별 소유권과
pipeline 조합은 [implementation.md](implementation.md#직접형-제어-소유권)를
따른다.

핵심 전략은 **tail-call CPS + evidence-based handler dispatch**이다.
모든 CPS 이전은 proper tail call이며 제어 값을 반환하지 않는다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="direct-style-control-boundary"></a>

## 직접형 호출 대상과 제어 경계

모든 ability operation invocation은 typechecking이 확정한
`operation_kind = @fn | @op`를 가진 `tribute_control.perform`이다. 일반 callable도
`tribute_control.func`, `lambda`, `func_ref`, `call`, `call_indirect`, `return`을
사용한다. Operation kind와 callable convention은 typechecking 결과이며
frontend나 conversion이 body 형상에서 추론하거나 재분류하지 않는다.

```text
Direct < EvidenceDirect < Cps
```

Effect row, convention 순서, 실행 region 내부의 ANF invariant는 바뀌지 않는다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="pre-cps-callable-shape"></a>

### CPS 변환의 호출 대상 형상

입력 형상은 [ir.md](ir.md#direct-style-control)의 operation/type 계약을 따른다.
모든 callable은 exact `tribute.calling_convention` code 0/1/2를 가진 logical
type과 source parameter/result만 사용하며 evidence, environment, `ContinuationFrame<R>`는
아직 없다.

`tribute_control_to_cps`는 closure extraction 전에 전체 callable graph를 하나의
단위로 변환한다:

| Convention | Physical parameter | Result |
| ---- | ---- | ---- |
| `Direct` | 소스 parameter | 소스 result |
| `EvidenceDirect` | evidence 뒤 소스 parameter | 소스 result |
| `Cps` | evidence, `ContinuationFrame<R>`, source parameter 순서 | logical `core.never`, physical empty result |

Parameter 순서는 기존 `CallableAbi` 순서다. `ContinuationFrame<R>`는 CPS callable이
전달하는 continuation/control frame이며 `Done<R>`와 내부의 어휘적 `Dispatch<R>`를
함께 보존한다. `Completion<X, R>`와 `ResumeExact<I, R>`는 각각
`(Evidence, ContinuationFrame<R>, X)`와 `(Evidence, ContinuationFrame<R>, I)`를 받아
logical `core.never`로 끝난다. Control lowering
뒤 worker와 continuation은 result vector가 비어 있으며 제어 값을 반환하지 않는다.

Conversion은 먼저 모든 logical callable type, definition, lambda, `func_ref`,
direct/indirect call, return의 대응 관계를 검증하고 physical symbol과 type을
계산한 뒤 함께 rewrite한다:

- `tribute_control.func`는 `func.func`와 physical `func.func_sig`를 만든다.
- `tribute_control.lambda`는 physical `closure.lambda`와
  `closure.closure<func.func_sig<...>>`를 만든다. 이 단계의 signature에는
  `CallableAbi` hidden parameter가 있지만 environment는 없으며, 기존 closure
  lowering이 environment를 interpose한다.
- `tribute_control.func_ref`는 빈 environment의 `closure.new`와 env-bearing
  adapter `func.func`를 만든다. Adapter는 result callable의 같거나 더 강한
  convention을 사용하고 필요한 hidden operand와 source argument를 대상 physical
  worker에 전달하므로 named function도 다른 closure와 같은 `call_indirect` ABI를
  갖는다. 기존 closure lowering은 이 `closure.new`를 `func.constant`와 physical
  closure struct로 바꾼다.
- `Direct`/`EvidenceDirect`의 `tribute_control.call`과 `call_indirect`는 각각
  physical `func.call`과 `func.call_indirect`가 된다. `Cps` call은 suffix를
  담은 `ContinuationFrame<R>`를 전달하고 named target에는 `func.tail_call`, dynamic target에는
  `func.tail_call_indirect`를 쓴다. Evidence, ContinuationFrame과 environment는 `CallableAbi`
  순서로 삽입한다.
- `tribute_control.return`은 `Direct`/`EvidenceDirect`에서 `func.return`이 된다.
  `Cps`에서는 ContinuationFrame의 `Done<R>`으로 `value`를 이전하며 뒤에
  `func.return`이나 result가 없다.

알려진 CPS target으로의 최종 이전은 `func.tail_call`, closure, continuation,
ContinuationFrame의 `Done<R>`처럼 동적인 target으로의 이전은 `func.tail_call_indirect`를 사용한다.
Shared IR의 caller/callee result는 `core.never`이고 target signature의 result
vector는 비어 있어야 한다.
생성한 continuation, `done_k`, handler-dispatch function에도
`tribute.calling_convention = 2`를 붙여 backend-ready verifier가 semantic role을
식별하게 한다.

기존 `CallableAbi::interpose_environment`에 따라 extracted lambda와 `func_ref`
adapter의 최종 parameter 순서는 `Direct`에서 `environment, source...`,
`EvidenceDirect`에서 `evidence, environment, source...`, `Cps`에서
`evidence, environment, ContinuationFrame<R>, source...`다. `call_indirect`도 같은 순서로
environment를 삽입한다.

Convention-proven physical `closure.closure` type은 outer type에 exact
`tribute.closure_environment_index`도 기록한다. 일반 callable과 생성된
continuation은 `CallableAbi`가 정한 Evidence와 ContinuationFrame entry parameter를 사용한다.
따라서 producer가
실제 physical entry slot을 type provenance에 명시하고 lambda lifting과 indirect
call lowering은 그 slot을 그대로 cross-check/consume한다. Consumer가 arity,
parameter type, body shape, 또는 calling-convention marker만으로 slot이나 hidden
operand를 복원하는 것은 illegal이다.

Environment-bearing `func.func`는 zero-based physical slot을
`tribute.closure_environment_index`로 기록한다. Bodyless declaration은 이
function-level provenance가 필수이며, definition은 entry block의 `__env` marker와
같은 slot이어야 한다. Slot은 outer convention-proven closure type이 기록한 exact
physical order와 exact `tribute_rt.anyref` type에 일치해야 하며 type이나 arity로
추측하지 않는다.

생성한 physical definition, lambda, adapter, direct/indirect call에는 logical
type의 convention을 기존 `tribute.calling_convention` attribute로 복사한다.
Metadata/type/symbol 불일치는 conversion failure이며 hidden operand를 추측하지
않는다. TrunkIR의 `TypeConverter`, function signature conversion, dialect
builder를 재사용하되 `tribute_control` 전용 graph pattern은
`tribute_control_to_cps`가 소유한다.

같은 legalization에서 각 `tribute_control.perform`을 typecheck된
`operation_kind`에 따라 변환한다. `@fn`은 suffix를 capture하지 않고
`ability.call`을, `@op`은 `ability.perform`과 필요한 `ability.handle_dispatch`
표면을 만든다. 이 pass는 `effect.dispatch_*`나 `effect.extend`를 만들지 않는다.
이 결정은 `CallableAbi`에 encode하지 않으며 pass가 operation kind를 추론하거나
변경해서는 안 된다.

Root `main` delimiter와 external ABI 조합 책임은
[implementation.md](implementation.md#직접형-제어-소유권)을 따른다.

### 논리적 CPS 적법화

Shared conversion은 실행 가능한 region 하나를
`convert_region(region, exit_k)`로 lower한다. `exit_k`는 region result를 위한
logical continuation이며 선택된 backend carrier가 아니다. Conversion은
operation을 왼쪽에서 오른쪽으로 소비한다:

- 일반 value operation은 기존 dialect에 남고 그 result는 남은 suffix로 흐른다.
- `tribute_control.call` 또는 `call_indirect`의 convention이 `Cps`이면 현재
  suffix continuation을 공급한다. `Direct`와 `EvidenceDirect` call result는
  일반 suffix로 흐른다.
- `tribute_control.perform`은 검증된 `operation_kind`만으로 분기한다. `@fn`이면
  suffix를 capture하지 않고 `ability.call`을 만들며 반환된 operation result가
  일반 suffix로 흐른다. `@op`이면 현재 suffix와 일치하는 dynamic handle
  boundary까지 선택된 모든 enclosing structured exit를 capture한 뒤
  `ability.perform`을 만든다. 후속 `lower_ability_perform`이 각각
  `effect.dispatch_tail`과 `effect.dispatch_cps`로 낮춘다. Operand packing은
  이 lower 경계에서만 수행한다.
  `op -> Never`에는 suffix를 capture하지 않는다.
  기존 필수 ABI operand에는 zero-capture reject continuation을 공급한다.
  Reject closure의 callable type은 canonical `Resume<R>`이며 다음과 같다.

  ```text
  (Evidence, exact ContinuationFrame<R>, anyref) -> Never
  ```

  세 번째 formal은 읽지 않는다.
  Source operation result는 `Never`로 남는다.
- `tribute_control.yield value`는 region의 `exit_k(value)`를 호출한다.

일반 structured control은 기존 `scf.*` dialect에 남고
`tribute_control_to_cps`가 region과 suffix를 재귀적으로 변환한다.
`tribute_control.if`는 추가하지 않는다. `scf.if`, case lowering, guarded arm,
short-circuit selection의 각 실행 branch는 독립적으로 변환한다. 각 branch의 exit
continuation은 먼저 structured merge에 도달한 뒤 enclosing suffix를 실행한다.
Condition, scrutinee, 앞선 strict value는 source 순서로 한 번만 평가하며 선택된
branch, guard, 오른쪽 항만 실행한다.

`tribute_control.handle`은 delimiter를 만든다:

1. 정상 body yield는 completion region에 정확히 한 번 들어가며 completion
   yield는 enclosing `exit_k`를 계속 실행한다.
2. Dynamic하게 설치된 handler 중 가장 가까운 일치 handler가 operation
   argument를 받는다. `fn` arm은 operation result를 yield하고 자동으로 resume한다.
3. Resumptive general `op` arm은 affine resumption을 받는다.
   `tribute_control.resume`은 수행된 operation result를 공급하고 capture한 body
   continuation을 실행한 뒤, 반환된 handle answer로 arm-local suffix를 실행한다.
   Source `op -> Never` arm은 resumption을 받지 않고 `resume`을 포함할 수 없으며
   yield type은 enclosing handle result type과 같아야 한다. Converter는 rewrite
   전에 verifier와 같은 조건을 확인한다.
4. Resume하지 않고 yield하는 general arm은 해당 handle을 직접 완료한다. 포기된
   performed-computation suffix와 completion region을 건너뛴다.
5. Nested handle은 같은 delimiter를 재귀적으로 만든다. Resumed path는 perform과
   handler 사이에서 capture된 모든 case/conditional/short-circuit/nested-handle
   frame에 다시 진입한다. Resume은 동적 `ContinuationFrame<R>`에서 새 불변 어휘적
   dispatcher를 만들고, 그 ContinuationFrame을 다음 suffix와 resume에 전달한다. Non-resumed
   path는 해당 frame을 포기한다.

Converter는 region, block suffix, `tribute_control.yield`, 검증된
`operation_kind`, callable convention metadata에서 모든 continuation을
도출해야 한다. Typed AST를 다시 scan하거나 AST containment를 검사하거나
operation kind를 추론하거나 case, guard, short-circuit, nested handle 전용 규칙을
추가해서는 안 된다.

Affine `resume_token`의 type/placement는 operation-local verifier가, use-def와
closure-capture를 지나는 single static ownership path는 whole-IR verifier가
검사한다. Static SSA 검사는 capture된 closure의 반복 호출을 막지 못하므로
conversion은 runtime consumed state를 만들고 두 번째 호출을 continuation
재진입 전에 거부하거나 trap한다. Token은 post-CPS IR에 남지 않는다.

Canonical `core.never`인 source `op -> Never`는 token과 source suffix
continuation을 만들지 않는다. 기존 ABI에는 capture가 없고 body가
`func.unreachable`인 typed reject continuation을 전달한다.
이 closure는 dispatch가 요구하는 canonical `Resume<R>` type을 정확히 사용한다.
`anyref` formal은 읽지 않는다. Null, in-band sentinel, 임의의 `anyref`는
대체할 수 없으며 호출되면 trap한다.

### 적법화 경계

`tribute-control-pre-cps` named boundary는 검증된 `tribute_control.*`과 일반
value/structured dialect만 허용한다. Frontend 적합성 검사는 `func.*`,
`closure.*`, `func.func_sig`, `closure.closure`, 기존 `ability.*`, `effect.*`,
legacy CPS-dispatch operation을 모두 거부한다. Partial rewrite
도중에는 logical/physical operation이 일시적으로 공존할 수 있지만 이 상태는
named pre-CPS boundary가 아니다.

성공한 shared conversion은 defining rule이
`illegal_dialect("tribute_control")`인 partial
`tribute-control-post-cps` target과 Tribute type walk를 검증한다. 남은
`tribute_control` operation 또는 `callable`/`resume_token` type은 source
location에서 conversion failure가 된다. 이 경계에는 일관된 physical
`func.*`/`closure.*`/`func.func_sig` graph와 logical `ability.*` dispatch 표면만
남는다. `lower_closure_lambda`와 `prepare_closure_lowering`은 이 shared graph를
준비하지만 `closure.new`, `closure.func`, `closure.env`와 convention-proven closure
type은 target ABI validation까지 유지한다. `lower_ability_perform`,
`resolve_evidence`, `lower_handle_dispatch`가 `ability.*`를 `effect.*`까지 낮춘 뒤,
target pipeline의 closure storage finalization과 Native/Wasm evidence pass가
backend ABI로 제거한다.
Backend-ready Tribute boundary는 남은 `tribute_control.*`, `ability.*`,
`effect.*`를 각각 독립적으로 거부한다.

논리적 CPS 함수는 source result를 직접 반환하지 않는다. 완료 값은 ContinuationFrame의
`Done<R>`에만 전달되고 logical control result는 `core.never`다. Physical control
lowering은 이를 empty-result 함수와 proper tail transfer로 바꾸며 반환되는
control value를 만들지 않는다. Resume하지 않는 handler arm도
`Escape`를 반환하는 대신 해당 handle의 exit continuation으로 직접 tail
transfer하므로 completion region과 포기한 suffix를 구조적으로 건너뛴다.

물리화는 전체 callable과 dispatch/frame/R 계약을 검증하고 변환을 계획한 뒤 적용한다.
실패하면 기존 operation, 타입 참조, attribute와 alias를 변경하지 않는다. Alias,
closure signature, function constant, exact indirect signature, SSA 결과와 block
argument, 중첩 타입 attribute를 함께 변환한다. 일반 `never`·`nil` 치환은 하지 않는다.
Closure tail lowering은 caller·callee·exact indirect signature의 전체 결과 목록을
비교하여 논리 `[never]`와 물리 `[]`를 각각 지원한다.

Final native/Wasm backend-ready 경계는 `Cps` worker, continuation, `done_k`,
handler-dispatch의 result vector가 비어 있고 모든 CPS transfer가
`func.tail_call` 또는 `func.tail_call_indirect`로 끝나는지 검사한다.
`Step`, trampoline, CPS control-result 역할의 `anyref`와
`__tribute_cps_control` private enum은 거부한다. Boxed source value, erased effect
payload, closure environment와 dispatch closure field에 쓰는 일반 `anyref`는 이
검사의 대상이 아니다.

## 핵심 설계

### `fn` operation: direct dispatch

직접형 입력은 위 규칙의 `operation_kind = @fn` perform이며,
`tribute_control_to_cps`는 continuation을 만들지 않고 기존 `ability.call`
경로로 내린다:

```text
%result = ability.call %arg
  { ability_ref = @Logger, op_name = @log }
```

공통 lowering은 payload packing 전에 exact resume의 frame 입력과 frame metadata에서
답 타입 `R`을 구하고 evidence·dispatch·resume의 연결을 검증한다.

```text
%payload = cast %arg to anyref
effect.dispatch_cps %ev, %dispatch, %resume, %payload
  { ability_ref = @State, op_name = @get, answer_type = R }
```

필수 `answer_type: Type`은 `ContinuationFrame<R>`의 의미적 `R`이다. 호출 결과,
물리화 완료 marker 또는 callable provenance가 아니며 `ability.perform`에 중복하지
않는다. Dispatch가 요구하는 resume과 실제 resume의 exact 타입이 같아야 한다.
같은 `R`을 가진 서로 다른 nominal frame도 호환되지 않는다. Dispatch와 resume의
convention은 `Cps`, environment 위치는 각각 1과 0이다. 이미 낮춘 closure는
기존에 인증된 callable provenance만 사용하며 이름이나 저장 형태에서 추론하지 않는다.
누락되거나 잘못된 attribute 및 frame·R 불일치는 명시적 lowering 오류다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="dispatch-layers"></a>

### Dispatch 계층

여기에는 서로 다른 세 dispatch 계층이 있다.

1. `ContinuationFrame<R>`의 내부 `Dispatch<R>`는 resume에서 어휘적 dispatcher를 재구성하는
   CPS closure이며 `(evidence, resume, prompt, ability_id, op_id, payload)`를 받는다.
2. `effect.dispatch_cps(evidence, dispatch, resume, payload)`는 필수
   `answer_type = R`을 보존하는 결과 없는 대상 독립적 operation이다.
3. 대상 dispatch ABI는 compiler-owned
   `(Evidence, Environment, Resume, Prompt, AbilityId, OperationIndex, Payload)`
   입력과 빈 결과 목록을 갖는다.
   Native는 canonical shared signature에 기존 Native 변환을 적용하여
   `(ptr, ptr, ptr, i32, i32, i32, ptr) -> ()`를 얻는다. Wasm은 대상의 canonical
   evidence/environment/closure 저장 타입, 세 i32와 payload 타입을 사용한다.
   Closure 입력은 canonical `_closure` ADT이며 일반 `wasm.structref`로 대체하지
   않는다. 기존 대상 타입 변환은 정확한 공통 closure 저장 타입을 이 ADT로 변환한다.

의미 계약은 closure 타입 소거와 physicalization 변경 전에 검증한다. 최종 target
lowering은 실제 operand와 독립적으로 고정 signature를 구성하고 operand를 대조한다.
서로 다른 유효한 `R`도 동일한 물리 ABI를 가지며 resume의 frame은 별도 dispatch
입력이 아니다. `answer_type`은 일반 재귀 타입 변환에 참여하고 effect operation 제거
시 소비한다. Raw target call에 복제하지 않으며 legacy dispatch 계약은 유지한다.

정의, lambda, adapter, direct/indirect call, return, suffix, resume, handle은
ContinuationFrame을 같은 callable provenance로 전달한다. 내부 Dispatch를 effect operation이나
target handler ABI로 대체하지 않는다.

Effect point 이후의 코드는 이미 `%continuation` closure 안에 있으므로,
`ability.perform` 이후의 같은 function-body ops는 dead code가 된다.

이 lowering은 source kind를 재분류하지 않는다. 일반 `op` handler가 실제로
항상 tail-resumptive인지 분석하여 tail path로 최적화하는 작업은 표준 `@op`
semantic lowering 이후의 별도 IR optimization이다.

### Root `main` delimiter

Root `main`은 하나뿐인 target-independent CPS delimiter다. Source residual-effect
계약은 기존 pure-or-`Io` entry를 유지하며 residual general effect는 backend 전에
거부한다. Nested module의 `main`은 일반 worker다.

Target-independent 경계는 Direct/EvidenceDirect export wrapper가 source result
type의 completion cell과 이를 capture한 terminal `Done<R>` 및 terminal
`Dispatch<R>`를 담은 정확한 `ContinuationFrame<R>`를 소유한다는 추상 조합
계약만 정한다. Worker ABI의 두 번째 operand는 이 nominal frame이며 bare
`done_k`로 대체하거나 closure storage/arity에서 복원하지 않는다. Shared IR에서
CPS entry와 `done_k`의 result는 `core.never`이며,
`func.tail_call`과 `func.tail_call_indirect` verifier도 caller/callee의
`core.never` 일치를 검사한다.

최종 계약에서는 atomic physical CPS switch 이후 target signature lowering이
이 CPS signature를 native/Wasm empty-result signature로 바꾼 뒤 실제 wrapper와
결과 없는 ordinary call을 합성한다. Root `done_k`는 source result를
cell에 정확히 한 번 쓰고 terminal dispatch는 root 밖 general operation transfer를
끝내며, wrapper는 이 둘을 immutable `ContinuationFrame<R>`로 materialize해 worker에
전달한 뒤 proper-tail-call chain이 끝나면 cell을 읽어 source result로 반환한다.
공통 `func.func_sig`와 `func.call`은 0개 또는 1개 결과를 지원한다. 논리 CPS
producer는 `[core.never]`를 유지하고 물리화는 정확한 Cps 결과만 `[]`로 바꾼다.
이 adapter는 answer-type polymorphism, trampoline, in-band sentinel 또는
control carrier가 아니다.

### `handle`: evidence extension + handler closures

`handle` lowering은 두 종류의 dispatch closure를 만든다.

- `handler_dispatch`: `(k, op_idx, value) -> ()`
  - general `op` handlers용
  - closure environment가 handle exit를 소유하며 `resume`과 non-resuming exit는
    각 continuation으로 indirect tail transfer한다.
- `tr_dispatch_fn`: `(op_idx, value) -> anyref`
  - `fn` handlers용
  - `anyref`는 erased source result이며 CPS control carrier가 아니다.

`resolve_evidence`는 handler boundary에서 새 marker를 만들어 evidence를
확장한다.

Shared evidence resolution represents handler installation with the same effect
ABI instead of constructing the concrete Marker layout directly:

```text
%ev2 = effect.extend %ev, %prompt_tag, %tr_dispatch_fn, %handler_dispatch
  { ability_ref = @State }
```

Backends lower `effect.extend` to their own evidence representation. The native
backend maps it to the `__tribute_evidence_extend` ABI.

```text
struct Marker {
    ability_id: i32,
    prompt_tag: i32,
    tr_dispatch_fn: ptr,
    handler_dispatch: ptr,
}
```

Evidence는 ability id 기준으로 정렬된 marker 배열이며, handler 설치 시
새 evidence 값을 만든다.

Marker layout과 evidence runtime ABI는 `tribute-ir`의
`ability::MarkerField`와 `ability::evidence_abi`가 컴파일러 내부의 단일
정의다. 필드 순서는 다음과 같고 모든 shared pass와 backend lowering은 이
순서를 직접 숫자로 복제하지 않는다.

| Field | Index | Type | Meaning |
| --- | ---: | --- | --- |
| `ability_id` | 0 | `i32` | stable ability key for sorted evidence lookup |
| `prompt_tag` | 1 | `i32` | prompt installed for the active handler |
| `tr_dispatch_fn` | 2 | `ptr` | tail-resumptive dispatch closure or null |
| `handler_dispatch` | 3 | `ptr` | full CPS dispatch closure or null |

WasmGC uses the same field order and shared field identifiers, but its concrete
GC marker type stores the dispatch closures as `anyref` closure references
instead of native `ptr` values. Wasm effect ABI lowering therefore expands
`effect.dispatch_tail` and `effect.dispatch_cps` into evidence lookup,
`wasm.struct_get` of the selected marker closure, closure table-index/env
decomposition, and `wasm.call_indirect`.

Empty evidence is represented in high-level IR as an empty `core.array(Marker)`
or null evidence placeholder, and backend lowering turns that into the target
runtime representation. Native lowering maps it to `__tribute_evidence_empty()`.
When a handler for the same `ability_id` is nested inside an outer handler,
evidence extension replaces the existing marker so lookup resolves to the
nearest handler.

Native runtime ABI:

```text
__tribute_evidence_empty() -> ptr
__tribute_evidence_lookup(ev: ptr, ability_id: i32) -> i32
__tribute_evidence_extend(
    ev: ptr,
    ability_id: i32,
    prompt_tag: i32,
    tr_dispatch_fn: ptr,
    handler_dispatch: ptr,
) -> ptr
__tribute_evidence_lookup_tr(ev: ptr, ability_id: i32) -> ptr
__tribute_evidence_lookup_handler(ev: ptr, ability_id: i32) -> ptr
```

### `ability.handle_dispatch`

`ability.handle_dispatch`는 runtime dispatch loop가 아니다. Effect 발생 시점에서
이미 handler closure로 tail-call되므로,
`lower_handle_dispatch`는 body result에 `done` handler를 적용하는 역할만 한다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="shared-middle-end-pipeline"></a>

## 공통 middle-end 파이프라인

Callable/control과 effect 관련 pass의 순서는 다음과 같다:

이 순서는 exact root contract가 있는 source-logical CPS route의 contract다.
Compatibility route는 그 metadata 없이 기존 closure-lowering 순서를 유지하며,
이를 이 target ABI boundary에 진입시키지 않는다.

```text
ast_to_ir (tribute_control callable/control + ordinary value IR)
→ tribute_control_to_cps
→ lower_closure_lambda
→ prepare_closure_lowering
→ lower_ability_perform
→ resolve_evidence
→ lower_handle_dispatch
→ effect ABI verification
→ target ABI validation and CPS signature physicalization
→ lower_closures_in_func
→ native evidence lowering → finalize_closure_storage_layout → native tail-call lowering
  or finalize_closure_storage_layout → integrated Wasm evidence/tail-call lowering
→ backend-ready verification
```

`tribute_control_to_cps`의 출력은 physical `func.*`/`closure.*` callable 표면과
logical `ability.*` dispatch 표면이다. Shared ability/evidence pass와 strict target
ABI validation은 convention-proven `closure.closure` callable type을 그대로
소비한다. 그 뒤 target pipeline이 closure operation과 모든 type-bearing storage
surface를 canonical `_closure` layout으로 함께 바꾸고 backend evidence/runtime와
proper tail transfer lowering이 이를 소비한다. `tribute.closure_callable_type`은
exact closure lowering에서만 잠시 쓰고 storage finalization에서 제거한다. 이는
semantic type equivalence를 만들지 않는다.

## Effect ABI Boundary

The `effect` dialect is the target-independent boundary between language
semantics and concrete runtime layout.

Initial operations:

- `effect.extend(evidence, prompt_tag, tr_dispatch_fn, handler_dispatch)
  { ability_ref } -> evidence`
- `effect.dispatch_tail(evidence, payload) { ability_ref, op_name } -> result`
- `effect.dispatch_cps(evidence, dispatch, resume, payload)
  { ability_ref, op_name, answer_type } -> ()`

Rules:

- `ability.perform` and `ability.call` are illegal after the shared
  ability-dispatch lowering boundary.
- `effect.*` operations may remain after shared lowering and before
  backend-specific effect ABI lowering.
- `effect.dispatch_cps`는 control result를 만들지 않으며 backend lowering은
  handler-dispatch closure로 proper indirect tail transfer한다.
- Backend-ready conversion targets must reject residual `effect.*` operations.
- Shared passes must not inspect Marker field numbers, handler-table storage
  layout, closure field positions, or backend function-pointer representation.
- Payload value는 shared lowering에서 이미 single value로 pack한다. 직접형 frontend는
  source-logical `tribute_control.perform` operand를 유지하며 dispatch 전략에
  맞춰 pack하지 않는다. 누락된 payload는 `effect.*`에 도달하기 전에
  target-independent null/empty value로 명시적으로 표현한다.

## Backend Implications

### Native

Evidence runtime은 `tribute-runtime`의
`__tribute_evidence_*` C ABI 함수로 제공되고, native effect ABI lowering은
`effect.*`를 marker lookup helper, runtime evidence extension, closure
decomposition, and indirect calls로 변환한다.

### WasmGC

WasmGC도 같은 shared middle-end를 사용한다. `wasm/evidence_to_wasm`은
`effect.extend`를 marker construction + `__tribute_evidence_extend` helper
call로 낮추고, `effect.dispatch_tail` / `effect.dispatch_cps`는
`__tribute_evidence_lookup`, marker closure field access, closure
table-index/env unpacking으로 낮춘다. CPS control transfer는
`func.tail_call_indirect`를 거쳐 `wasm.return_call_indirect`가 된다. Source data를
반환하는 일반 indirect call은 계속 `wasm.call_indirect`를 사용할 수 있다.
