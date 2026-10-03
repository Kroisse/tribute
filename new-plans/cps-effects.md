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
`operation_kind = "fn" | "op"`를 가진 `tribute_control.perform`이다. 일반 callable도
`tribute_control.func`, `lambda`, `func_ref`, `call`, `call_indirect`, `return`을
사용한다. Operation kind와 callable convention은 typechecking 결과이며
frontend나 conversion이 body 형상에서 추론하거나 재분류하지 않는다.

```text
Direct < EvidenceDirect < Cps
```

Effect row, convention 순서, 실행 region 내부의 ANF invariant는 바뀌지 않는다.

Local lambda의 convention도 typechecking이 확정한 effect row에서 계산한다. Lambda
본문은 닫힌 빈 accumulator로 검사하고, 본문이 남긴 open row 또는 lambda를
검사한 callable context의 open tail만 callable type에 남긴다. pure body와
effect-free consumer의 lambda는 `Direct`가 될 수 있다. 반대로 반환되거나
escaping 저장 위치에 들어가 open callable contract를 받거나, open-effect
consumer가 요구하는 callback이면 open row를 유지하여 `Cps`가 된다. 본문이
요구한 ability도 row에 남으며 그 ability의 convention lower bound를 적용한다.
이 선택은 이미 생성된 Cps 값을 Direct로 바꾸는 후처리가 아니며,
`calling_convention_for_effect_row`의 open-row ⇒ `Cps` 규칙을 완화하지 않는다.

지역 source lambda의 검사된 인스턴스와 실제 소비 worker의 callable parameter
계약을 구별한다. 고정된 데이터 타입의 lambda는 원래 binding 위치에서 각 필요한
인스턴스와 convention으로 생성한다. 이는 이미 생성된 Cps 값을 Direct로 cast하는
것이 아니다. Named callable은 정확한 target identity를 가진 `func_ref`의 기존
adapter를 사용하며, semantic use가 pure여도 소비 worker가 요구하는 더 강한
convention을 보존한다.

Lambda capture 목록은 생성된 body가 실제 사용하는 외부 SSA 값과 일치해야 한다.
Named reference를 새 `func_ref`로 대체하여 사라진 capture는 제거하고, 남는 값은
lexical scope와 dominance를 유지하여 중복 없이 전달한다. 이 변환은 RHS 평가를
복제하거나 resume token의 affine capture 경로를 늘려서는 안 된다.

Source 타입 검사는 정확한 equality, `Never` 표현식의 방향성 있는 제거, 공통 source
결과 추론을 구별한다. 분기의 answer를 맞추기 위해 정상 source 값을 `Never`로
cast해서는 안 된다. 실제 source `Never` operation의 결과 타입과 재개할 수 없다는
의미는 그대로 보존한다. 이 표현식 규칙은 logical CPS의 `core.never` 결과나
target이 담당하는 물리적인 빈 결과 표현을 변경하지 않는다.

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

Parameter 순서는 `CallableAbi`가 정의한다. `ContinuationFrame<R>`는 CPS callable이
전달하는 continuation/control frame이며 `Done<R>`와 내부의 어휘적 `Dispatch<R>`를
함께 보존한다. `Completion<X, R>`와 `ResumeExact<I, R>`는 각각
`(Evidence, ContinuationFrame<R>, X)`와 `(Evidence, ContinuationFrame<R>, I)`를 받아
logical `core.never`로 끝난다. Target signature physicalization 뒤 worker와
continuation은 result vector가 비어 있으며 제어 값을 반환하지 않는다.

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
  순서로 삽입한다. `call_indirect`의 callee는 logical callable type에 exact contract를
  보존하므로 변환된 `func.call_indirect`는 그 converted closure contract에서 유도한 exact
  `signature`를 싣는다. environment는 여기서 아직 interpose하지 않는다.
- `tribute_control.return`은 `Direct`/`EvidenceDirect`에서 `func.return`이 된다.
  `Cps`에서는 ContinuationFrame의 `Done<R>`으로 `value`를 이전하며 뒤에
  `func.return`이나 result가 없다.

알려진 CPS target으로의 최종 이전은 `func.tail_call`, closure, continuation,
ContinuationFrame의 `Done<R>`처럼 동적인 target으로의 이전은 `func.tail_call_indirect`를 사용한다.
Shared IR의 caller/callee result는 `core.never`이고 target signature의 result
vector는 비어 있어야 한다.
생성한 continuation, `done_k`, handler-dispatch function에도
`tribute.calling_convention = 2`를 붙여 target ABI 검증이 semantic role을
식별하게 한다. 이 속성은 [representation/ABI 경계](ir.md#representationabi-경계)
안에서 소비되며, 경계 이후에는 signature의 `call_conv`와 proper-tail operation만
남는다.

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
추측하지 않는다. 이 function-level provenance의 마지막 독자는 물리화 검증이다.

생성한 physical definition, lambda, adapter, direct/indirect call에는 logical
type의 convention을 기존 `tribute.calling_convention` attribute로 복사한다.
Metadata/type/symbol 불일치는 conversion failure이며 hidden operand를 추측하지
않는다. TrunkIR의 `TypeConverter`, function signature conversion, dialect
builder를 재사용하되 `tribute_control` 전용 graph pattern은
`tribute_control_to_cps`가 소유한다.

같은 legalization에서 각 `tribute_control.perform`을 typecheck된
`operation_kind`에 따라 변환한다. `"fn"`은 suffix를 capture하지 않고
`ability.call`을, `"op"`은 `ability.perform`과 필요한 `ability.handle_dispatch`
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
- `tribute_control.perform`은 검증된 `operation_kind`만으로 분기한다. `"fn"`이면
  suffix를 capture하지 않고 `ability.call`을 만들며 반환된 operation result가
  일반 suffix로 흐른다. `"op"`이면 현재 suffix와 일치하는 dynamic handle
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
`closure.*`, `func.func_sig`, `closure.closure`, `ability.*`, `effect.*`를 모두 거부한다.
Partial rewrite 도중에는 logical/physical operation이 일시적으로 공존할 수 있지만 이 상태는
named pre-CPS boundary가 아니다.

성공한 shared conversion은 defining rule이
`illegal_dialect("tribute_control")`인 partial
`tribute-control-post-cps` target과 Tribute type walk를 검증한다. 남은
`tribute_control` operation 또는 `func_sig`/`resume_token` type은 source
location에서 conversion failure가 된다. 이 경계에는 일관된 physical
`func.*`/`closure.*`/`func.func_sig` graph와 logical `ability.*` dispatch 표면만
남는다. `lower_closure_lambda`는 이 shared graph의 lambda를 추출하지만
`closure.new`, `closure.func`, `closure.env`와 convention-proven closure type은
target ABI validation까지 유지한다. `lower_ability_perform`,
`resolve_evidence`, `lower_handle_dispatch`가 `ability.*`를 `effect.*`까지 낮춘 뒤,
target pipeline의 closure storage finalization과 Native/Wasm evidence pass가
backend ABI로 제거한다.
Shared evidence resolution은 lookup/extend용 가짜 함수 정의를 만들지 않는다.
Native extern 선언과 Wasm helper 구현은 각 target evidence pass가 생성한다.
Backend-ready Tribute boundary는 남은 `tribute_control.*`, `ability.*`,
`effect.*`를 각각 독립적으로 거부한다.

논리적 CPS 함수는 source result를 직접 반환하지 않는다. 완료 값은 ContinuationFrame의
`Done<R>`에만 전달되고 logical control result는 `core.never`다. Physical control
lowering은 이를 empty-result 함수와 proper tail transfer로 바꾸며 반환되는
control value를 만들지 않는다. Resume하지 않는 handler arm은 해당 handle의
exit continuation으로 직접 tail transfer하므로 completion region과 포기한 suffix를 구조적으로 건너뛴다.

물리화는 전체 callable과 dispatch/frame/R 계약을 검증하고 변환을 계획한 뒤 적용한다.
실패하면 기존 operation, 타입 참조, attribute와 alias를 변경하지 않는다. Alias,
closure signature, function constant, exact indirect signature, SSA 결과와 block
argument, 중첩 타입 attribute를 함께 변환한다. 일반 `never`·`nil` 치환은 하지 않는다.
Closure tail lowering은 caller·callee·exact indirect signature의 전체 결과 목록을
비교하여 논리 `[never]`와 물리 `[]`를 각각 지원한다.
물리화는 검증에 쓴 provenance 중 이후 단계가 읽지 않는 것을 적용 단계에서 소비한다.
Continuation frame 타입의 `tribute.cps_continuation_frame_result`를 지우고,
`func.func`의 `tribute.closure_environment_index`도 지운다. 물리 frame에서 `R`은
`Done<R>`의 입력 타입이 나타내고, environment는 physical signature의 순서 있는
입력이다. Root bridge 합성은 이렇게 물리화된 frame을 검증하며 frame answer
provenance가 남아 있으면 거부한다. 물리화 이후에 합성되는 함수는 이 속성들을 기록하지 않는다.

`tribute.calling_convention`도 operation 종류마다 마지막으로 읽는 단계가 소비하며,
경계 출구에서 일괄로 지우지 않는다. 물리화는 직접 `func.call`/`func.tail_call`과
callee가 convention-proven closure가 아닌 indirect transfer의 convention을 callee
계약과 대조한 뒤 지운다. Root bridge 합성은 source `main`의 convention을 읽어
wrapper 입력을 만들고, 자신이 만드는 호출에는 이 속성을 기록하지 않는다. Closure
lowering은 closure transfer의 convention과 그것을 감싼 `func.func`의 convention으로
caller·callee 계약을 검증하고 각 함수를 낮춘 뒤 둘 다 지운다. 낮춘 target 호출에는
이 속성을 복사하지 않는다.

[Representation/ABI 경계](ir.md#representationabi-경계)의 출구 검증은 `Cps`
worker, continuation, `done_k`, handler-dispatch의 result vector가 비어 있고 모든
CPS transfer가 `func.tail_call` 또는 `func.tail_call_indirect`로 끝나는지 검사한다.
이때 의미적 convention은 이미 소비되어 남아 있지 않다. 출구 이후의 backend-ready 검증은 exact signature,
`call_conv`, proper-tail operation만으로 같은 성질을 검사한다.
`Step`, trampoline, CPS control-result 역할의 `anyref`와
`__tribute_cps_control` private enum은 거부한다. Boxed source value, erased effect
payload, closure environment와 dispatch closure field에 쓰는 일반 `anyref`는 이
검사의 대상이 아니다.

## 핵심 설계

### `fn` operation: direct dispatch

직접형 입력은 위 규칙의 `operation_kind = "fn"` perform이며,
`tribute_control_to_cps`는 continuation을 만들지 않고 기존 `ability.call`
경로로 내린다:

```text
%result = ability.call %arg
  { ability_ref = core.ability_ref<{name = "Logger"}>, op_name = "log" }
```

`lower_ability_perform`은 enclosing callable의 exact `EvidenceDirect`/`Cps`
convention과 `CallableAbi`가 정한 첫 evidence parameter를 확인한다. Evidence와
같은 타입인 다른 parameter를 찾거나 본문에서 hidden input을 추론하지 않는다.
Convention이 없거나 slot/type이 잘못되면 lower하지 않고 ability boundary에서
거부한다. Source argument를 canonical payload로 pack한 뒤 이 evidence를 명시적으로
받는 `effect.dispatch_tail`을 만든다. Backend는 marker에서 `tr_dispatch_fn`을
선택해 일반 indirect call을 수행한다. 반환된 값은 erased source result다.

### `op` operation: continuation dispatch

공통 lowering은 payload packing 전에 exact resume의 frame 입력과 frame metadata에서
답 타입 `R`을 구하고 evidence·dispatch·resume의 연결을 검증한다.

```text
%product = pack %args into the canonical operation payload product
%payload = cast %product to anyref
effect.dispatch_cps %ev, %dispatch, %resume, %payload
  { ability_ref = core.ability_ref<{name = "State"}>, op_name = "get", answer_type = R }
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

Evidence 배열 참조와 두 closure 입력은 canonical 타입 동일성을 그대로 요구한다.
다른 배열 표기는 evidence 배열이 아니며, 일반 `wasm.structref`는 공통 closure
layout이 아니다. Payload 슬롯만 [물리적 참조 할당 가능성](wasm-backend.md)을
받아들인다. Payload packing의 `anyref` upcast가 no-op이면 concrete 참조가 남으므로,
등록된 struct·array 참조, `core.array` 표기, 소거된 ADT 표기가 `anyref` 슬롯을
만족한다. 고정 ABI의 입력 개수와 순서, 그리고 `answer_type`에서 독립적으로
구성되는 signature는 이 완화와 무관하게 유지한다.

의미 계약은 closure 타입 소거와 physicalization 변경 전에 검증한다. 최종 target
lowering은 실제 operand와 독립적으로 고정 signature를 구성하고 operand를 대조한다.
서로 다른 유효한 `R`도 동일한 물리 ABI를 가지며 resume의 frame은 별도 dispatch
입력이 아니다. `answer_type`은 일반 재귀 타입 변환에 참여하고 effect operation 제거
시 소비한다. Raw target call에 복제하지 않으며 결과를 반환하는 별도 dispatch
호환 경로를 두지 않는다.

정의, lambda, adapter, direct/indirect call, return, suffix, resume, handle은
ContinuationFrame을 같은 callable provenance로 전달한다. 내부 Dispatch를 effect operation이나
target handler ABI로 대체하지 않는다.

Effect point 이후의 코드는 이미 `%continuation` closure 안에 있으므로,
`ability.perform` 이후의 같은 function-body ops는 dead code가 된다.

이 lowering은 source kind를 재분류하지 않는다. 일반 `op` handler가 실제로
항상 tail-resumptive인지 분석하여 tail path로 최적화하는 작업은 표준 `"op"`
semantic lowering 이후의 별도 IR optimization이다.

### Root `main` delimiter

Root `main`은 하나뿐인 target-independent CPS delimiter다. Source residual-effect
계약은 기존 pure-or-`Io` entry를 유지하며 residual general effect는 backend 전에
거부한다. Nested module의 `main`은 일반 worker다.

Frontend가 root `main`을 Cps로 승격하면 그 정의에 source result type을
`tribute.root_source_result`로 기록한다. 이 속성이 root CPS 계약의 유일한 표식이며
root bridge 합성이 소비한다. Root wrapper는 항상 매개변수 없는 Direct 함수이므로
원래의 export 규약은 기록하지 않는다.

Target-independent 경계는 root wrapper가 source result
type의 completion cell과 이를 capture한 terminal `Done<R>` 및 terminal
`Dispatch<R>`를 담은 정확한 `ContinuationFrame<R>`를 소유한다는 추상 조합
계약만 정한다. Worker ABI의 두 번째 operand는 이 nominal frame이며 bare
`done_k`로 대체하거나 closure storage/arity에서 복원하지 않는다. Shared IR에서
CPS entry와 `done_k`의 result는 `core.never`이며,
`func.tail_call`과 `func.tail_call_indirect` verifier도 caller/callee의
`core.never` 일치를 검사한다.

Target signature lowering이 CPS signature를 native/Wasm empty-result signature로
바꾼 뒤 실제 wrapper와 결과 없는 ordinary call을 합성한다. Root `done_k`는 source
result를 cell에 정확히 한 번 쓰고 terminal dispatch는 root 밖 general operation transfer를
끝내며, wrapper는 이 둘을 immutable `ContinuationFrame<R>`로 materialize해 worker에
전달한 뒤 proper-tail-call chain이 끝나면 cell을 읽어 source result로 반환한다.
공통 `func.func_sig`와 `func.call`은 0개 또는 1개 결과를 지원한다. 논리 CPS
producer는 `[core.never]`를 유지하고 물리화는 정확한 Cps 결과만 `[]`로 바꾼다.
이 adapter는 answer-type polymorphism, trampoline, in-band sentinel 또는
control carrier가 아니다.

Root bridge 합성은 source `main`의 calling convention과 관계없이 그 함수를 하나의
root worker로 바꾸고, hidden 매개변수가 없는 Direct wrapper `main`을 합성한다
([진입점 계약](ir.md#representationabi-경계)). Wrapper는 worker 규약이 요구하는
입력을 스스로 만든다.

| Worker 규약 | Wrapper가 만드는 입력 | Wrapper의 결과 |
| --- | --- | --- |
| `Direct` | 없음 | worker 결과 |
| `EvidenceDirect` | target의 초기 evidence | worker 결과 |
| `Cps` | 초기 evidence와 completion cell을 capture한 `ContinuationFrame<R>` | call이 돌아온 뒤 읽은 cell 값 |

그래서 root마다 worker는 정확히 하나이고, wrapper는 export 규약을 보존하지 않는다.
Native entrypoint와 Wasm `_start`는 source calling convention을 읽지 않는다.

### `handle`: evidence extension + handler closures

`handle` lowering은 두 종류의 dispatch closure를 만든다. Environment를 포함한
물리적 입력은 다음과 같다:

- `handler_dispatch`: `(Evidence, Environment, Resume, Prompt, AbilityId,
  OperationIndex, Payload) -> ()`. General `op` handler용이며 closure environment가
  handle exit를 소유한다. `resume`과 non-resuming exit는 각각의 continuation으로
  indirect tail transfer한다.
- `tr_dispatch_fn`: `(Evidence, Environment, OperationIndex, Payload) -> anyref`.
  `fn` handler용이며 `anyref`는 erased source result다.

`resolve_evidence`는 explicit handler delimiter의 prompt와 dispatch closure를
소비하여 `effect.extend`를 만든다. Fresh prompt placeholder는 해당 delimiter에서
한 번만 materialize하며, body의 evidence 인자 사용을 확장된 값으로 치환한다.
호출이 가진 evidence 선택(`evidence_plan`, [ir.md](ir.md#direct-style-control))은
같은 pass가 그 호출의 evidence operand 앞에 `effect.mask`/`effect.dup`으로 만든다.
CPS legalization은 선택을 계산하거나 바꾸지 않고 만든 호출로 옮기기만 한다.
이 pass는 함수 signature나 본문 형상에서 hidden evidence를 추론하지 않는다.

```text
%ev2 = effect.extend %ev, %prompt_tag, %tr_dispatch_fn, %handler_dispatch
  { ability_ref = core.ability_ref<{name = "State"}> }
```

Concrete Marker layout은 backend가 소유한다. Native는
`__tribute_evidence_extend` ABI를 사용한다.

```text
struct Marker {
    ability_id: i32,
    prompt_tag: i32,
    tr_dispatch_fn: ptr,
    handler_dispatch: ptr,
    shadowed: ptr,
}
```

Evidence는 ability id 기준으로 정렬된 marker 배열이다. 각 칸은 그 ability의 가장
위 marker이며, marker는 자신이 가린 같은 ability의 marker를 `shadowed`로 가리킨다.
Handler 설치, `mask`, `dup`은 모두 새 evidence 값을 만들며 기존 값을 바꾸지 않는다.

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
| `shadowed` | 4 | `ptr` | marker of the same ability this one shadows, or null |

WasmGC uses the same field order and shared field identifiers, but its concrete
GC marker type stores the dispatch closures as `anyref` closure references
instead of native `ptr` values. Marker construction and field access stay
inside the target's helper implementations, so effect lowering never builds or
reads a marker directly.

Empty evidence is represented in high-level IR as an empty `core.array<Marker>`
or null evidence placeholder, and backend lowering turns that into the target
runtime representation. Native lowering maps it to `__tribute_evidence_empty()`.
<!-- markdownlint-disable-next-line MD033 -->
<a id="evidence-lookup"></a>
같은 `ability_id`의 handler를 다시 설치하면 새 marker가 기존 marker를 가린다
(shadow). Lookup은 가장 위의 marker를 고르고, 가려진 marker는 아래에 남는다.
호출의 evidence 선택이 이 순서를 row 구조에 맞추므로, 가장 위의 marker는 항상
현재 callable row의 명시 label에 묶인 handler다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="row-directed-evidence"></a>

#### Row 위치에 따른 evidence 선택

[type-inference.md](type-inference.md#호출의-evidence-선택)가 각 호출에 정하는 선택은
caller evidence의 ability별 marker 순서에 대한 두 연산으로 표현된다.

| 연산 | 뜻 | 쓰는 경우 (`k`) |
| --- | --- | --- |
| `mask L` | `L`의 가장 위 marker를 걷어 내 그 아래 marker를 드러낸다 | 0 |
| (없음) | 그대로 전달한다 | 1 |
| `dup L` | `L`의 가장 위 marker를 한 번 더 쌓는다 | 2 |

선택이 모두 그대로 전달인 호출은 evidence를 바꾸지 않는다. 선택은 직접 호출,
간접 호출, closure 호출, `resume`에 똑같이 적용한다. Perform은 별도의 분류 없이
가장 위의 marker를 사용한다. Typechecking이 perform의 label을 언제나 둘러싼
callable 또는 handle body row의 명시 label로 확정하기 때문이다.

Handler와 evidence의 연결은 다음과 같다.

- **Handle body:** 처리하는 label마다 `effect.extend`로 새 marker를 쌓은 evidence를 받는다.
- **Handler arm, `do` arm:** handle을 설치한 지점의 evidence(바깥)로 실행한다.
  `handler_dispatch`와 `tr_dispatch_fn` closure는 이 evidence를 environment에
  capture하며, dispatch가 넘기는 perform 지점 evidence를 arm에 전달하지 않는다.
- **Arm 안의 `resume`:** [abilities.md](abilities.md#resume과-handler-선택)에 따라
  arm 본문의 resume은 자기 handle body의 evidence를, arm 안 lambda의 resume은 그
  lambda가 받은 evidence를 continuation에 넘긴다.
- **재개된 frame:** 포착된 경로의 각 frame은 resume이 넘긴 handle body evidence에서
  자기 위치까지의 호출 선택과 그 사이에 설치된 handler를 다시 적용한 evidence를
  본다. 포착 시점의 evidence를 그대로 재사용하지 않으며, resume이 넘긴 evidence를
  모든 frame에 그대로 흘리지도 않는다.

두 target의 effect lowering은 같은 evidence runtime helper ABI를 호출한다.
아래는 native 표기이며, Wasm은 `ptr` evidence 대신 GC evidence 배열 참조를,
dispatch closure `ptr` 대신 canonical closure 참조를 쓴다.

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
__tribute_evidence_mask(ev: ptr, ability_id: i32) -> ptr
__tribute_evidence_dup(ev: ptr, ability_id: i32) -> ptr
__tribute_evidence_lookup_tr(ev: ptr, ability_id: i32) -> ptr
__tribute_evidence_lookup_handler(ev: ptr, ability_id: i32) -> ptr
```

`extend`는 같은 ability의 기존 marker를 새 marker의 `shadowed`로 둔다. `mask`는
그 칸을 `shadowed`로 바꾸고, `shadowed`가 null이면 칸을 지운다. `dup`은 가장 위
marker의 복사본이 원본을 가리게 한다. 없는 ability를 `mask`하거나 `dup`하는 것은
compiler bug이며 runtime은 이를 검사하지 않는다.

### `ability.handle_dispatch`

`ability.handle_dispatch`는 runtime dispatch loop가 아니다. Effect 발생 시점에서
이미 handler closure로 tail-call되므로,
`resolve_evidence`가 body의 evidence 인자를 명시적인 extended evidence로 대체한 뒤,
`lower_handle_dispatch`는 resultless body를 바깥 block에 옮기고 delimiter를 제거한다.
정상 완료와 resume하지 않는 handler exit의 transfer는 shared CPS legalization이 구성한다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="shared-middle-end-pipeline"></a>

## 공통 middle-end 파이프라인

Callable/control과 effect 관련 pass의 순서는 다음과 같다:

모든 source 함수는 이 shared route와 target ABI boundary를 통과한다. Root wrapper의
생성은 exact root contract에 따르며, 별도의 호환 closure-lowering 경로는 두지 않는다.

```text
ast_to_ir (tribute_control callable/control + ordinary value IR)
→ tribute_control_to_cps
→ lower_closure_lambda
→ lower_ability_perform
→ resolve_evidence
→ lower_handle_dispatch
→ effect ABI verification
── representation/ABI 경계 ──
→ target ABI validation and CPS signature physicalization
→ root and entry bridge composition
→ lower_closures_in_func
→ target evidence lowering (native 또는 Wasm)
→ finalize_closure_storage_layout
→ boundary exit verification
── target dialect lowering ──
→ proper-tail lowering과 target dialect conversion
→ backend-ready verification
```

`tribute_control_to_cps`의 출력은 physical `func.*`/`closure.*` callable 표면과
logical `ability.*` dispatch 표면이다. Shared ability/evidence pass와 strict target
ABI validation은 convention-proven `closure.closure` callable type을 그대로
소비한다. 그 뒤 경계 안에서 closure operation과 모든 type-bearing storage
surface를 canonical `_closure` layout으로 함께 바꾸고 target evidence/runtime
lowering이 이를 소비한다. 경계 이후의 proper tail transfer lowering은 이 layout을
명시적 layout 식별자로 구별하며 이름으로 판별하지 않는다. Closure lowering이
`closure.new`를 `_closure` pack으로 바꾸면 남은 사용처는
`core.unrealized_conversion_cast`로 exact closure type을 유지한다. Storage
finalization이 closure type을 `_closure`로 바꾸면 이 cast는 identity가 되어 제거된다.
Pack에 별도 provenance 속성을 붙이지 않으며, 이는 semantic type equivalence를
만들지 않는다.

## Effect ABI Boundary

The `effect` dialect is the target-independent boundary between language
semantics and concrete runtime layout.

Operations:

- `effect.extend(evidence, prompt_tag, tr_dispatch_fn, handler_dispatch)
  { ability_ref } -> evidence`
- `effect.mask(evidence) { ability_ref } -> evidence`
- `effect.dup(evidence) { ability_ref } -> evidence`
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
- Payload value는 shared dispatch lowering에서 canonical operation product 하나로
  pack한다. Zero/one argument도 같은 product contract를 사용하며 null 또는 in-band
  sentinel로 대체하지 않는다. 각 field를 dynamic storage에 맞춘 뒤 product 전체를
  `anyref`로 erase한다. Frontend는 source-logical `tribute_control.perform`의
  개별 operand를 유지한다. Handler unpacking은 같은 exact product layout을 쓴다.

## Backend Implications

### Native

Evidence runtime은 `tribute-runtime`의
`__tribute_evidence_*` C ABI 함수로 제공되고, native effect ABI lowering은
`effect.*`를 marker lookup helper, runtime evidence extension, closure
decomposition, and indirect calls로 변환한다.

### WasmGC

WasmGC도 같은 shared middle-end를 사용한다. `wasm/evidence_to_wasm`은
representation/ABI 경계 안에서 native와 같은 구조로 `effect.*`를 낮춘다.
`effect.extend`는 `__tribute_evidence_extend` 호출이 되고, `effect.dispatch_tail` /
`effect.dispatch_cps`는 `__tribute_evidence_lookup_tr` / `__tribute_evidence_lookup`,
closure field 접근, `func.call_indirect` 또는 proper-tail `func.tail_call_indirect`가
된다. Wasm dialect lowering이 이를 `wasm.call_indirect` /
`wasm.return_call_indirect`로 바꾼다. Helper 구현은 target runtime으로서 출구 뒤에
GC 배열 위의 `wasm.func`로 바인딩된다.
