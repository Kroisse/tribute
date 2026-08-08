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

### CPS 변환 전 호출 대상 형상

입력 형상은 [ir.md](ir.md#direct-style-control)의 operation/type 계약을 따른다.
모든 callable은 exact `tribute.calling_convention` code 0/1/2를 가진 logical
type과 source parameter/result만 사용하며 evidence, environment, `done_k`는
아직 없다.

`tribute_control_to_cps`는 closure extraction 전에 전체 callable graph를 하나의
단위로 변환한다:

| Convention | Physical parameter | Result |
| ---- | ---- | ---- |
| `Direct` | 소스 parameter | 소스 result |
| `EvidenceDirect` | evidence 뒤 소스 parameter | 소스 result |
| `Cps` | evidence, `Done<R>`, `Dispatch<R>`, 소스 parameter 순서 | logical `core.never`, physical empty result |

Parameter 순서는 기존 `CallableAbi` 순서다. CPS continuation 형상은 모두 exact
`tribute.calling_convention = 2`를 가진 closure이며 다음과 같다.

```text
Done<R>             = closure(core.func<never, R>)
Parent<R>           = immutable private struct { done: Done<R>, dispatch: Dispatch<R> }
Completion<X, R>    = closure(core.func<never, Evidence, Parent<R>, X>)
ResumeExact<I, R>   = closure(core.func<never, Evidence, Parent<R>, I>)
Resume<R>           = closure(core.func<never, Evidence, Parent<R>, anyref>)
Dispatch<R>         = closure(core.func<never,
                         Evidence, Resume<R>, i32 prompt_tag, i32 ability_id,
                         i32 op_id, anyref payload>)
```

이 typed CPS control callback ABI에서 `anyref` 위치는 `Resume<R>`의 source
operation input과 payload뿐이다. `Done`, completion, resumption, dispatch,
control result의 erase/cast는 금지한다.
`Flow<R> = (Done<R>, Dispatch<R>)`, `Context<R> = (Evidence, Parent<R>)`는
컴파일러 내부 pair이고 `Parent<R>`만 이를 pack하는 immutable nominal SSA 값이다.
재개 callback은 명시적 `Context<R>`를 받아 nearest prompt와 parent dispatch를
함께 복원한다. mutation, global slot, carrier, ordinary-call fallback과 control
result는 없으며 CPS worker/continuation은 empty result만 가진다.

Conversion은 callable graph 전체를 검증한 뒤 physical symbol/type을 함께
rewrite한다.

- definition/lambda/`func_ref`는 `func.func` 또는 typed `closure.closure`가 된다.
  closure type은 exact convention attribute를 보존하고 closure lowering만
  environment를 interpose한다.
- Direct/EvidenceDirect call은 일반 call이다. Boundary `R`에서 source result `X`인
  CPS call은 `Completion<X,R>`, `Done<X>`, `Dispatch<X>` adapter를 만들어
  resume 시 받은 `Context<R>`로 fresh `Parent<X>`를 구성한 뒤 proper-tail
  transfer한다.
- CPS return은 `Done<R>(value)`로 tail transfer하며 result나 `func.return`이 없다.
- typed closure를 pointer로 바꿀 때 `func.indirect_call_signature`에 exact physical
  ABI를 보존한다. target은 이 provenance만 사용해 zero-size parameter를 투영한다.
  convention/type/symbol 불일치는 conversion failure다.

알려진 CPS target으로의 최종 이전은 `func.tail_call`, closure, continuation,
`done_k`처럼 동적인 target으로의 이전은 `func.tail_call_indirect`를 사용한다.
Shared IR의 caller/callee result는 `core.never`이고 target signature의 result
vector는 비어 있어야 한다.
생성한 continuation, `done_k`, handler-dispatch function에도
`tribute.calling_convention = 2`를 붙여 backend-ready verifier가 semantic role을
식별하게 한다.

Environment 뒤 parameter 순서는 Direct `source...`, EvidenceDirect
`evidence, source...`, Cps `evidence, Done<R>, Dispatch<R>, source...`이며 closure
lowering 뒤에는 environment를 evidence 다음에 넣는다. Function/call은 operation
attribute, closure value는 outer closure type attribute를 convention provenance로
사용한다.

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
  `op -> I`에는 exact one-shot `ResumeExact<I,R>`를 만들고 input `I`만 source
  data로 box한 `Resume<R>`를 현재 `Dispatch<R>`에 tail transfer한다. `op -> Never`에는
  suffix를 capture하지 않고 typed reject `Resume<R>`를 공급한다.
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
   frame에 다시 진입한다. Non-resumed path는 해당 frame을 포기한다.

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
`func.unreachable`인 typed reject continuation을 전달한다. Null, in-band
sentinel, 임의의 `anyref`는 대체할 수 없으며 호출되면 trap한다.

### 적법화 경계

`tribute-control-pre-cps` named boundary는 검증된 `tribute_control.*`과 일반
value/structured dialect만 허용한다. Frontend 적합성 검사는 `func.*`,
`closure.*`, `core.func`, `closure.closure`, 기존 `ability.*`, `effect.*`,
legacy CPS-dispatch operation을 모두 거부한다. Partial rewrite
도중에는 logical/physical operation이 일시적으로 공존할 수 있지만 이 상태는
named pre-CPS boundary가 아니다.

성공한 shared conversion은 defining rule이
`illegal_dialect("tribute_control")`인 partial
`tribute-control-post-cps` target과 Tribute type walk를 검증한다. 남은
`tribute_control` operation 또는 `callable`/`resume_token` type은 source
location에서 conversion failure가 된다. 이 경계에는 일관된 physical
`func.*`/`closure.*`/`core.func` graph와 logical `ability.*` dispatch 표면만
남는다. 이후 기존 `lower_closure_lambda`, `prepare_closure_lowering`,
`lower_closures_in_func`가 closure 표면을 소비하고, `lower_ability_perform`,
`resolve_evidence`, `lower_handle_dispatch`가 `ability.*`를 `effect.*`까지
낮춘다. Native와 Wasm의 evidence pass가 `effect.*`를 backend ABI로 제거한다.
Backend-ready Tribute boundary는 남은 `tribute_control.*`, `ability.*`,
`effect.*`를 각각 독립적으로 거부한다.

논리적 CPS 함수는 source result를 직접 반환하지 않는다. 완료 값은 `done_k`의
인자로만 전달되고 logical control result는 `core.never`다. Physical control
lowering은 이를 empty-result 함수와 proper tail transfer로 바꾸며 반환되는
control value를 만들지 않는다. Resume하지 않는 handler arm도
`Escape`를 반환하는 대신 해당 handle의 exit continuation으로 직접 tail
transfer하므로 completion region과 포기한 suffix를 구조적으로 건너뛴다.

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

공통 lowering은 이를 target-independent effect ABI operation으로 바꾼다.

```text
%payload = cast %arg to anyref
%result = effect.dispatch_tail %ev, %payload
  { ability_ref = @Logger, op_name = @log }
```

Native lowering은 이 ABI operation을 evidence lookup과 indirect call로 낮춘다.

```text
%marker = ability.evidence_lookup %ev { ability_ref = @Logger }
%tr_dispatch = adt.struct_get %marker, MarkerField::TrDispatchFn
%fn = adt.struct_get %tr_dispatch, 0
%env = adt.struct_get %tr_dispatch, 1
%op_idx = arith.const <hash(Logger, log)>
%result = func.call_indirect %fn(%ev, %env, %op_idx, %arg_anyref)
```

즉 `fn` operation은 CPS 변환, continuation allocation, resume dispatch를
우회한다.

### `op` operation의 typed tail-call CPS dispatch

직접형 입력은 위 규칙의 `operation_kind = @op` perform이며,
`tribute_control_to_cps`는 block suffix에서 exact one-shot
`ResumeExact<I,R>`와 result-indexed `Resume<R>`를 구성해 `ability.perform`으로 내린다.

```text
ability.perform %evidence, %dispatch, %resume, %arg
  { ability_ref = @State, op_name = @get }
```

공통 lowering은 이를 target-independent effect ABI operation으로 바꾼다.

```text
%payload = cast %arg to anyref
effect.dispatch_cps %ev, %dispatch, %resume, %payload
  { ability_ref = @State, op_name = @get }
```

Native/Wasm lowering은 evidence에서 nearest prompt tag를 찾은 뒤 명시적으로 받은
typed dispatch closure를 tail-call한다.

```text
%prompt = ability.evidence_lookup %ev { ability_ref = @State }
func.tail_call_indirect %dispatch(
  %ev, %resume, %prompt, <ability_id>, <op_id>, %arg_anyref)
```

Effect point 이후의 코드는 이미 `%continuation` closure 안에 있으므로,
`ability.perform` 이후의 같은 function-body ops는 dead code가 된다.

이 lowering은 source kind를 재분류하지 않는다. 일반 `op` handler가 실제로
항상 tail-resumptive인지 분석하여 tail path로 최적화하는 작업은 표준 `@op`
semantic lowering 이후의 별도 IR optimization이다.

### Root `main` delimiter

Root `main`만 target-independent CPS delimiter다. Nested module의 `main`은 일반
worker이며 residual general effect는 backend 전에 거부한다. CPS entry와 root
`done_k`는 shared IR에서 `core.never`, target에서는 empty result다.

Direct/EvidenceDirect wrapper는 completion cell과 root `done_k`를 만들고 ordinary
call로 CPS worker를 시작한다. `done_k`는 source result를 한 번 저장하고, wrapper는
tail chain 뒤 cell을 읽어 반환한다. Shared `func.call` result 계약은 유지하며
trampoline, carrier, sentinel과 call 뒤 return fallback은 없다.

### `handle`: immutable completion + dispatch closure

Body type이 `B`, handle answer가 `A`, enclosing result가 `O`이면
`after_handle: Completion<A,O>`와 `completion: Completion<B,A>`를 만든다. 정상 body
completion은 둘을 순서대로 한 번 실행한다. Token은 받은 fresh
`Evidence, Parent<A>`로 `Flow<B>`와 local dispatcher를 다시 만들어 raw `Resume<B>`에
전달한다. 따라서 nested resume도 immutable suffix stack을 보존한다. Non-resumed arm은
`Parent<A>.done(answer)`로 직접 tail transfer한다. Foreign prompt도 capture한
`Parent<A>.dispatch`와 재구성한 `Flow<B>`로 forward한다.

`Dispatch<R>`는 `(Evidence, Resume<R>, prompt_tag, ability_id, op_idx, payload) ->
Never`인 immutable general-`op` dispatch chain이다. `tr_dispatch_fn`은 `fn`
operation의 `(op_idx, value) -> anyref` closure이며, 이 `anyref`는 erased source
result일 뿐 control 값이 아니다.

`resolve_evidence`는 concrete storage를 보지 않고 다음 effect ABI로 handler를
설치한다.

```text
%ev2 = effect.extend %ev, %prompt_tag, %tr_dispatch_fn, %metadata
  { ability_ref = @State }
```

Evidence는 ability id로 정렬된 immutable marker 배열이다. 같은 ability의 nested
handler는 nearest marker를 선택한다. Marker storage와 field 번호는 backend-private다.
Native는 evidence runtime ABI로, WasmGC는 target closure 표현으로 이를 낮춘다.
`effect.dispatch_cps`는 marker closure를 읽지 않고 명시적 `Dispatch`를 분해해
proper tail transfer한다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="shared-middle-end-pipeline"></a>

## 공통 middle-end 파이프라인

Callable/control과 effect 관련 pass의 순서는 다음과 같다:

```text
ast_to_ir (tribute_control callable/control + ordinary value IR)
→ tribute_control_to_cps
→ lower_closure_lambda
→ prepare_closure_lowering
→ lower_closures_in_func
→ lower_ability_perform
→ resolve_evidence
→ lower_handle_dispatch
→ effect ABI verification
→ backend-specific effect/signature/tail-call lowering
→ backend-ready verification
```

`tribute_control_to_cps`는 physical callable과 logical `ability.*` dispatch를 만든다.
Closure pass가 전자를, ability/evidence pass가 후자를 `effect.*`까지 소비한 뒤
backend가 evidence runtime과 proper tail transfer로 제거한다.

## Effect ABI 경계

`effect` dialect는 language semantics와 runtime layout 사이의 target-independent
경계다. `effect.extend`, source-result `effect.dispatch_tail`, resultless
`effect.dispatch_cps`만 허용한다. 공통 ability-dispatch 뒤 `ability.*`는, backend
ABI lowering 뒤 `effect.*`는 illegal이다. `dispatch_cps`는 명시적 Dispatch closure로
proper indirect tail transfer하며 control result를 만들지 않는다. Wasm의 일반
source-data indirect call은 계속 `wasm.call_indirect`를 사용한다.
