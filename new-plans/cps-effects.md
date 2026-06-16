# CPS 기반 Effect Handling 파이프라인

이 문서는 현재 구현 기준의 ability lowering 전략을 설명한다.

핵심 전략은 **tail-call CPS + evidence-based handler dispatch**이다.
WasmGC yield bubbling, `YieldResult` 중심 trampoline, `cont.*` dialect 직접
lowering은 현재 경로가 아니다.

## 핵심 설계

### `fn` operation: direct dispatch

`fn`으로 선언된 ability operation은 tail-resumptive임을 선언부에서 보장한다.
호출 지점은 continuation을 만들지 않고 `ability.call`로 내려간다.

```text
%result = ability.call %arg
  { ability_ref = @Console, op_name = @print }
```

`lower_ability_perform`는 이를 다음 형태로 낮춘다:

```text
%marker = ability.evidence_lookup %ev { ability_ref = @Console }
%tr_dispatch = adt.struct_get %marker, 2
%fn = adt.struct_get %tr_dispatch, 0
%env = adt.struct_get %tr_dispatch, 1
%op_idx = arith.const <hash(Console, print)>
%result = func.call_indirect %fn(%ev, %env, %op_idx, %arg_anyref)
```

즉 `fn` operation은 CPS 변환, continuation allocation, resume dispatch를
우회한다.

### `op` operation: tail-call CPS dispatch

`op`으로 선언된 general operation은 명시적인 continuation closure와 함께
`ability.perform`으로 내려간다.

```text
%result = ability.perform %continuation, %arg
  { ability_ref = @State, op_name = @get }
```

`lower_ability_perform`는 evidence에서 `handler_dispatch` closure를 찾고,
그 closure로 tail-call한다:

```text
%marker = ability.evidence_lookup %ev { ability_ref = @State }
%handler = adt.struct_get %marker, 3
%fn = adt.struct_get %handler, 0
%env = adt.struct_get %handler, 1
%op_idx = arith.const <hash(State, get)>
%result = func.call_indirect %fn(%ev, %env, %continuation_anyref, %op_idx, %arg_anyref)
func.return %result
```

Effect point 이후의 코드는 이미 `%continuation` closure 안에 있으므로,
`ability.perform` 이후의 같은 function-body ops는 dead code가 된다.

### 중첩 표현식의 CPS lifting

CPS 호출이 더 큰 표현식 안에 중첩된 경우 `ast_to_ir`는 호출을 직접
lowering하지 않는다. 대신 현재 평가 영역 안에서 첫 CPS 호출을 synthetic
let binding으로 끌어올린 뒤 기존 block CPS lowering을 적용한다.

```text
consume(effectful(), pure_arg)

→ let __cps_tmp = effectful()
  consume(__cps_tmp, pure_arg)
```

이 변환은 소스의 좌에서 우 평가 순서를 보존한다. 호출 callee와 arguments,
tuple/constructor/record 요소, case scrutinee처럼 항상 평가되는 strict
subexpression만 현재 영역으로 끌어올린다.

다음 위치는 별도의 control-flow 또는 effect 경계이므로 바깥 영역으로
hoist하지 않는다.

- short-circuit 연산의 RHS
- case arm과 guard
- lambda body
- handle body와 handler arm

각 영역은 진입 시 자체적으로 같은 CPS lifting을 수행한다. 따라서 실행되지
않을 branch의 effectful call이 미리 실행되거나, handler boundary 밖의
continuation에 잘못 포함되어서는 안 된다.

### `handle`: evidence extension + handler closures

`handle` lowering은 두 종류의 dispatch closure를 만든다.

- `handler_dispatch`: `(k, op_idx, value) -> anyref`
  - general `op` handlers용
  - `resume`은 continuation closure 호출로 lowering된다.
- `tr_dispatch_fn`: `(op_idx, value) -> anyref`
  - `fn` handlers용
  - continuation 없이 handler 결과가 inline result가 된다.

`resolve_evidence`는 handler boundary에서 새 marker를 만들어 evidence를
확장한다.

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

### `ability.handle_dispatch`

현재 구현에서 `ability.handle_dispatch`는 runtime dispatch loop가 아니다.
Effect 발생 시점에서 이미 handler closure로 tail-call되므로,
`lower_handle_dispatch`는 body result에 `done` handler를 적용하는 역할만 한다.

## Shared Middle-End Pipeline

현재 shared pipeline의 핵심 순서는 다음과 같다.

```text
ast_to_ir
→ lower_closure_lambda
→ intrinsic_to_arith
→ closure_lower
→ lower_ability_perform
→ convert_tail_resumptive
→ resolve_evidence
→ lower_handle_dispatch
→ backend-specific lowering
```

`ast_to_ir` 단계에서 effectful function과 closure는 evidence parameter와
CPS calling convention을 반영한 IR로 생성된다. 이후 backend는 이미 lowered된
`func.call_indirect`, evidence runtime calls, closure representation을 각 타겟에
맞게 낮춘다.

## Backend Implications

### Native

Native target은 현재 주 개발 경로다. Evidence runtime은 `tribute-runtime`의
`__tribute_evidence_*` C ABI 함수로 제공되고, native lowering은 marker field
접근을 runtime lookup helper로 바꾼다.

### WasmGC

WasmGC도 원칙적으로 같은 shared middle-end를 사용할 수 있다. 다만 현재
WasmGC backend에는 이전 yield bubbling/trampoline 설계의 흔적이 남아 있으므로,
WasmGC를 다시 주요 경로로 올리기 전에 다음을 정리해야 한다.

- `Step`, `Continuation`, `ResumeWrapper` builtin 타입의 실제 필요 여부
- Marker field 문서와 구현의 4-field layout 일치
- closure table index 기반 `call_indirect`가 handler/tr dispatch closure에
  일관되게 적용되는지 검증

## 폐기된 접근

다음 접근은 현재 구현 기준의 active path가 아니다.

- WasmGC yield bubbling
- Koka-style `YieldResult { Done, Shift }`를 effectful return type으로 전파
- `cont_to_yield_bubbling` pass
- `cont.*` dialect를 libmprompt 또는 stack switching으로 직접 lowering

관련 과거 설계는 git history에서 확인할 수 있지만, 새 구현 작업의 기준으로
사용하지 않는다.
