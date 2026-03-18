# CPS 기반 Effect Handling 파이프라인 설계

> 이 문서는 설계 탐색 단계이며, Option A (AST-level CPS)를 권장 방향으로 채택.

## Context

현재 `cont_to_yield_bubbling` 패스(~2500줄, 8파일)는 SSA 형태의 TrunkIR에서
continuation을 사후 추출하는 모놀리식 패스다. 주요 복잡도 원천:

| 구성요소                               | 줄수 | 복잡도 원인                             |
| -------------------------------------- | ---- | --------------------------------------- |
| Resume function 추출 + SSA value remap | ~650 | SSA 값 clone + 참조 remap               |
| Chain function 생성                    | ~500 | 순차 effectful call의 Done/Shift 분기   |
| Truncation                             | ~500 | 추출 후 dead code 정리                  |
| Handler dispatch                       | ~600 | scf.loop + prompt tag 매칭              |
| Live variable analysis                 | ~300 | 비교적 단순한 알고리즘                  |

**핵심 아이디어**: continuation을 명시적 closure로 표현하면 yield bubbling이
단순해진다. CPS 변환 위치에 따라 두 가지 접근이 가능.

## 전제 조건

- cont_to_trampoline 제거 완료 (머지됨)
- 문법 수준 TR 구분은 미래 참고 (현재는 tail_resumptive 패스 유지)

## 핵심 설계: ability.handle + ability.perform

### ability.perform

`cont.shift`를 대체하는 high-level op. Ability operation 수행 + explicit
continuation:

```text
ability.perform @State, @get, [%args], %continuation_closure
```

- Lowering: evidence lookup → ShiftInfo/Continuation 구성 → `YieldResult::Shift` 반환
- Block terminator 성격 (CPS에서 이후 코드는 없음 — continuation closure에 포함)

### ability.handle (축소된 형태)

CPS에서 done handler는 body의 최외곽 continuation closure가 됨.
따라서 ability.handle에서 done region이 분리되어, **handler dispatch만 담당**:

```text
// 1. done handler → closure로 생성
%done_k = closure.new @handle_done, [captures]

// 2. evidence 설정
%tag = call @__tribute_next_tag()
%new_ev = call @__tribute_evidence_extend(%ev, %marker)

// 3. CPS body를 done_k와 함께 호출 → YieldResult 반환
%yr = call @handle_body(%new_ev, %done_k)

// 4. handler dispatch (Shift만 처리, Done은 body CPS chain에서 처리됨)
ability.handle_dispatch %yr, %tag {
  handler @get {
    ^bb0(%k: Closure, %value):
      func.call_indirect %k(42)    // resume = call continuation closure
  }
}
```

### cont dialect 제거

CPS에서 cont의 모든 역할이 대체됨:

| cont op               | CPS 대체                                        |
| --------------------- | ----------------------------------------------- |
| cont.shift            | ability.perform + continuation closure          |
| cont.resume           | func.call_indirect %k(%value)                   |
| cont.drop             | closure 미호출 (RC/GC 처리)                     |
| cont.push_prompt      | evidence extend + CPS body call                 |
| cont.handler_dispatch | ability.handle_dispatch (scf.loop)              |
| cont.done             | done handler가 CPS continuation closure로 변환  |
| cont.suspend/yield    | ability.handle_dispatch의 handler arm           |

## Option A: CPS at ast_to_ir (채택)

### 개요

ast_to_ir에서 effectful 함수 본문을 CPS로 생성. Lexical scoping이
SSA-level code extraction을 완전히 대체.

```text
ast_to_ir (CPS for effectful bodies, closure.lambda 생성)
→ evidence_params → closure_lower → evidence_calls
→ tail_resumptive → ability lowering (ability.perform → YieldResult)
→ lower_closure_lambda → lambda_flattening → dce → resolve_casts → emit
```

### 장점

- **SSA 추출/remap 제거**: ~1150줄(shift_lower + call_lower) 해당 복잡도 소멸
- **Truncation 불필요**: CPS에서 구조적으로 불필요
- **Live var analysis 불필요**: lexical scoping의 `collect_free_vars` 재사용
- **기존 lambda infrastructure 재사용**: lower_lambda, closure.new, closure_lower
- **조건 분기/case arm 안의 effect를 자연스럽게 처리**

### 단점

- **ast_to_ir 복잡화**: CPS block splitting 로직 추가. 단, `closure.lambda`
  도입으로 lambda lifting은 분리되어 복잡화가 **대폭 완화**됨.
- **CPS calling convention**: effectful 함수 시그니처 변경 (continuation param),
  evidence 패스와의 상호작용 검증 필요
- **done handler → closure 변환**: handle expression lowering이 현재보다 복잡

### 구현 포인트

#### expr.rs: `lower_block_cps()`

- 각 statement를 순회, effectful call 감지
- Effect point 발견 시: 나머지 stmts + value를 body region으로 만들어
  `closure.lambda` 생성 (lambda lifting은 별도 pass가 처리)
- 자연스러운 재귀: 분기/case arm 내 effect도 동일하게 처리

#### handle.rs: handle expression

- done handler → `closure.lambda`로 생성 (CPS body의 outermost continuation)
- push_prompt body → CPS 모드로 lowering (done_k를 continuation으로 전달)
- handler arms → ability.handle_dispatch region으로 생성

#### lambda.rs: 소스 lambda도 `closure.lambda` 생성

- 기존 inline lambda lifting 로직을 `closure.lambda` 생성으로 교체
- Closure conversion은 별도 pass (`lower_closure_lambda`)로 이동

## closure.lambda: 고수준 lambda op

### 동기

Option A의 주요 단점은 ast_to_ir 복잡화인데, 이는 ast_to_ir가
lambda lifting을 inline으로 수행하기 때문이다. 고수준 `closure.lambda` op을
도입하면 ast_to_ir는 의도만 표현하고, lifting은 별도 pass에서 처리.

### 설계

```text
// closure.lambda: body region + captures → closure value
%k = closure.lambda [%x, %y] {
  ^bb0(%param: anyref):
    %r = arith.add %x, %param
    scf.yield %r
}

// lower_closure_lambda pass에 의해 변환:
func.func @__lambda_0(%ev: Evidence, %env: anyref, %param: anyref) -> anyref {
  %x = adt.struct_get %env, 0 : anyref
  %y = adt.struct_get %env, 1 : anyref
  %r = arith.add %x, %param
  func.return %r
}
%k = closure.new @__lambda_0, [%x, %y]
```

**위치**: closure dialect (tribute-ir) — language-agnostic하며 `closure.new`의
pre-lifted 버전으로 자연스럽게 위치.

### CPS 생성 단순화

ast_to_ir에서 CPS continuation 생성이 극적으로 단순해짐:

```text
fn lower_block_cps(stmts, value):
  for stmt in stmts:
    if is_effectful(stmt):
      let body_region = lower_remaining_block(rest_stmts, value)
      let captures = collect_captured_values()
      emit closure.lambda(captures, body_region)
      emit ability.perform(..., closure_value)
      return
    else:
      lower_stmt(stmt)
  lower_expr(value)
```

### 기존 소스 lambda 전환

CPS continuation뿐 아니라 **소스 레벨 lambda도 `closure.lambda`로 전환** 가능:

```tribute
let f = fn(x) { x + captured_var }
```

현재: ast_to_ir → (inline lambda lift) → `closure.new @lambda_0, [%captured_var]`

전환 후: ast_to_ir → `closure.lambda [%captured_var] { ^bb0(%x): ... }`
→ lower_closure_lambda pass → `closure.new @lambda_0, [%captured_var]`

이로써:

- ast_to_ir의 lambda lifting 로직 (~200줄, lambda.rs) 분리
- 소스 lambda와 CPS continuation이 동일한 경로로 처리
- Closure conversion이 독립적으로 테스트 가능한 별도 pass가 됨

### lower_closure_lambda pass

**위치**: `crates/tribute-passes/src/lower_closure_lambda.rs` (신규)

역할:

1. 모듈 내 모든 `closure.lambda` op 수집
2. 각 lambda의 body region을 top-level `func.func`으로 추출
3. Captures를 env struct로 패킹
4. `closure.lambda` → `closure.new` 치환

기존 `lower_lambda()` (lambda.rs)의 closure conversion 로직을
TrunkIR pass로 재구성.

## 공통 구성요소

### Lambda Flattening / Inlining

Done 경로에서 single-use closure를 인라인:

```text
// Before: closure 생성 + 즉시 호출
%k = closure.new @cont_fn, %env
func.call_indirect %k(%value)

// After: body 직접 삽입, closure allocation 제거
// @cont_fn의 body가 인라인됨
```

Lambda flattening은 **general inlining의 특수한 경우** (single-use closure).
Heuristic 불필요 — single-use closure는 항상 flattening이 이득.

- **단기**: 별도 lambda flattening pass로 구현 (패턴 매칭만으로 가능, 단순)
- **장기**: General inlining pass를 만들 때 single-use closure 인라인을 포함하여
  통합. Lambda flattening pass는 제거.

### YieldResult 타입 시스템

기존 types.rs 재사용:

- YieldResult { Done(anyref), Shift(ShiftInfo) }
- ShiftInfo { value, prompt, op_idx, continuation }
- Continuation { resume_fn, state } → CPS에서는 closure로 단순화 가능

### tail_resumptive 상호작용

handler 쪽 (cont.suspend → cont.yield)에서 동작 — CPS 변환과 직교.

- cont.yield arm: handler 값 함수 직접 호출 → continuation 즉시 호출
- lambda_flattening이 즉시 호출 패턴 인라인

## 결정적 차이: 조건 분기 안의 effect

### 문제 케이스

```tribute
fn foo() ->{State} Int {
  let x = State::get()
  if x > 0 {
    State::set(x - 1)      // ← 분기 안의 effect
    x
  } else { 0 }
}
```

### Option A: 자연스럽게 처리됨

AST에서 각 분기는 독립된 expression. `lower_block_cps`가 재귀적으로
각 분기에 CPS splitting 적용:

```text
fn foo(k) {
  ability.perform @State, @get, [], closure.lambda @foo_k1, [k]
}
fn foo_k1(ev, env, x) {             // continuation after get
  if x > 0 {
    // if-branch에 effect → 재귀적 CPS splitting
    ability.perform @State, @set, [x-1], closure.lambda @foo_k2, [x, env.k]
  } else {
    func.call_indirect env.k(0)      // pure → 직접 continuation 호출
  }
}
fn foo_k2(ev, env, _) {             // continuation after set
  func.call_indirect env.k(env.x)
}
```

분기 안의 effect에 특별한 처리가 필요 없음 — AST 순회가 자연스럽게 재귀.

### Option B: 본질적으로 어려움

TrunkIR에서는 `scf.if`의 nested region 안에 `cont.shift`가 존재:

```text
%yr = scf.if %cond {
  cont.shift(tag, [x-1])       // nested region 안의 shift!
  scf.yield %x
} else {
  scf.yield %zero
}
// ... if 이후의 코드
```

문제점:

- Continuation이 "이 branch의 나머지" + "if 이후의 코드"를 포함해야 함
- `scf.yield`를 통한 값 흐름을 가로질러 추출해야 함
- **현재 `live_vars.rs`가 nested region 내 shift를 명시적으로 거부**
  (이것은 이 문제가 SSA 레벨에서 본질적으로 어렵다는 증거)
- Option B도 같은 제약에 부딪힘

## 장기 방향

### cont dialect 제거

CPS 접근에서 cont dialect의 모든 ops가 대체됨.
ability.handle_dispatch + ability.perform이 effect semantics를 담당.

### ability dialect 정리

- evidence_lookup/extend → `func.call @__tribute_*` (직접 호출)
- handler_table/entry → adt 기반 정적 데이터
- **ability.perform + ability.handle_dispatch**: 의미 있는 semantic markers로 유지

### core.Func effect row 분리 (미해결)

현재 `core.Func<Return, Params>`에 `effect?: Type` attribute가 있음
(`crates/trunk-ir/src/dialect/core.rs:25`). trunk-ir은 language-agnostic이어야
하므로, effect row를 tribute-specific 타입으로 분리하는 것이 바람직.

선택지:

- `tribute_rt.Func` 타입에 effect row 이동
- Effect row를 `func.func` op attribute로 이동
- 현재 상태 유지 (optional이므로 trunk-ir은 무시 가능)

## 열린 설계 질문

1. **ability.perform 세부 설계**: evidence 참조 방식, operand 구조
2. **ability.handle_dispatch 구조**: handler arm 표현, prompt tag 매칭
3. **CPS 함수 시그니처와 evidence 패스 호환**: continuation param 위치
4. **Pure/effectful 경계**: handle expression에서 CPS → direct style 전환
5. **core.Func effect row**: trunk-ir에서 분리할지 여부
