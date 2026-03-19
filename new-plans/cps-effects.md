# CPS 기반 Effect Handling 파이프라인

## 핵심 설계: ability.handle + ability.perform

### ability.perform

Effect operation 수행을 위한 high-level op. CPS 변환에서 explicit
continuation과 함께 사용:

```text
ability.perform @State, @get, [%args], %continuation_closure
```

- Lowering: evidence lookup → ShiftInfo/Continuation 구성 → `YieldResult::Shift` 반환
- Block terminator 성격 (CPS에서 이후 코드는 없음 — continuation closure에 포함)

### ability.handle_dispatch

CPS에서 done handler는 body의 최외곽 continuation closure가 됨.
따라서 handle expression에서 done region이 분리되어, **handler dispatch만 담당**:

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

## 파이프라인

```text
ast_to_ir (CPS for effectful bodies, closure.lambda 생성)
→ evidence_params → lower_closure_lambda → closure_lower → evidence_calls
→ tail_resumptive → ability lowering (ability.perform → YieldResult)
→ lambda_flattening → dce → resolve_casts → emit
```

### ast_to_ir: `lower_block_cps()`

- 각 statement를 순회, effectful call 감지
- Effect point 발견 시: 나머지 stmts + value를 body region으로 만들어
  `closure.lambda` 생성 (lambda lifting은 별도 pass가 처리)
- 자연스러운 재귀: 분기/case arm 내 effect도 동일하게 처리

### handle expression

- done handler → `closure.lambda`로 생성 (CPS body의 outermost continuation)
- body → CPS 모드로 lowering (done_k를 continuation으로 전달)
- handler arms → ability.handle_dispatch region으로 생성

### 소스 lambda 처리

CPS continuation뿐 아니라 소스 레벨 lambda도 `closure.lambda`로 처리:

```tribute
let f = fn(x) { x + captured_var }
```

ast_to_ir → `closure.lambda [%captured_var] { ^bb0(%x): ... }`
→ lower_closure_lambda pass → `closure.new @lambda_0, [%captured_var]`

## closure.lambda

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

### lower_closure_lambda pass

**위치**: `crates/tribute-passes/src/lower_closure_lambda.rs`

역할:

1. 모듈 내 모든 `closure.lambda` op 수집
2. 각 lambda의 body region을 top-level `func.func`으로 추출
3. Captures를 env struct로 패킹
4. `closure.lambda` → `closure.new` 치환

## 조건 분기 안의 effect

```tribute
fn foo() ->{State} Int {
  let x = State::get()
  if x > 0 {
    State::set(x - 1)      // ← 분기 안의 effect
    x
  } else { 0 }
}
```

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

- YieldResult { Done(anyref), Shift(ShiftInfo) }
- ShiftInfo { value, prompt, op_idx, continuation }
- Continuation은 CPS closure로 표현

### tail_resumptive 상호작용

handler 쪽 (ability.suspend → ability.yield)에서 동작 — CPS 변환과 직교.

- ability.yield arm: handler 값 함수 직접 호출 → continuation 즉시 호출
- lambda_flattening이 즉시 호출 패턴 인라인

## 장기 방향

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

1. **CPS 함수 시그니처와 evidence 패스 호환**: continuation param 위치
2. **Pure/effectful 경계**: handle expression에서 CPS → direct style 전환
3. **core.Func effect row**: trunk-ir에서 분리할지 여부
