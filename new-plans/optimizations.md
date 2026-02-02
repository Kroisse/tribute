# Tribute Optimizations

> 이 문서는 Tribute 컴파일러의 최적화 전략을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목                            | 선택       | 효과                      |
| ------------------------------- | ---------- | ------------------------- |
| 타입 monomorphization           | 하이브리드 | 런타임 오버헤드 제거      |
| Effect call-site specialization | 지원       | Evidence 전달 제거/인라인 |
| Tail-resumptive optimization    | 지원       | shift/reset 제거          |
| Handler inlining                | 지원       | 간접 호출 제거            |

---

## Optimization Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│                      최적화 파이프라인                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Type Monomorphization                                   │
│     - 제네릭 타입/함수 특수화                                  │
│     - 다형적 재귀만 uniform rep                              │
│                                                             │
│  2. Call-Site Effect Specialization                         │
│     - 순수 call site: evidence 제거                          │
│     - 구체적 handler: evidence 인라인                         │
│     - row variable: evidence 유지                           │
│                                                             │
│  3. Tail-Resumptive Optimization                            │
│     - 즉시 resume하는 handler 감지                            │
│     - shift/reset 제거                                      │
│                                                             │
│  4. Handler Inlining                                        │
│     - 정적으로 알려진 handler 인라인                           │
│     - 간접 호출 → 직접 호출                                   │
│                                                             │
│  5. Standard Optimizations                                  │
│     - Dead code elimination                                 │
│     - Constant folding                                      │
│     - Inlining                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Type Monomorphization

### 기본 원리

제네릭 타입과 함수를 구체적 타입으로 특수화:

```rust
// 원본
fn identity(a)(x: a) -> a { x }

identity(42)       // identity<Int>
identity("hello")  // identity<Text>

// 최적화 후
fn identity$Int(x: Int) -> Int { x }
fn identity$Text(x: Text) -> Text { x }

identity$Int(42)
identity$Text("hello")
```

### 장점

- 런타임 타입 디스패치 제거
- 특수화된 기계어 생성 가능
- 인라이닝 기회 증가

### 다형적 재귀 처리

순수 monomorphization이 불가능한 경우 uniform representation 사용:

```rust
fn nest(n: Int, x: a) -> ??? {
    if n == 0 { x }
    else { nest(n - 1, Pair(x, x)) }
}

// → uniform representation (anyref)
fn nest(n: Int, x: anyref) -> anyref { ... }
```

자세한 내용은 `generics.md` 참조.

---

## Call-Site Effect Specialization

### 기본 원리

효과 다형적 함수도 call site에서 효과가 구체적으로 알려지면 특수화 가능:

```rust
// 정의: 효과 다형적
fn map(a, b)(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)

// Call site 1: e = {} (순수)
map(list, fn(x) x * 2)
// → Evidence 파라미터 완전 제거!

// Call site 2: e = {State(Int)}
run_state(fn() map(list, fn(x) { State::put(x); x }), 0)
// → State handler 연산 인라인!
```

### 특수화 조건

| Call Site 컨텍스트   | `{e}` 값     | 최적화                |
| -------------------- | ------------ | --------------------- |
| 순수 컨텍스트        | `{}`         | Evidence 완전 제거    |
| `handle` 블록 내부   | 구체적       | Handler 인라인        |
| 효과 다형적 컨텍스트 | row variable | Evidence passing 유지 |

### 예시: 순수 call site

```rust
// 원본
fn map(a, b)(xs: List(a), f: fn(a) ->{e} b, ev: *Evidence) ->{e} List(b)

// Call site에서 e = {}
map(list, fn(x) x * 2)

// 특수화 후: evidence 파라미터 제거
fn map$Int$Int$pure(xs: List$Int, f: fn(Int) -> Int) -> List$Int {
    // 순수 버전, evidence 없음
}
```

### 예시: 구체적 handler 내부

```rust
run_state(fn() {
    map(list, fn(x) { State::put(x); x })
}, 0)

// 특수화 후: State 연산 인라인
fn map$Int$Int$State_Int(
    xs: List$Int,
    f: fn(Int) -> Int,  // State::put이 인라인됨
    state_ptr: *Int      // State를 직접 포인터로 전달
) -> List$Int {
    // ...
}
```

---

## Tail-Resumptive Optimization

### 정의

Handler가 **항상 즉시 `k(value)`로 끝나면** (tail-resumptive), continuation 캡처가 불필요:

```rust
// Tail-resumptive handler 예시
{ State::get() -> k } -> k(current_state)
{ State::put(v) -> k } -> k(())
{ Console::print(s) -> k } -> { do_print(s); k(()) }
```

### 최적화 전

```rust
fn state_get(ev: *Evidence) -> s {
    let marker = (*ev).get(STATE_ID)
    shift(marker.prompt(), |k| {
        let op_table = &OP_TABLES[marker.op_table_index]
        (op_table.get)(k)  // k를 캡처하고 handler 호출
    })
}
```

### 최적화 후

```rust
fn state_get_optimized(ev: *Evidence) -> s {
    let marker = (*ev).get(STATE_ID)
    let op_table = &OP_TABLES[marker.op_table_index]
    (op_table.get_value)()  // 직접 값 반환, shift 없음!
}
```

### 감지 알고리즘

Handler의 각 case arm 분석:

1. `-> k` 패턴인지 확인
2. 본문이 `k(expr)`로만 끝나는지 확인
3. `k`가 다른 곳에서 사용되지 않는지 확인

```rust
// Tail-resumptive
{ A::op() -> k } -> k(value)           // OK
{ A::op() -> k } -> { stmt; k(value) } // OK

// Non-tail-resumptive
{ A::op() -> k } -> { k(v1); k(v2) }   // k 두 번 사용
{ A::op() -> k } -> { save(k); v }     // k 저장
{ A::op() -> k } -> other_func(k)      // k 전달
```

### 대부분의 실용적 Ability는 Tail-Resumptive

| Ability   | Tail-Resumptive? | 이유                     |
| --------- | ---------------- | ------------------------ |
| State     | Yes              | get/put 모두 즉시 resume |
| Reader    | Yes              | ask는 즉시 resume        |
| Writer    | Yes              | tell은 즉시 resume       |
| Console   | Yes              | print는 즉시 resume      |
| Exception | No               | fail은 resume 안 함      |
| Async     | 조건부           | await 후 resume          |
| Choice    | No               | 여러 번 resume 가능      |

---

## Handler Inlining

### 기본 원리

정적으로 알려진 handler를 인라인:

```rust
// 원본
fn program() ->{State(Int)} Int {
    State::get()
}

handle program() {
    { State::get() -> k } -> k(42)
}

// 최적화 후: handler 인라인
fn program_inlined() -> Int {
    42
}
```

### 조건

Handler inlining이 가능한 경우:

1. `handle` 블록이 컴파일 타임에 알려짐
2. 피호출 함수가 인라인 가능
3. Tail-resumptive optimization이 적용됨

### 복합 예시

```rust
// 원본
fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::put(n + 1)
    n
}

fn main() {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}

// 모든 최적화 적용 후
fn main() {
    let mut state = 0
    let r1 = state; state = state + 1
    let r2 = state; state = state + 1
    let r3 = state; state = state + 1
    r3
}
```

---

## Combined Optimization Example

### 원본 코드

```rust
fn sum_with_state(xs: List(Int)) ->{State(Int)} Int {
    xs.fold(0, fn(acc, x) {
        State::put(State::get() + 1)  // 카운트 증가
        acc + x
    })
}

fn main() {
    run_state(fn() sum_with_state([1, 2, 3, 4, 5]), 0)
}
```

### 최적화 단계

**1. Type Monomorphization:**

```rust
fn sum_with_state(xs: List$Int) ->{State(Int)} Int {
    xs.fold$Int$Int(0, fn(acc, x) {
        State::put(State::get() + 1)
        acc + x
    })
}
```

**2. Call-Site Effect Specialization:**

```rust
// run_state 내부이므로 e = {State(Int)}
fn sum_with_state$State_Int(xs: List$Int, state_ptr: *Int) -> Int {
    xs.fold$Int$Int$pure(0, fn(acc, x) {
        *state_ptr = *state_ptr + 1
        acc + x
    })
}
```

**3. Tail-Resumptive Optimization:**

- State::get/put은 이미 직접 포인터 접근으로 변환됨

**4. Handler Inlining:**

```rust
fn main() {
    let mut state = 0
    let result = sum_with_state_inlined(&mut state, [1, 2, 3, 4, 5])
    result  // state = 5 (5번 증가)
}
```

### 결과

- Evidence 조회: **제거됨**
- Continuation 캡처: **제거됨**
- 간접 호출: **제거됨**
- 런타임 오버헤드: **거의 0**

---

## Implementation

### 구현 순서 (개발 순서)

설계상 Monomorphization이 기본 전략이지만, **구현은 반대 순서**로 진행:

```text
Phase 1: Uniform Representation (기반)
├─ 모든 제네릭이 동작 (느리지만 정확)
├─ Int: i31ref + BigInt (이미 anyref 서브타입)
├─ Float: BoxedF64
├─ 제네릭 타입/함수: anyref 파라미터
└─ 의미론 테스트 가능

Phase 2: Monomorphization (최적화 레이어)
├─ Uniform rep 위에 추가
├─ 인스턴스화 수집 및 특수화
├─ 다형적 재귀 감지 → Phase 1로 폴백
└─ 정확성은 Phase 1과 비교 검증
```

**이유:**

| 관점         | Uniform Rep 먼저       | Monomorph 먼저     |
| ------------ | ---------------------- | ------------------ |
| 완전성       | 모든 케이스 커버       | 다형적 재귀 불가   |
| 테스트       | 의미론 검증 가능       | 부분적으로만       |
| 점진적 개발  | 동작 → 최적화          | 최적화부터 시작    |
| 디버깅       | 기준점 있음            | 기준점 없음        |

### 최적화 패스 순서

```rust
// src/pipeline.rs

pub fn compile_optimized(db: &dyn Database, source: SourceCst) -> Module {
    let module = stage_tdnr(db, source);

    // 1. Monomorphization
    let module = stage_monomorphize(db, module);

    // 2. Effect specialization
    let module = stage_effect_specialize(db, module);

    // 3. Tail-resumptive analysis & optimization
    let module = stage_tail_resumptive(db, module);

    // 4. Handler inlining
    let module = stage_handler_inline(db, module);

    // 5. Standard optimizations
    let module = stage_dce(db, module);
    let module = stage_const_fold(db, module);

    module
}
```

### 주요 파일

- `crates/tribute-passes/src/monomorphize/` - 타입 monomorphization
- `crates/tribute-passes/src/effect_specialize/` - 효과 특수화
- `crates/tribute-passes/src/tail_resumptive/` - Tail-resumptive 분석
- `crates/tribute-passes/src/inline/` - 인라이닝

---

## Benchmarks (예상)

### Ability 오버헤드

| 시나리오               | 최적화 전 | 최적화 후 |
| ---------------------- | --------- | --------- |
| State::get (100만 회)  | ~50ms     | ~5ms      |
| Reader::ask (100만 회) | ~40ms     | ~3ms      |
| 순수 함수 호출         | 0         | 0         |

### 제네릭 오버헤드

| 시나리오              | Type Erasure | Monomorphization |
| --------------------- | ------------ | ---------------- |
| List.map (100만 요소) | ~100ms       | ~20ms            |
| 정수 산술 (fixnum)    | ~10ms        | ~10ms            |
| 정수 산술 (bignum)    | ~50ms        | ~50ms            |

---

## Future Optimizations

### Escape Analysis

Continuation이 탈출하지 않으면 스택 할당:

```rust
handle {
    // k가 이 스코프 내에서만 사용됨
    // → 힙 대신 스택에 continuation 저장
}
```

### Partial Evaluation

컴파일 타임 값이 알려지면 미리 계산:

```rust
fn factorial(n: Int) -> Int {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

factorial(5)  // → 120으로 상수 폴딩
```

### Profile-Guided Optimization

런타임 프로파일 기반 최적화:

- Hot path 인라이닝
- Bignum 사용 패턴에 따른 특수화
- Effect handler 호출 패턴 분석

---

## References

- [Generalized Evidence Passing (Koka)](https://www.microsoft.com/en-us/research/publication/generalized-evidence-passing-for-effect-handlers-or-efficient-compilation-of-effect-handlers-to-c/)
- [GHC Specialisation](https://wiki.haskell.org/Inlining_and_Specialisation)
- [MLton Whole-Program Optimization](http://mlton.org/WholeProgramOptimization)
