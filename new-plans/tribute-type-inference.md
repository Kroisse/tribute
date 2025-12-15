# Tribute Type Inference

> 이 문서는 Tribute의 타입 추론 시스템, 특히 row polymorphic effect typing과 bidirectional typing의 통합을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목 | 선택 | 대안 (채택하지 않음) |
|------|------|----------------------|
| 타입 추론 방식 | Bidirectional | 순수 HM, 전면 양방향 |
| Effect polymorphism | Row variables | Subtyping constraints |
| Effect 흐름 | Hybrid (inward + outward) | Frank (순수 inward), Koka (순수 outward) |
| 중복 label | 금지 | 허용 (런타임 모호성) |
| 암묵적 polymorphism | `fn(a) -> b` = `fn(a) ->{e} b` | 항상 명시 |

---

## Effect Row Syntax

### 기본 문법

```rust
// 구체적 ability만
fn foo() ->{State(Int), Console} Nil

// Row 변수 (나머지를 나타냄)
fn bar(f: fn() ->{e} a) ->{e} a

// Row 변수 + 구체적 ability
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a

// 여러 row 변수의 합집합
fn compose(f: fn(a) ->{e1} b, g: fn(b) ->{e2} c) -> fn(a) ->{e1, e2} c

// 순수 함수 (빈 row)
fn pure(x: a) ->{} a

// 암묵적 polymorphic (생략 시)
fn map(xs: List(a), f: fn(a) -> b) -> List(b)
```

### 암묵적 Effect Polymorphism

Effect annotation이 생략되면 암묵적으로 polymorphic:

```rust
// 이 두 선언은 동일
fn map(xs: List(a), f: fn(a) -> b) -> List(b)
fn map(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)
```

고차 함수에서 전달받은 함수의 effect가 그대로 전파된다.

### Row 구성 요소

Effect row는 다음으로 구성된다:

```
Row ::= {}                    -- 빈 row (순수)
      | {A₁, A₂, ..., Aₙ}     -- 구체적 ability들
      | {e}                   -- row 변수
      | {A₁, ..., Aₙ, e}      -- 구체적 + row 변수
      | {e₁, e₂}              -- row 변수 합집합
      | {A₁, ..., e₁, e₂}     -- 혼합
```

### 중복 Label 금지

같은 ability가 row에 두 번 나타나면 컴파일 에러:

```rust
// OK: 타입 파라미터가 다르면 다른 ability로 취급
fn foo() ->{State(Int), State(String)} Nil

// Error: 동일한 ability 중복
fn bar() ->{State(Int), State(Int)} Nil
```

**이유**: 중복 허용 시 `State::get()`이 어떤 handler를 참조하는지 타입 수준에서 결정할 수 없다. "가장 안쪽 handler"는 런타임 개념이지 타입 시스템이 추적할 수 있는 정보가 아니다.

**향후 확장**: 동일 ability의 여러 인스턴스가 필요한 경우, effect row에서 이름을 붙일 수 있다:

```rust
// 향후 가능한 확장 (현재 미지원)
fn nested() ->{State(Int) as counter, State(Int) as total} Nil {
    let n = counter::get()
    let t = total::get()
    counter::set(n + 1)
    total::set(t + n)
}
```

이름이 네임스페이스 역할을 하여 `counter::get()`과 `total::get()`을 구분한다.

---

## Row Unification

### 기본 규칙

두 row를 unify할 때, 공통 label을 맞추고 나머지를 row 변수로 표현한다:

```
unify({A, B | e₁}, {A, C | e₂})

1. 공통 label A 확인
2. e₁ = {C | e₃} 로 인스턴스화
3. e₂ = {B | e₃} 로 인스턴스화
4. 결과: {A, B, C | e₃}
```

**중복 검사**: Unification 결과에 동일한 ability가 두 번 나타나면 에러:

```
unify({State(Int) | e₁}, {State(Int) | e₂})
// e₁ = e₂ 로 unify됨, 결과: {State(Int) | e₁}  -- OK (중복 아님)

unify({State(Int)}, {State(Int), State(Int)})
// Error: 동일한 ability State(Int) 중복
```

### 예시

```rust
fn example(f: fn() ->{State(Int)} a, g: fn() ->{Console} b) {
    f()  // effect: {State(Int)}
    g()  // effect: {Console}
}
// 추론된 타입: fn(...) ->{State(Int), Console} Nil
```

Unification 과정:
```
{State(Int) | e₁} ∪ {Console | e₂}
= {State(Int), Console | e₃}
  where e₁ = {Console | e₃}, e₂ = {State(Int) | e₃}
```

### Occurs Check

Row 변수에 대해서도 occurs check 필요:

```
unify(e, {State(Int) | e})  -- 에러: e가 자기 자신을 포함
```

---

## Handler와 Effect 소비

### Effect 제거

Handler는 특정 ability를 row에서 "소비"한다:

```rust
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a
```

타입 규칙:
```
Γ ⊢ comp : fn() ->{e, State(s)} a
Γ ⊢ init : s
────────────────────────────────────────
Γ ⊢ run_state(comp, init) ->{e} a
```

`comp`의 effect `{e, State(s)}`에서:
- `State(s)`는 handler가 처리
- `e`(나머지)만 외부로 전파

### Row에서 Ability 제거

```
remove(State(s), {State(s), Console | e}) = {Console | e}
remove(State(s), {Console | e}) = 에러: State(s)가 없음
remove(State(s), {e}) = e' where e = {State(s) | e'}
```

마지막 경우는 row 변수 `e`가 `State(s)`를 포함한다고 가정하고, 새 변수 `e'`를 도입한다.

---

## Bidirectional Typing

### Judgment 형태

```
Γ ⊢ e ⇒ A ; E    -- Infer: 표현식 e의 타입 A와 effect E를 추론
Γ ⊢ e ⇐ A ; E    -- Check: 표현식 e가 타입 A, effect E를 가지는지 검사
```

### 핵심 규칙

#### 변수 (Infer)
```
x : A ∈ Γ
─────────────────
Γ ⊢ x ⇒ A ; {}
```

#### 람다 (Check)
```
Γ, x : A ⊢ body ⇐ B ; E
───────────────────────────────
Γ ⊢ fn(x) body ⇐ fn(A) ->{E} B ; {}
```

#### 람다 (Infer)
```
fresh α, β, e
Γ, x : α ⊢ body ⇐ β ; e
─────────────────────────────────
Γ ⊢ fn(x) body ⇒ fn(α) ->{e} β ; {}
```

#### 함수 적용 (Infer)
```
Γ ⊢ f ⇒ fn(A) ->{E} B ; E₁
Γ ⊢ x ⇐ A ; E₂
──────────────────────────────
Γ ⊢ f(x) ⇒ B ; E ∪ E₁ ∪ E₂
```

#### Ability Operation (Infer)
```
op : fn(A₁, ..., Aₙ) ->{Eff} B ∈ Ability
Γ ⊢ eᵢ ⇐ Aᵢ ; Eᵢ
─────────────────────────────────────
Γ ⊢ Eff::op(e₁, ..., eₙ) ⇒ B ; {Eff} ∪ E₁ ∪ ... ∪ Eₙ
```

#### Handle (Infer)
```
Γ ⊢ comp ⇒ fn() ->{E, Eff} A ; E₁
Γ ⊢ clauses handle Eff with continuation type
────────────────────────────────────────────
Γ ⊢ handle comp() { clauses } ⇒ B ; E ∪ E₁
```

### Subsumption

Effect row 간의 subsumption:

```
E₁ ⊆ E₂
Γ ⊢ e ⇐ A ; E₁
───────────────────
Γ ⊢ e ⇐ A ; E₂
```

`{State(Int)} ⊆ {State(Int), Console}` 이므로, 더 적은 effect를 가진 표현식은 더 많은 effect가 허용되는 컨텍스트에서 사용 가능하다.

---

## Type Inference Algorithm

### 개요

1. **Parse** → Surface AST
2. **Rename** → 고유한 이름 부여
3. **Constraint Generation** → bidirectional traversal로 제약 수집
4. **Constraint Solving** → unification (타입 + row)
5. **Generalization** → let-polymorphism

### Constraint 종류

```
C ::= τ₁ = τ₂           -- 타입 동치
    | ρ₁ = ρ₂           -- row 동치
    | ρ₁ ⊆ ρ₂           -- row 포함 (subsumption)
    | A ∈ ρ             -- ability 멤버십
    | C₁ ∧ C₂           -- conjunction
```

### Row Unification Algorithm

```
unify_row(ρ₁, ρ₂):
  match (ρ₁, ρ₂):
    ({}, {}) → success
    
    ({A | ρ₁'}, {A | ρ₂'}) → 
      unify_row(ρ₁', ρ₂')
    
    ({A | ρ₁'}, {B | ρ₂'}) where A ≠ B →
      fresh e
      unify_row(ρ₁', {B | e})
      unify_row(ρ₂', {A | e})
    
    (e, ρ) where e is variable →
      if e ∈ FV(ρ) then error "occurs check"
      else substitute e := ρ
    
    (ρ, e) where e is variable →
      unify_row(e, ρ)
```

### Generalization

Let binding에서 effect가 비어있을 때만 generalize:

```rust
// OK: f의 effect가 {}이므로 generalize 가능
let f = fn(x) x + 1
// f : ∀a. fn(a) ->{} a  (a가 Num일 때)

// 주의: effect가 있으면 value restriction 적용
let g = fn(x) { State::get(); x }
// g : fn(a) ->{State(s)} a  (generalize 안 함)
```

---

## 예시

### 기본 함수

```rust
fn add(x: Int, y: Int) -> Int {
    x + y
}
// 추론: fn(Int, Int) ->{} Int
```

### Effect 전파

```rust
fn fetch_and_print(url: String) ->{Http, Console} Nil {
    let response = Http::get(url)
    Console::println(response.body)
}
// Http::get : fn(String) ->{Http} Response
// Console::println : fn(String) ->{Console} Nil
// 합집합: {Http, Console}
```

### Handler

```rust
fn with_state(comp: fn() ->{e, State(Int)} a) ->{e} a {
    run_state(comp, 0)
}

fn example() ->{Console} Int {
    with_state(fn() {
        let n = State::get()
        State::set(n + 1)
        Console::println("incremented")
        State::get()
    })
}
// comp의 effect: {Console, State(Int)}
// State(Int) 소비 후: {Console}
```

### 고차 함수

```rust
fn twice(f: fn(a) ->{e} a, x: a) ->{e} a {
    f(f(x))
}

fn use_twice() ->{State(Int)} Int {
    twice(fn(n) { State::set(n); n + 1 }, 0)
}
// f의 effect {State(Int)}가 twice의 결과로 전파
```

### 합성

```rust
fn compose(f: fn(a) ->{e1} b, g: fn(b) ->{e2} c) -> fn(a) ->{e1, e2} c {
    fn(x) g(f(x))
}

let h = compose(
    fn(x) { Console::println(x); x },      // ->{Console}
    fn(x) { Http::get(x) }                  // ->{Http}
)
// h : fn(String) ->{Console, Http} Response
```

---

## Design Rules

### Effect Annotation 규칙

Effect annotation을 생략하면 fresh한 ability 변수가 생성된다:

```rust
// 단순 함수: 생략 가능
fn fetch(url: String) -> Response
// 위는 아래와 동일:
fn fetch(url: String) ->{e} Response

// 순수 함수는 명시적으로 {} 표기
fn add(x: Int, y: Int) ->{} Int { x + y }

// 특정 effect
fn fetch_data(url: String) ->{Http} Response { ... }
```

**고차 함수에서 effect 전파**: 내부 함수의 effect를 외부로 전파하려면 **같은 변수를 명시**해야 한다:

```rust
// 생략하면 다른 변수가 됨 (전파 안 됨)
fn map(xs: List(a), f: fn(a) -> b) -> List(b)
// 위는 아래와 동일:
fn map(xs: List(a), f: fn(a) ->{e1} b) ->{e2} List(b)
// e1 ≠ e2 → f의 effect가 map으로 전파되지 않음!

// 올바른 선언: 같은 변수 사용
fn map(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)

// 여러 함수의 effect 합치기
fn compose(f: fn(a) ->{e1} b, g: fn(b) ->{e2} c) -> fn(a) ->{e1, e2} c
```

**참고**: 순수 함수(`->{} T`)와 polymorphic 함수(`-> T`, 즉 `->{e} T`)는 다르다:
- `->{} Int`: 어떤 effect도 수행하지 않음
- `-> Int`: 암묵적 effect 변수, 컨텍스트의 effect 수행 가능

---

## Open Questions

1. **에러 메시지**: Row unification 실패 시 사용자 친화적인 메시지 생성

2. **IDE 지원**: 추론된 effect를 어떻게 표시할지

3. **Effect aliases** (향후 고려):
   ```rust
   type IO = {Console, FileSystem, Http}
   fn main() ->{IO} Nil
   ```

---

## References

- [Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism](https://www.cl.cam.ac.uk/~nk480/bidir.pdf)
- [Koka: Programming with Row-polymorphic Effect Types](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/koka-effects-2013.pdf)
- [Do Be Do Be Do](https://arxiv.org/abs/1611.09259) (Frank)
- [Extensible Records with Scoped Labels](https://www.microsoft.com/en-us/research/publication/extensible-records-with-scoped-labels/)
