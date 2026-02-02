# Tribute Generics

> 이 문서는 Tribute의 제네릭 타입 및 함수 처리 전략을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목        | 선택                                        | 대안 (채택하지 않음)             |
| ----------- | ------------------------------------------- | -------------------------------- |
| 기본 전략   | Monomorphization                            | Type erasure, Dictionary passing |
| 다형적 재귀 | Uniform representation (anyref)             | 에러로 거부                      |
| 제네릭 타입 | 완전 monomorphization                       | Uniform representation           |
| 효과 다형성 | Evidence passing + call-site specialization | 전면 monomorphization            |

---

## Syntax

### 제네릭 타입 정의

```rust
// 타입 파라미터는 소괄호 안에 소문자로
struct Box(a) {
    value: a
}

struct Pair(a, b) {
    first: a
    second: b
}

enum Option(a) {
    None
    Some(a)
}

enum Result(a, e) {
    Ok { value: a }
    Error { error: e }
}
```

### 제네릭 타입 사용

```rust
let box: Box(Int) = Box { value: 42 }
let pair: Pair(Int, Text) = Pair { first: 1, second: "hello" }
let result: Result(Int, Text) = Ok { value: 42 }
```

### 제네릭 함수

```rust
// 타입 파라미터는 함수 이름 뒤 소괄호에
fn identity(a)(x: a) -> a {
    x
}

fn map(a, b)(xs: List(a), f: fn(a) -> b) -> List(b) {
    // ...
}

// 효과 다형적 함수
fn map_effect(a, b)(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b) {
    // ...
}
```

---

## Hybrid Monomorphization

### 전략

```text
제네릭 함수/타입
       │
       ▼
  다형적 재귀 감지?
       │
   ┌───┴───┐
   │ No    │ Yes
   ▼       ▼
Monomorph  Uniform Rep
(특수화)   (anyref/boxing)
```

### 처리 방식

| 대상                   | 조건            | 처리 방식              |
| ---------------------- | --------------- | ---------------------- |
| 제네릭 타입            | 일반            | Monomorphization       |
| 제네릭 함수            | 일반            | Monomorphization       |
| 제네릭 함수            | **다형적 재귀** | Uniform representation |
| Effect만 다형적인 함수 | -               | Evidence passing       |

---

## Monomorphization

### 파이프라인 위치

```text
Stage 4: Type Inference
    ↓
Stage 5: TDNR
    ↓
Stage 5.5: Monomorphization ← 새로운 단계
    ↓
Stage 6: Lower Case
    ↓
Stage 7: Codegen (Wasm/Cranelift)
```

### 이름 맹글링

```text
Box(Int)           → Box$Int
List(Option(Int))  → List$Option_Int_
Pair(Int, Text)    → Pair$Int$Text
identity<Int>      → identity$Int
map<Int, Text>     → map$Int$Text
```

### 알고리즘

1. **수집 (Collection)**
   - 타입 추론 완료 후 모듈 순회
   - 모든 제네릭 인스턴스화 수집
   - 재귀 타입은 placeholder로 사이클 처리

2. **생성 (Generation)**
   - 각 인스턴스화에 대해 특수화된 타입/함수 정의 생성
   - 타입 파라미터를 구체 타입으로 치환

3. **재작성 (Rewriting)**
   - 호출 사이트를 특수화된 버전으로 변환
   - 타입 참조를 맹글된 이름으로 교체

### 예시

```rust
// 원본
fn identity(a)(x: a) -> a { x }

fn main() {
    identity(42)       // identity<Int>
    identity("hello")  // identity<Text>
}

// Monomorphization 후
fn identity$Int(x: Int) -> Int { x }
fn identity$Text(x: Text) -> Text { x }

fn main() {
    identity$Int(42)
    identity$Text("hello")
}
```

---

## Polymorphic Recursion

### 정의

함수 `f<a>` 가 **다형적 재귀**인 경우:

- `f`가 자기 자신을 호출하면서
- 타입 인자가 원래 `a`와 다름

```rust
// 다형적 재귀 예시
fn nest(a)(n: Int, x: a) -> ??? {
    if n == 0 { x }
    else { nest(n - 1, Pair(x, x)) }
    //         ↑ nest<Pair<a, a>> 호출 (a가 아님!)
}
```

### 문제점

순수 monomorphization으로는 무한 인스턴스화 발생:

- `nest<Int>` → `nest<Pair<Int, Int>>`
  → `nest<Pair<Pair<Int, Int>, Pair<Int, Int>>>` → ...

### 해결책: Uniform Representation

다형적 재귀 함수는 자동으로 **uniform representation** 사용:

```rust
// 컴파일러가 자동으로 변환
fn nest(n: Int, x: anyref) -> anyref {
    if n == 0 { x }
    else { nest(n - 1, box(Pair(unbox(x), unbox(x)))) }
}
```

### 감지 알고리즘

타입 추론 중 재귀 호출 패턴 분석:

1. 함수 `f<a>`의 본문에서 `f` 호출 찾기
2. 호출 시 타입 인자가 `a`를 포함하지만 `a`와 다른지 확인
3. 다르면 다형적 재귀로 표시

---

## WasmGC Integration

### Monomorphized Types

```wasm
;; Box$Int
(type $Box$Int (struct (field $value i64)))

;; Box$Text
(type $Box$Text (struct (field $value (ref $text))))

;; Pair$Int$Text
(type $Pair$Int$Text (struct
  (field $first i64)
  (field $second (ref $text))))
```

### Uniform Representation

다형적 재귀 함수에서는 `anyref` 사용:

```wasm
;; 타입 계층
any (anyref) ← 다형적 값의 공통 타입
 ├─ i31      ← 31비트 정수 (힙 할당 없음!)
 └─ struct   ← 박스/사용자 정의 타입
```

**Boxing 전략:**

| 원본 타입    | WasmGC 표현        | 힙 할당   |
| ------------ | ------------------ | --------- |
| Int (fixnum) | `i31ref`           | 없음      |
| Int (bignum) | `(ref $BigInt)`    | 자동 승격 |
| Float        | `(ref $BoxedF64)`  | 필요      |
| struct/enum  | 기존 참조 업캐스트 | 없음      |

---

## Implementation

### 주요 파일

- `src/pipeline.rs` - 새 단계 추가
- `crates/tribute-passes/src/monomorphize/` - 새 모듈
  - `mod.rs` - 모듈 정의
  - `collect.rs` - 인스턴스화 수집
  - `types.rs` - InstantiationKey, MonomorphizationPlan
  - `transform.rs` - 변환 로직
- `crates/tribute-passes/src/typeck/solver.rs` - TypeSubst 재사용

### 데이터 구조

```rust
/// 인스턴스화 키
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstantiationKey<'db> {
    /// 원본 제네릭 정의 이름
    pub name: Symbol,
    /// 구체적 타입 인자
    pub type_args: IdVec<Type<'db>>,
}

/// Monomorphization 계획
#[salsa::tracked]
pub struct MonomorphizationPlan<'db> {
    /// 타입 인스턴스화: 제네릭 타입 → 인스턴스화 키 목록
    type_instantiations: HashMap<Symbol, Vec<InstantiationKey<'db>>>,
    /// 함수 인스턴스화: 제네릭 함수 → 인스턴스화 키 목록
    func_instantiations: HashMap<Symbol, Vec<InstantiationKey<'db>>>,
    /// 인스턴스화 키 → 맹글된 이름
    instantiation_names: HashMap<InstantiationKey<'db>, Symbol>,
    /// 다형적 재귀 함수 집합
    polymorphic_recursive: HashSet<Symbol>,
}
```

---

## Edge Cases

| 케이스           | 해결책                             |
| ---------------- | ---------------------------------- |
| 다형적 재귀      | 자동 감지 → uniform representation |
| 미해결 타입 변수 | 명시적 타입 주석 요구              |
| 분리 컴파일      | 향후 링크 타임 중복 제거           |
| 재귀 타입        | Placeholder로 사이클 처리          |

---

## References

- [MLton Monomorphise](http://mlton.org/Monomorphise)
- [GHC Representation Polymorphism](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/representation_polymorphism.html)
- [OCaml Polymorphism](https://ocaml.org/manual/5.1/polymorphism.html)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/MVP.md)
