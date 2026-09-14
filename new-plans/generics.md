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

`$`만 구조적 문자로 사용한다. `$0`/`$1`은 중첩 타입 인자의 시작/끝을 나타내며,
Tribute 식별자는 숫자로 시작할 수 없으므로 타입 이름과 충돌하지 않는다.

```text
identity + [Int]              → identity$Int
first + [Int, Text]           → first$Int$Text
map + [Int, Option(Int)]      → map$Int$Option$0$Int$1
f + [List(Option(Int))]       → f$List$0$Option$0$Int$1$1
apply + [fn(Int) -> Bool]     → apply$Fn$0$Int$1$Bool
swap + [(Int, Bool)]          → swap$Tup$0$Int$Bool$1
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

Monomorphization은 `tribute-front`의 typed AST와 명시적인 함수 인스턴스 기록을
소비한다. 함수 특수화와 nominal 타입 특수화는 같은 frontend 준비 단계에 속한다.

### 파이프라인의 책임

`parse_and_lower_ast()`는 이름 해석과 타입 추론·TDNR을 수행하고, 선택된 함수
인스턴스와 semantic 메타데이터를 반환한다. `prepare_frontend_for_lowering()`은
Prelude 병합, 도달하는 인스턴스 검증, 특수화 및 메타데이터 재작성을 담당한다.
IR lowering은 이 준비 단계가 성공한 결과만 소비한다.

```text
parse_and_lower_ast: 해석·타입 추론·TDNR·인스턴스 기록
    ↓
prepare_frontend_for_lowering: 병합·검증·특수화·메타데이터 재작성
    ↓
ast_to_ir: IR lowering
```

### 함수 인스턴스 수집과 특수화

수집과 호출 재작성은 함수 참조 NodeId에 연결된 checked instance의
`(FuncDefId, type_arguments)`를 사용한다. `node_types`나 callable 타입에서 타입
인자를 역추론하지 않는다. 함수 값으로 전달되는 참조도 같은 계약을 따른다.

선택된 선언 스킴, 인자 개수, callable 및 row 인자의 일관성을 먼저 검사한다.
원본 함수의 binder 순서에 맞춰 타입을 치환하고, 특수화된 정의의 모든 typed
메타데이터에도 같은 치환을 적용한다. 스킴 본문과 합집합·차집합 제약은 함께
변환한다. 생성된 clone 내부의 참조가 새 인스턴스를 드러내면 같은 인스턴스 키를
재사용하며 고정점까지 수집한다. 확장 한도를 넘으면 구조적 진단을 반환한다.

타입 인자를 갖는 함수만 이 과정으로 특수화한다. Row만 다형적인 함수는 기존
인스턴스 전달을 유지한다. 도달하지 않는 generic template은 허용하되, 도달하는
함수 인스턴스나 ability 인자가 미해결이면 타입 erasure 전에 거부한다.

### Nominal 타입 수집과 재작성

Nominal 타입의 수집 시작점은 AST 안의 typed reference와 재작성 대상 메타데이터다.
함수·생성자 스킴의 본문과 보존된 합집합·차집합 제약, 함수 인스턴스의 callable과
타입·row 인자, node 타입, lambda 시그니처, handler·perform의 인자·결과를 포함한다.
메타데이터에만 등장하는 concrete 타입도 특수화 선언을 생성해야 한다.

각 시작점 내부의 함수·continuation effect와 중첩 nominal 타입 인자를 재귀적으로
수집한다. 정확한 `TypeDefId`와 concrete 인자 조합으로 struct/enum 선언 및 생성자
스킴을 만들고, 같은 rewrite map을 AST와 메타데이터에 적용한다. 선언의 기준
스킴을 보관하는 provenance 기록은 인스턴스의 치환 결과와 구분한다.

### 이름과 identity

맹글링은 생성된 선언의 이름을 부여한다. 이름으로 선언 identity나 인스턴스 인자를
복원하지 않는다. 예를 들어 `identity`와 `[Int]`는 `identity$Int`로 표현하고, 중첩
타입 인자는 `$0`/`$1`로 감싸지만 특수화 키는 정확한 선언 ID와 타입 인자로 유지한다.

---

## References

- [MLton Monomorphise](http://mlton.org/Monomorphise)
- [GHC Representation Polymorphism](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/representation_polymorphism.html)
- [OCaml Polymorphism](https://ocaml.org/manual/5.1/polymorphism.html)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/MVP.md)

## Explicit checked function instances

Type checking records each function reference's declaration, source scheme,
ordered type arguments, quantified-row arguments, and instantiated callable
type. The record is keyed by the reference NodeId, shared across revisits of
that node, and independently instantiated at other nodes. Solving and binder
finalization apply the same mapping to the callable and all its arguments.

Collection and rewriting consume this record rather than reconstructing type
arguments by reverse matching the callable type. IDE type rendering accepts
function-local quantified variables as well as declaration binders; rendering
does not change their semantic scope or instantiate them.
Prelude name resolution, type
exports, and typed bodies share the same parsed declarations, including their
nominal definition identities. Independently reparsing the same text does not
establish declaration identity. Prelude merging, deferred
method desugaring, and specialized AST cloning preserve the record. Clones
substitute their enclosing type arguments into nested instance records.
Effect-only type arguments participate in ordinary type specialization; open
residual rows continue to use evidence passing. Unused generic templates are
permitted, but incomplete reached instances fail before logical lowering.

Function quantifiers cover the callable interface and retained semantic row
relations. Inference variables occurring only in the checked body remain
body-owned local existentials; they do not add phantom arguments to callers.
The pre-erasure ability-instance check still rejects unresolved local
existentials in reachable handler or perform ability arguments.
