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

Monomorphization은 AST 레벨에서 동작하며, `tribute-front` 크레이트 안에서 처리한다.
제네릭 함수를 먼저 구현하고, 제네릭 타입(struct/enum)은 후속 작업으로 분리한다.

### 구현 순서

1. **Phase A: 제네릭 함수** — 함수 특수화 + call site 재작성
2. **Phase B: 제네릭 타입** — struct/enum 특수화 + 생성자 재작성

### 파이프라인 삽입 위치

TDNR 이후, ast_to_ir 이전에 삽입한다. TDNR까지 마치면 모든 타입이 구체적으로
확정되어 있으므로, `node_types`에서 call site의 concrete type을 추출할 수 있다.

```text
TDNR (Module<TypedRef>)
    ↓
★ monomorphize — AST 레벨, tribute-front 크레이트
    ↓
ast_to_ir (IR lowering)
```

수정 대상: `src/pipeline.rs`의 `parse_and_lower_ast()` 함수.

### 기존 인프라 재사용

| 용도 | 기존 코드 | 위치 |
| ---- | --------- | ---- |
| BoundVar 치환 | `substitute_bound_vars()` | `typeck/subst.rs` |
| 함수 시그니처 조회 | `function_types: HashMap<Symbol, TypeScheme>` | `TypeCheckOutput` |
| 표현식 concrete type | `node_types: HashMap<NodeId, Type>` | `TypeCheckOutput` |
| 제네릭 여부 판별 | `TypeScheme.type_params` 비어있지 않으면 제네릭 | `ast/types.rs` |

### 모듈 구조

```text
crates/tribute-front/src/monomorphize/
├── mod.rs        — 공개 API
├── collect.rs    — 인스턴스화 수집
├── mangle.rs     — 이름 맹글링
└── specialize.rs — 특수화된 정의 생성 + call site 재작성
```

### Phase A: 제네릭 함수 Monomorphization

#### Step 1: 이름 맹글링 (`mangle.rs`)

Type → mangled name 변환. `$0`/`$1`로 중첩 타입 인자를 감싼다:

```text
identity + [Int]           → identity$Int
first + [Int, Text]        → first$Int$Text
map + [Int, Option(Int)]   → map$Int$Option$0$Int$1
```

#### Step 2: 인스턴스화 수집 (`collect.rs`)

`Module<TypedRef>` 전체를 순회하며 제네릭 함수 호출을 찾는다.

**Type arg 추론 방법:**

TypeScheme의 param types와 call site의 concrete argument types를 매칭하여
BoundVar → concrete type 매핑을 역추론한다.

```text
TypeScheme: fn identity(a)(x: a) -> a
  body = Func { params: [BoundVar(0)], result: BoundVar(0) }

Call site: identity(42)
  arg types = [Int]  (node_types에서 조회)

매칭: BoundVar(0) = Int
  → type_args = [Int]
```

여러 파라미터가 같은 BoundVar를 참조하면 일관성 검증한다.
수집 결과는 `HashMap<Symbol, HashSet<Vec<Type>>>` (함수명 → type arg 조합 집합).

#### Step 3: 특수화된 함수 생성 (`specialize.rs`)

각 (function_name, type_args)에 대해:

1. 원본 `FuncDecl<TypedRef>` 복제
2. `type_params` 비움
3. `substitute_bound_vars()`로 body의 모든 `TypedRef.ty`에서 BoundVar 치환
4. mangled name 적용
5. `function_types`에 specialized TypeScheme 등록

#### Step 4: Call site 재작성

Module 순회하며 제네릭 함수 호출의 `ResolvedRef::Function { id }` →
specialized function의 id로 교체. 원본 제네릭 선언은 유지한다
(향후 DCE에서 제거 가능).

#### Step 5: Pipeline 통합

`parse_and_lower_ast()`에서 TDNR 후 monomorphize 호출.
`TypeCheckOutput`의 `function_types`, `node_types`를 monomorphize에 전달하고,
결과로 updated module + function_types를 받아 ast_to_ir에 넘긴다.

### Phase B: 제네릭 타입 (후속 작업)

Phase A 완료 후 별도 이슈로 추적:

- `StructDecl`, `EnumDecl`의 타입 파라미터 특수화
- 생성자 호출(`Variant`, `StructNew`) 재작성
- IR lowering에서 specialized struct/enum 타입 생성

### 데이터 구조

```rust
/// 인스턴스화 키 — 함수/타입 이름 + concrete type args
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct InstantiationKey<'db> {
    name: Symbol,
    type_args: Vec<Type<'db>>,
}
```

### 리스크와 대응

| 리스크 | 대응 |
| ------ | ---- |
| Polymorphic recursion → 무한 인스턴스화 | depth limit으로 방어, #54에서 본격 처리 |
| Higher-order functions의 type arg 추적 | closure 타입에서도 BoundVar 매칭 |
| Effect-only polymorphism | skip — evidence passing이 이미 처리 |
| 미해결 타입 변수 (BoundVar 잔존) | anyref 폴백 유지 (기존 uniform rep) |

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
