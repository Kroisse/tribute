# Tribute Generics

> 이 문서는 Tribute의 제네릭 타입 및 함수 처리 전략을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목               | 선택                                        | 대안 (채택하지 않음)             |
| ------------------ | ------------------------------------------- | -------------------------------- |
| 기본 전략          | Monomorphization                            | Type erasure, Dictionary passing |
| 함수의 다형적 재귀 | Uniform representation (anyref)             | 에러로 거부                      |
| 제네릭 타입        | 완전 monomorphization                       | Uniform representation           |
| 효과 다형성        | Evidence passing + convention class 특수화  | Row 전체의 monomorphization      |

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
let pair: Pair(Int, String) = Pair { first: 1, second: "hello" }
let result: Result(Int, String) = Ok { value: 42 }
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
제네릭 함수
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

Effect row는 값으로 특수화하지 않는다. Row 변수가 요구하는 calling convention만
[convention class](#row-변수의-convention-class)로 특수화한다.

이 도식의 uniform representation 선택은 함수의 다형적 재귀에 적용한다.
Nominal 타입의 의존 인스턴스 수집과 확장 한도는 아래의
[Nominal 타입 수집과 재작성](#nominal-타입-수집과-재작성) 계약을 따른다.

---

## Monomorphization

### 파이프라인 위치

```text
Type inference + TDNR
    → checked function instances
    → frontend preparation + monomorphization
    → source-logical AST-to-IR lowering
    → shared CPS legalization
    → target lowering
```

Pattern matching도 frontend의 source-logical IR 생성에 포함한다. 별도 case lowering
단계를 두거나 monomorphization에서 physical CPS signature를 생성하지 않는다.

### 이름 맹글링

`$`만 구조적 문자로 사용한다. `$0`/`$1`은 중첩 타입 인자의 시작/끝을 나타내며,
Tribute 식별자는 숫자로 시작할 수 없으므로 타입 이름과 충돌하지 않는다.

소스 선언이 아니라 컴파일러가 구성하는 타입은 모두 숫자 태그로 시작한다. 함수
타입은 `$6`, 튜플은 `$7`, n번째 bound type variable은 `$8$n`, 내장 `List` 같은
compiler-owned nominal 타입은 `$5`로 시작한다. 따라서 문자로 시작하는 구간은
항상 원시 타입이나 소스가 선언한 이름이고, 소스 타입 이름이 `Fn`이나 `Tup`이어도
구조 타입과 같은 이름을 얻지 않는다.

```text
identity + [Int]              → identity$Int
first + [Int, String]           → first$Int$String
map + [Int, Option(Int)]      → map$Int$Option$0$Int$1
f + [List(Option(Int))]       → f$5$List$0$Option$0$Int$1$1
apply + [fn(Int) -> Bool]     → apply$6$0$Int$1$Bool
swap + [(Int, Bool)]          → swap$7$0$Int$Bool$1
```

[Convention class](#row-변수의-convention-class) 목록은 타입 인자 뒤에 `$9`와 class
변수 순서대로 한 글자씩(`D`, `E`, `C`) 쓴다. 모든 class가 `Cps`이면 쓰지 않는다.

```text
map + [Int, Bool] + [Direct]          → map$Int$Bool$9D
map + [Int, Bool] + [EvidenceDirect]  → map$Int$Bool$9E
map + [Int, Bool] + [Cps]             → map$Int$Bool
apply_twice + [] + [Direct]           → apply_twice$9D
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

fn main() -> Nil {
    identity(42)       // identity<Int>
    identity("hello")  // identity<String>
}

// Monomorphization 후
fn identity$Int(x: Int) -> Int { x }
fn identity$String(x: String) -> String { x }

fn main() -> Nil {
    identity$Int(42)
    identity$String("hello")
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
    case n == 0 {
        True -> x
        False -> nest(n - 1, Pair(x, x))
        //            ↑ nest<Pair<a, a>> 호출 (a가 아님!)
    }
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
    case n == 0 {
        True -> x
        False -> nest(n - 1, box(Pair(unbox(x), unbox(x))))
    }
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

;; Box$String
(type $Box$String (struct (field $value (ref $string))))

;; Pair$Int$String
(type $Pair$Int$String (struct
  (field $first i64)
  (field $second (ref $string))))
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

타입 인자나 [class 변수](#row-변수의-convention-class)를 갖는 함수를 이 과정으로
특수화한다. 둘 다 없는 함수는 정의 하나를 그대로 쓴다. 도달하지 않는 generic
template은 허용하되, 도달하는 함수 인스턴스나 ability 인자가 미해결이면 타입 erasure
전에 거부한다.

### Row 변수의 convention class

Effect row는 타입 인자처럼 치환하지 않는다. Row의 내용은 실행 중 evidence로
전달한다. 특수화하는 것은 row 변수가 요구하는 calling convention뿐이며, 이를 그
변수의 **convention class**라 한다. Class는 기존 순서
`Direct < EvidenceDirect < Cps`의 세 값 중 하나다.

**Class 변수.** 정의의 스킴이 양화한 row 변수 중 그 class가 정의의 코드를 바꾸는
것이 class 변수다. 다음 가운데 하나에 해당하는 변수이며, 순서는 스킴의 row binder
순서를 따른다.

- 매개변수 타입에 포함된 함수 타입의 row tail
- 본문의 람다나 지역 callable 타입의 row tail
- 본문이 참조하는 정의의 class 변수 자리에 넘기는 row의 tail

마지막 조건은 참조되는 정의의 class 변수에 의존하므로, 모든 정의에 대한 최소
고정점으로 정한다. 정의 자신의 row에만 나타나는 tail도 이 조건으로 class 변수가 될
수 있다.

정의의 signature나 본문에서 ability 인자 안에 나타나는 row 변수는 class 변수가
아니다. Ability instance identity가 그 row를 포함하므로, handler와 perform 지점이
같은 callable convention을 보아야 한다. 이런 변수의 class는 항상 `Cps`다.

Compiler가 생성하는 정의도 같은 규칙을 따른다. 필드의 `modify`는 callback의 row
변수를 class 변수로 가진다.

**Class 인자.** 참조 지점의 class 인자는 checked instance가 기록한 row 인자에서
계산한다. Callable 타입에서 역추론하지 않는다.

```text
class({A₁, ..., Aₙ})     = requirement(A₁) ⊔ ... ⊔ requirement(Aₙ)
class({A₁, ..., Aₙ | e}) = 위 값 ⊔ class(e)

class(e) = 참조를 포함한 인스턴스가 e에 고정한 class   (e가 그 인스턴스의 class 변수)
         = Cps                                          (그 밖의 row 변수)
```

`requirement`는 ability 단위의
[convention bound](implementation.md#selective-transformation)다.

**인스턴스.** 인스턴스 key는 `(선언, 타입 인자, class 인자)`다. Clone의 NodeId를
구별하는 값도 타입 인자와 class 인자를 함께 반영한다. Class 인자가 모두 `Cps`인
인스턴스는 class 특수화가 없는 정의와 같은 convention과 이름을 가진다.

Root `main`은 참조 없이 존재하는 인스턴스다. 그 row를 instantiate하는 호출자가
없으므로 tail은 비어 있고, class 변수는 모두 `Direct`로 고정한다.

**인스턴스 안에서의 규칙.** 인스턴스는 class 변수에서 class로 가는 표를 가진다.
Lowering이 row의 convention을 계산할 때 tail이 표에 있으면 그 class를 쓰고, 없으면
열린 row의 규칙대로 `Cps`를 쓴다. 매개변수의 callable 타입, 그 callable의 호출,
인스턴스 자신의 worker convention이 모두 이 표를 따른다. 따라서 호출자가 넘긴
callable의 convention과 인스턴스가 기대하는 convention이 class 인자에서 함께
정해진다.

Class 특수화는 row를 스킴이나 metadata에 치환하지 않는다. 호출의 evidence 선택은
typechecking이 원래 정의에 대해 계산한 것을 그대로 물려받으며 다시 계산하지 않는다.
Evidence를 받지 않는 convention의 호출에는 선택을 싣지 않는다.

Class 값은 셋뿐이므로 class 변수가 `n`개인 정의의 인스턴스는 타입 인자 조합마다
최대 `3ⁿ`개다. 재귀가 새 class 인자를 만들어도 이 한도 안에서 고정점에 도달한다.

### Nominal 타입 수집과 재작성

Nominal 타입의 수집 시작점은 AST 안의 typed reference와 재작성 대상 메타데이터다.
함수·생성자 스킴의 본문과 보존된 합집합·차집합 제약, 함수 인스턴스의 callable과
타입·row 인자, node 타입, lambda 시그니처, handler·perform의 인자·결과를 포함한다.
메타데이터에만 등장하는 concrete 타입도 특수화 선언을 생성해야 한다.

함수 특수화가 끝난 뒤 원본 nominal 선언과 canonical constructor identity를
한 번 수집하여 준비 단계 안에서 공유한다. 이 선언 인덱스는 seed 수집, 의존성
확장, struct·enum 생성이 함께 사용하는 임시 자료이며 별도 공개 registry가 아니다.
Generic 여부는 인덱스에 보관된 원본 선언의 타입 매개변수 유무로 판별한다.

각 시작점 내부의 callable·tuple·continuation effect와 중첩 nominal 타입 인자를
재귀적으로 수집한다. 발견한 struct·enum 인스턴스의 checked constructor 스킴을
검증·치환하고, 그 본문과 보존된 row 제약이 드러내는 의존 인스턴스를 고정점까지
수집한다. 중복 판별은 원본 `TypeDefId`와 concrete 타입 인자 목록을 사용한다.
생성 연산 없이 signature나 필드에서만 참조되는 인스턴스도 이 계약을 따른다.

의존성 수집에서 치환한 스킴은 최종 constructor 메타데이터로 재사용한다.
원본 스킴을 다시 읽어 동일 인스턴스를 재치환하지 않는다. 고정점 도달에 성공한
뒤 맹글링한 이름으로 선언과 스킴을 게시하고, 같은 nominal rewrite map을 AST와
메타데이터에 적용한다. 선언의 기준 스킴을 보관하는 provenance 기록은 인스턴스의
치환 결과와 구분한다. AST annotation 치환과 semantic 스킴 치환은 각 표현에
필요한 별도 작업이다.

Enum의 치환된 스킴은 복제된 variant의 `NodeId`에 연결한다. 원본 constructor
identity와 runtime variant tag는 유지하며, 다른 모듈이나 다른 타입 인자의 스킴을
덮어쓰지 않는다. Lowering은 이 연결을 필수 입력으로 소비하고, 누락된 스킴을
원본 generic 스킴으로 대체하지 않는다.

동일 인스턴스의 재귀·상호 재귀는 중복 생성하지 않는다. 계속 새로운 타입 인자를
만드는 확장은 최대 64회 의존성 확장과 4,096개 고유 nominal 인스턴스 한도에서
진단한다. 한도 초과나 잘못된 constructor 스킴으로 준비가 실패하면 부분 결과를
IR 생성에 전달하지 않는다. 이 한도는 함수의 다형적 재귀 처리 정책을 바꾸지 않는다.

### 이름과 identity

맹글링은 생성된 선언의 이름을 부여한다. 이름으로 선언 identity나 인스턴스 인자를
복원하지 않는다. 예를 들어 `identity`와 `[Int]`는 `identity$Int`로 표현하고, 중첩
타입 인자는 `$0`/`$1`로 감싸지만 특수화 키는 정확한 선언 ID와 타입 인자로 유지한다.

Import alias와 prelude의 짧은 조회 이름은 선언의 canonical path를 대체하지
않는다. `List::prepend`가 표준 binding을 선택했다면 특수화 대상은
`std::collections::List::prepend`이며, 완전한 경로로 선택한 동일 인스턴스와
특수화 정의를 공유한다. 사용자 `List::prepend`의 인스턴스는 별도로 유지한다.

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
residual rows continue to use evidence passing, and the recorded row arguments
select each instance's [convention classes](#row-변수의-convention-class).
Unused generic templates are permitted, but incomplete reached instances fail
before logical lowering.

Function quantifiers cover the callable interface and retained semantic row
relations. Inference variables occurring only in the checked body remain
body-owned local existentials; they do not add phantom arguments to callers.
The pre-erasure ability-instance check still rejects unresolved local
existentials in reachable handler or perform ability arguments.
