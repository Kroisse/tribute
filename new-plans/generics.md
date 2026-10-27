# Tribute Generics

> 이 문서는 Tribute의 제네릭 타입 및 함수 처리 전략을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목               | 선택                                        | 대안 (채택하지 않음)             |
| ------------------ | ------------------------------------------- | -------------------------------- |
| 기본 전략          | Monomorphization                            | Type erasure, Dictionary passing |
| 함수의 다형적 재귀 | 에러로 거부                                 | Uniform representation (anyref)  |
| 제네릭 타입        | 완전 monomorphization                       | Uniform representation           |
| 효과 다형성        | Evidence passing                            | Row 전체의 monomorphization      |

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
let result: Result(Int, String) = .Ok { value: 42 }
```

### 제네릭 함수

```rust
// 시그니처에 쓴 소문자 타입 이름이 함수의 타입 파라미터다
fn identity(x: a) -> a {
    x
}

fn map(xs: List(a), f: fn(a) -> b) -> List(b) {
    // ...
}

// 효과 다형적 함수
fn map_effect(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b) {
    // ...
}
```

---

## 특수화 전략

### 처리 방식

| 대상                   | 조건            | 처리 방식        |
| ---------------------- | --------------- | ---------------- |
| 제네릭 타입            | 일반            | Monomorphization |
| 제네릭 함수            | 일반            | Monomorphization |
| 제네릭 함수            | **다형적 재귀** | 컴파일 에러      |
| Effect만 다형적인 함수 | -               | Evidence passing |

Ability row는 값으로 특수화하지 않는다. 실행 전략은 row 변수에 따라 정의를 따로
만들 수 있다([아래](#row-변수의-convention-class)).

모든 제네릭 함수는 monomorphization으로만 컴파일한다. 값의 표현은 컴파일러가
고르며 코드 작성자가 제어하지 않으므로
([runtime-types.md](runtime-types.md#표현은-의미에-드러나지-않는다)), 특수화로 컴파일할 수 없는 다형적 재귀를
다른 표현으로 대신 컴파일하지 않고 [거부한다](#polymorphic-recursion).
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

CPS 전략이 row 변수의 class마다 만드는 인스턴스는, class가 모두 `Cps`이면 이
이름을 그대로 쓰고 아니면 뒤에 접미사를 붙인다
([class 특수화 pass](cps-effects.md#class-특수화-pass)).

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
fn identity(x: a) -> a { x }

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

호출 그래프에서 서로 도달 가능한 함수들은 하나의 **재귀 그룹**을 이룬다(강연결
요소). 자기 자신을 호출하는 함수는 혼자서도 재귀 그룹이다.

함수 `f`의 본문이 같은 재귀 그룹의 함수 `g`(`f` 자신 포함)를 참조하고, 그 참조가
선택한 타입 인자 중 하나가 `f`의 타입 매개변수를 **진부분으로 포함**하면, 이
참조는 **다형적 재귀**다. 진부분으로 포함한다는 것은 타입 인자가 `f`의 타입
매개변수를 언급하지만 그 매개변수 하나만으로 이루어지지 않았다는 뜻이다.

```rust
struct Pair(a) { left: a, right: a }

fn depth(n: Nat, x: a) -> Nat {
    case n == 0 {
        True -> 0
        False -> 1 + depth(n - 1, Pair { left: x, right: x })
        //           ↑ depth를 Pair(a)로 참조한다
    }
}
```

다형적 재귀가 아닌 경우:

- 같은 타입 매개변수로 다시 인스턴스화하는 재귀(`f(a)`가 `f(a)`를 호출).
- 타입 매개변수의 순서만 바꾸는 재귀(`f(a, b)`가 `f(b, a)`를 호출).
- `f`의 타입 매개변수를 언급하지 않는 타입 인자(`f(a)`가 `f(Int)`를 호출).
- 재귀 그룹 밖의 함수를 `Pair(a)` 같은 타입 인자로 호출하는 경우.

### 거부

Monomorphization은 다형적 재귀에서 끝나지 않는다. `depth(Int)`는
`depth(Pair(Int))`를, 그것은 다시 `depth(Pair(Pair(Int)))`를 요구한다. Tribute는
다형적 재귀를 지원하지 않으며 이를 컴파일 에러로 거부한다.

이 규칙은 정의의 성질이다. 도달 여부와 관계없이 특수화 전에 모든 제네릭 함수의
checked function instance로 검사한다. 판정은 재귀 그룹 안의 각 참조만 보고
내리므로 보수적이다. `f(a)`가 `g`를 `Pair(a)`로 호출하고 `g`가 `f`를 `Int`로만
호출해 실제 인스턴스가 유한하더라도 거부한다.

진단은 다형적 재귀인 참조의 위치에서 보고하며, 호출하는 함수, 호출되는 함수,
그 참조가 요구하는 타입 인자를 호출하는 함수의 타입 매개변수 이름으로 표시한다.

```text
`depth` calls itself at `Pair(a)`; polymorphic recursion is not supported
```

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

---

## Implementation

Monomorphization은 `tribute-front`의 typed AST와 명시적인 함수 인스턴스 기록을
소비한다. 함수 특수화와 nominal 타입 특수화는 같은 frontend 준비 단계에 속한다.

### 파이프라인의 책임

`parse_and_lower_ast()`는 이름 해석과 타입 추론·TDNR을 수행하고, 선택된 함수
인스턴스와 semantic 메타데이터를 반환한다. 타입 검사는 의존 패키지 `std`의
선언을 의존성 표로 조회하며, 그 결과에는 사용자 패키지의 선언만 담긴다.
`prepare_frontend_for_lowering()`은 의존 패키지 본문의 병합, 도달하는 인스턴스
검증, 특수화 및 메타데이터 재작성을 담당한다. IR lowering은 이 준비 단계가 성공한
결과만 소비하며, 타입 검사 결과로부터 lowering 입력을 만드는 경로는 없다.

```text
parse_and_lower_ast: 해석·타입 추론·TDNR·인스턴스 기록
    ↓
prepare_frontend_for_lowering: 의존 패키지 병합·검증·특수화·메타데이터 재작성
    ↓
ast_to_ir: IR lowering
```

### 함수 인스턴스 수집과 특수화

수집과 호출 재작성은 함수 참조 NodeId에 연결된 checked instance의
`(FuncDefId, type_arguments)`를 사용한다. `node_types`나 callable 타입에서 타입
인자를 역추론하지 않는다. 함수 값으로 전달되는 참조도 같은 계약을 따른다.

[다형적 재귀](#polymorphic-recursion) 검사가 이 과정보다 먼저 실행된다. 이어서
선택된 선언 스킴, 인자 개수, callable 및 row 인자의 일관성을 검사한다.
원본 함수의 binder 순서에 맞춰 타입을 치환하고, 특수화된 정의의 모든 typed
메타데이터에도 같은 치환을 적용한다. 스킴 본문과 합집합·차집합 제약은 함께
변환한다. 생성된 clone 내부의 참조가 새 인스턴스를 드러내면 같은 인스턴스 키를
재사용하며 고정점까지 수집한다. 확장 한도는 다형적 재귀 검사가 잡지 못한 다른
무한 확장을 막는 안전장치이며, 한도를 넘으면 구조적 진단을 반환한다.

타입 인자를 갖는 함수를 이 과정으로 특수화한다. 타입 인자가 없는 함수는 정의 하나를
그대로 쓴다. [Class 변수](#row-변수의-convention-class)에 따른 특수화는 이 과정이 아니라
source-logical IR의 pass가 맡는다. 도달하지 않는 generic
template은 허용하되, 도달하는 함수 인스턴스나 ability 인자가 미해결이면 타입 erasure
전에 거부한다. 다형적 재귀는 정의의 성질이므로 도달하지 않는 template에서도 거부한다.

### 타입 순회와 binder별 치환

Frontend의 타입 순회와 치환은 하나의 구조적 순회를 공유한다. 순회는 `Named`의
타입 인자, 함수의 매개변수·결과·ability row, tuple 원소, continuation의 인자·결과·
ability row로 내려가며, row 안에서는 각 ability 인자의 타입 인자로 내려간다. 스킴을
순회하면 본문에 이어 보존된 합집합 제약(각 source, 그다음 result)과 차집합 제약
(source, removed, result)의 row를 같은 규칙으로 순회한다. 순서는 이 나열 순서이며,
등장 순서로 binder 번호를 정하는 일반화가 이 순서에 의존한다. 순회는 스킴의 binder
목록 자체를 바꾸지 않는다. 바뀐 부분이 없는 타입과 row는 원래 값을 그대로 쓴다.

치환 주체와 실패 처리는 binder 종류마다 다르다.

- 스킴의 `BoundVar`는 인스턴스의 타입 인자로 치환한다. 인자 범위를 벗어난 index는
  치환 실패로 돌려주며, 진단으로 보고할지 내부 불변식 위반으로 중단할지는 호출자가
  정한다. 치환된 인자 안으로는 다시 내려가지 않으므로 호출자의 타입에 있는 row
  변수는 인스턴스화의 row 변수 재명명을 받지 않는다.
- `UniVar`는 solver가 자신의 치환으로 해석한다. 치환된 타입은 다시 해석하며, row는
  solver의 row 치환을 먼저 적용한 뒤 그 ability 인자를 해석한다. 일반화는 남은
  `UniVar`를 함수 스킴의 `BoundVar` 또는 지역 스킴의 `LocalBoundVar`로 바꾼다.
- `LocalBoundVar`는 지역 binder를 소유한 typed-body 메타데이터가 해석한다. 스킴
  인스턴스화와 solver 치환은 이를 바꾸지 않고 통과시킨다.

특수화가 typed 메타데이터에 하는 세 작업, 즉 clone의 타입 인자 치환, nominal
재작성, nominal 수집 시작점 순회는 모든 메타데이터 표의 타입을 같은 순회로 다룬다.
함수 인스턴스 기록의 선언 스킴은 callee 자신의 binder로 쓰였으므로 호출자 쪽
치환·재작성·수집의 대상이 아니다. 이 작업들은 그 기록의 callable, 타입 인자, row
인자만 다룬다. 지역 callable 인스턴스의 스킴은 감싸는 함수의 binder로 쓰였으므로
다른 타입과 함께 다룬다.

이름 mangling, 표시 형식, lowering의 identity key처럼 출력 형식이 계약인 변환은
이 순회와 다른 순서나 정렬을 쓸 수 있으며, 각자의 형식 규칙을 따른다.

### Row 변수의 convention class

Ability row는 타입 인자처럼 치환하지 않는다. Row의 내용은 실행 중 evidence로
전달한다. 실행 전략은 row 변수가 요구하는 calling convention마다 정의를 따로 만들 수
있으며, CPS 전략에서 이를 그 변수의
[convention class](cps-effects.md#row-변수의-convention-class)라 한다. Class마다
인스턴스를 만드는 일은 frontend의 특수화가 아니라 CPS 전략의
[class 특수화 pass](cps-effects.md#class-특수화-pass)가 source-logical IR에서 한다.
이름 있는 정의는 class 인자마다 인스턴스 하나가 되고, 람다와 함수 참조는 class마다
[convention entry](ir.md#convention-entry)를 가진 값이 된다. 따라서 callable 값의 타입은 class에
따라 갈라지지 않으며, 같은 값을 다른 class의 자리에서 그대로 쓴다.

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
IR 생성에 전달하지 않는다. 다형적 재귀는 이 한도에 앞서 [거부한다](#polymorphic-recursion).

### 이름과 identity

맹글링은 생성된 선언의 이름을 부여한다. 이름으로 선언 identity나 인스턴스 인자를
복원하지 않는다. 예를 들어 `identity`와 `[Int]`는 `identity$Int`로 표현하고, 중첩
타입 인자는 `$0`/`$1`로 감싸지만 특수화 키는 정확한 선언 ID와 타입 인자로 유지한다.

인스턴스의 path는 선언의 path에서 마지막 component만 맹글링한 것이다. 둘러싼
모듈의 component는 그대로 둔다. 맹글링에 쓰인 타입 인자나 ability의 표기에 `::`가
들어 있어도 맹글링된 이름은 component 하나다. `A::identity`와 `[B::Thing]`의
인스턴스는 component `A`와 `identity$B::Thing`의 path다.

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
nominal definition identities. A source declaration's identity is its origin
kind, its package-qualified path, and the number of same-named type
declarations before it in its module (see `types.md`), so reparsing the same
declaration yields the same identity, and an edit elsewhere changes it only
when it changes that path or that count.
Prelude merging, deferred method desugaring, and specialized AST cloning
preserve the record. Clones substitute their enclosing type arguments into
nested instance records.
Effect-only type arguments participate in ordinary type specialization; open
residual rows continue to use evidence passing. The recorded row arguments
become the `row_args` of references in source-logical IR, from which the class
specialization pass selects each instance's
[convention classes](#row-변수의-convention-class).
Unused generic templates are permitted, but incomplete reached instances fail
before logical lowering.

Function quantifiers cover the callable interface and retained semantic row
relations. Inference variables occurring only in the checked body remain
body-owned local existentials; they do not add phantom arguments to callers.
The pre-erasure ability-instance check still rejects unresolved local
existentials in reachable handler or perform ability arguments.
