# Tribute Type System

> 이 문서는 Tribute의 타입 선언 문법, record, UFCS 규칙을 정의한다.

## Type Declarations

Tribute는 `struct`와 `enum` 두 키워드로 타입을 선언한다.

### Nominal Identity

Every `struct` and `enum` declaration introduces a distinct nominal type
identity. Identity comes from the resolved declaration, not from the displayed
type name alone. Two declarations with the same short name remain different
types, including declarations in different nested modules.

선언 identity는 선언의 출처 종류(source 선언, compiler 소유 타입), package를
포함한 path, 그리고 같은 module에서 그 선언보다 앞선 같은 이름의 타입 선언 수로
정해진다. 마지막 값은 module이 이름을 다시 선언하지 않으면 0이며, 다시 선언한
잘못된 module에서도 각 선언이 자기 identity를 갖게 한다. 이름으로 타입을 가리키는
참조는 그 path의 첫 선언을 뜻한다. Source 위치는 identity에 들어가지 않으므로, 같은
선언을 다시 파싱하거나 다른 선언을 편집해도 identity는 바뀌지 않는다. 바뀌는 것은
편집이 선언의 path나 그보다 앞선 같은 이름의 타입 선언 수를 바꿀 때뿐이다. 같은 철자의
source 타입과 compiler 소유 타입(`List`)은 출처 종류로 구별된다.

Qualified and locally unqualified references to one declaration resolve to the
same identity. Type display remains source-oriented and may use the same short
spelling for different declarations; equality, unification, method receiver
selection, and specialization must still compare declaration identity.

특수화된 nominal 타입은 해석된 선언의 qualified name을 기준으로 타입 인자를
맹글링한 normalized symbol을 사용한다. 예를 들어 서로 다른 선언
`A::Token(Nat)`과 `B::Token(Nat)`은 각각 `A::Token$Nat`, `B::Token$Nat`이다.
특수화된 타입의 path는 선언의 모듈 component를 유지하고 마지막 component만
맹글링한다.
중첩 타입 인자는 기존 `$0`/`$1` 맹글링 규칙을 따른다. 이 symbol은 IR의
`adt.typeref`와 대응하는 struct/enum layout의 `name`에 동일하게 사용하며,
선언 identity나 타입 인자를 이름에서 역으로 복원하는 근거로 쓰지 않는다.

## Primitive Numeric Types

`Float`는 IEEE 754 부동소수점 값을 표현한다. 현재 산술/비교 연산은
기본 연산자 subset만 정의한다.

Float 비교는 C/Rust 스타일 NaN 의미를 따른다:

| 연산자 | IR predicate | NaN 포함 시 결과 |
| ------ | ------------ | ---------------- |
| `==`   | `oeq`        | `False`          |
| `!=`   | `une`        | `True`           |
| `<`    | `olt`        | `False`          |
| `<=`   | `ole`        | `False`          |
| `>`    | `ogt`        | `False`          |
| `>=`   | `oge`        | `False`          |

전체 ordered/unordered predicate 집합은 아직 surface language에 노출하지
않는다.

## Never와 표현식의 제거 규칙

`Never`는 값이 없는 타입이며, 현재 continuation으로 돌아오지 않는 계산을
나타낸다. `Nil`을 비롯한 다른 모든 구체적인 타입과 구별된다. 정상적인 값은
기대 타입이 `Never`인 위치에 사용할 수 없다.

실제 타입이 `Never`인 표현식은 암묵적 제거를 통해 어떤 기대 타입으로도 검사할
수 있다. 이는 방향성 있는 표현식 규칙이며 타입 equality나 일반 subtyping이
아니다. 기존 `List(Never)`, tuple, 명목 타입, 함수 값의 내부에 재귀적으로 적용하지
않는다. 람다 리터럴은 본문을 문맥의 기대 반환 타입으로 검사하며, 이 규칙이 기존
함수 값의 variance를 허용하지는 않는다.

Case 결과와 리스트 리터럴 원소는 공통 타입 관계를 사용한다. `Never`가 아닌
결과들은 같은 타입이어야 하며, 모두 `Never`이면 공통 타입도 `Never`다. 빈 리스트의
원소 타입은 fresh 변수로 남는다. Handle의 공통 answer에는 정상 완료 경로(`do`,
또는 `do`가 없으면 처리 대상 본문)와 `op` handler의 반환 경로가 포함된다.
`fn` handler는 operation의 결과 타입을 반환한다. `resume`이 바깥 answer를 참조하는
것은 그 answer 타입을 결정하는 독립적인 근거가 아니다.

이 규칙은 명시적 제거 표현식이나 빈 case 문법을 추가하지 않는다.

## Function Results

Source-level functions have exactly one logical result. `Nil` and `Never` are
ordinary logical result types rather than absence-of-result markers: a Nil
function has the logical result list `[core.nil]`, and a function whose result
type is `Never` has `[core.never]`. How an execution strategy represents a
callable that returns through a continuation instead, such as CPS with an empty
result list, is specified with that strategy ([cps-effects.md](cps-effects.md)).

각 타겟은 callable contract를 독립적으로 소유한다. 특히 네이티브 타겟은
`clif.func_sig`로 순서 있는 0개 이상의 결과 목록을 표현할 수 있지만, 이 사실이
소스 함수의 단일 결과나 공통 `func.func_sig` 계약을 넓히지는 않는다.

### Struct (Product Type)

```rust
struct User {
    name: String
    age: Int
}

struct Point {
    x: Int
    y: Int
}

// 제네릭 struct
struct Box(a) {
    value: a
}

struct Pair(a, b) {
    first: a
    second: b
}

// 한 줄이면 쉼표 필수
struct Flags { read: Bool, write: Bool, execute: Bool }
```

### Enum (Sum Type)

```rust
// 단순 enum
enum Bool {
    False
    True
}

// 제네릭 enum - positional fields
enum Option(a) {
    None
    Some(a)
}

enum Tree(a) {
    Leaf(a)
    Branch(Tree(a), Tree(a))
}

// Named fields를 가진 variant
enum Result(a, e) {
    Ok { value: a }
    Error { error: e }
}

// 혼합 가능
enum Expr {
    Lit(Int)
    Var(String)
    BinOp { op: String, lhs: Expr, rhs: Expr }
}

// 한 줄이면 쉼표 필수
enum Ordering { Less, Equal, Greater }
```

### Enum variant의 이름

Enum variant는 그 enum의 [동명 namespace](modules.md#타입과-동명-네임스페이스)에만
선언된다. Enum을 둘러싼 모듈에는 variant의 이름이 생기지 않는다. `Option`의
variant는 `Option::None`과 `Option::Some`이고, 이 경로가 variant의 식별자다.

- Variant는 경로(`Option::Some(1)`), [암묵적 생성자](type-inference.md#암묵적-생성자)
  (`.Some(1)`), 또는 그 variant를 `use`한 짧은 이름으로 가리킨다
  (`use std::Option::Some` 뒤의 `Some(1)`). 짧은 이름 `Some`은 보통의 이름
  조회이므로 `use` 없이는 해석 오류다. 스코프의 타입에 그 이름의 생성자가 있으면
  진단은 `.Some`을 제안한다.
- 다른 enum의 variant는 이름이 같아도 서로의 이름을 가리지 않는다. `Result::Error`와
  `std::io::Error`는 둘 다 짧은 이름으로 스코프에 들어오지 않는 한 충돌하지 않는다.
- Struct의 생성자는 struct 타입과 같은 이름으로 struct를 선언한 모듈에 있다.
- Variant는 따로 가시성을 쓰지 않으며, 그 enum의 가시성을 따른다.

### 타입 파라미터

타입 파라미터는 소괄호를 사용한다:

```rust
struct Box(a) { value: a }
enum Option(a) { None, Some(a) }
enum Result(a, e) { Ok { value: a }, Error { error: e } }

// 사용
let box: Box(Int) = Box { value: 42 }
let opt: Option(String) = .Some("hello")
let res: Result(Int, String) = .Ok { value: 42 }
```

### 선언 안의 함수 타입

타입 선언에는 row를 묶는 매개변수가 없다. 그래서 struct·enum 필드와 ability
operation의 매개변수·결과에 쓴 함수 타입은 함수 시그니처와 달리 fresh row 변수를
만들지 않는다:

- Ability annotation을 생략하면 빈 row이다. `fn(a) -> b`는 `fn(a) ->{} b`와 같다.
- Row 변수를 쓰면 묶이지 않은 변수이므로 컴파일 에러다.

```rust
struct Handler { run: fn(Int) -> Int }               // fn(Int) ->{} Int
enum Step { Next(fn() ->{State(Int)} Nil), Done }    // 구체적 ability는 허용
struct Bad { run: fn(Int) ->{e} Int }                // Error: e가 묶이지 않음
```

열린 row의 callback은 순수 필드에 담을 수 없다. 함수 시그니처의 생략된 row는
fresh 변수이므로, 필드에 담을 callback은 시그니처에서 `->{}`로 명시한다.

---

## Construction

### Struct 생성

중괄호를 사용한다:

```rust
let user = User { name: "Alice", age: 30 }
let point = Point { x: 10, y: 20 }
let box = Box { value: 42 }
```

### Enum Variant 생성

Positional field는 소괄호, named field는 중괄호. Variant는 enum의 namespace
경로나 [암묵적 생성자](type-inference.md#암묵적-생성자)로 가리킨다:

```rust
// Positional
let some = Option::Some(42)
let none: Option(Int) = .None
let tree = Tree::Branch(.Leaf(1), .Leaf(2))

// Named
let ok: Result(Int, String) = .Ok { value: 42 }
let error = Result::Error { error: "something went wrong" }

// 암묵적 생성자 패턴은 매칭하는 값의 타입에서 생성자를 찾는다
case opt {
    .Some(x) -> x
    .None -> 0
}
```

## List

`List(a)` is an opaque nominal immutable persistent sequence. Its nominal
identity is compiler-owned and distinct from every source declaration, including
a user declaration also spelled `List`. Name-based equality is not sufficient
for named types: type checking and lowering compare declaration identities.

The concrete RRB-node layout is not a source contract. In particular, `Empty`
and `Cons` are not public constructors or patterns. Construction uses list
literals and the canonical persistent prepend operation:

```rust
let empty: List(Int) = []
let values = [first(), second(), third()]
let dynamic = List::prepend(next_value, previous_values)
```

List literal element expressions are evaluated from left to right, exactly once.
Construction performed after those evaluations must preserve the same sequence
order. Lists are immutable and persistent: observing a tail never changes the
original list, and a tail retains the original element order.
`List::prepend(value, tail)` returns a new canonical `List(a)` whose first
element is `value` and whose remaining sequence is `tail`; it does not mutate or
expose the representation of `tail`. This is the minimal general source-level
List construction operation.

List patterns use sequence views:

```rust
[]                    // exactly empty
[x, y]                // exactly two elements
[head, ..tail]        // at least one element; tail is a List(a)
[first, second, ..]   // at least two elements; ignore the remainder
```

Exact patterns require the stated length. Prefix-rest patterns require at least
the prefix length and bind the remaining sequence without copying or mutation.
Element subpatterns are matched left to right.

`List(a)` 원소 패턴이 관찰하고 바인딩하는 값의 타입은 `a`다.

Every backend represents `List(a)` as an RRB tree. The branching factor, node
packing, and allocation layout are target-private, while persistence, logarithmic
concatenation, and efficient slicing are representation requirements. A
transient or uniqueness optimization is optional and must not alter source
syntax, `List(a)` identity, or shared `list.*` IR contracts.

### Shorthand Syntax

변수명과 필드명이 같으면 생략 가능 (Rust 스타일):

```rust
let name = "Alice"
let age = 30

// 이 둘은 동일
let user = User { name: name, age: age }
let user = User { name, age }
```

### Spread Syntax

`..`로 기존 값을 복사하며 일부 필드만 변경:

```rust
let user = User { name: "Alice", age: 30 }

// 일부 필드만 변경
let older = User { ..user, age: 31 }
let renamed = User { ..user, name: "Jane" }

// 여러 필드 변경
let updated = User { ..user, name: "Jane", age: 31 }
```

### 레코드 필드 검증

레코드 생성은 이름 해석으로 확정한 constructor 선언의 필드를 기준으로 검사한다.
Constructor는 nominal struct이거나 모든 필드에 이름이 있는 enum variant다.
명시한 필드는 해당 선언에 속해야 하며, 같은 필드를 두 번 지정할 수 없다.
Spread가 없으면 모든 선언 필드를 지정해야 한다. Spread는 생략한 필드를 제공하지만,
알 수 없는 필드나 중복 필드를 허용하지는 않는다.

Spread는 struct 레코드에만 쓸 수 있다. Enum 타입의 값은 어느 variant인지 정적으로
알 수 없으므로, variant 레코드는 모든 필드를 명시해야 한다. 위치 기반 필드를 가진
variant는 레코드 형식으로 생성할 수 없고 소괄호 형식으로 생성한다.

이러한 오류는 IR 생성 전 타입 검사 단계에서 진단한다. 명시한 필드를 소스 순서대로
검사하여 알 수 없는 필드와 중복 필드를 보고한다. Spread가 없으면 그 뒤에 선언 순서상
첫 번째 누락 필드를 보고한다. 알 수 없는 필드를 반복해서 지정하면 각 항목을 알 수
없는 필드로 진단하고, 선언에 있는 필드를 반복했을 때만 중복 필드로 진단한다.
진단은 레코드 전체의 span과 이름 해석으로 확정한 constructor의 한정 이름을 유지한다.

필드 구성에 오류가 있어도 모든 명시적 필드 표현식과 spread의 타입을 검사한다.
정상 레코드는 spread를 먼저 평가하고, 명시적 필드를 소스 순서대로 각각 정확히 한 번
평가한다. 평가한 값은 선언 필드 순서에 맞춰 조립한다.

필드 타입에 사용하지 않는 phantom 타입 매개변수를 허용한다. 예를 들어
`struct Tag(a) {}`의 `Tag {}`는 기대 타입 `Tag(Int)`를 통해 `a`를 결정할 수 있다.
별도 marker 필드를 요구하지 않으며, 필드에 쓰이지 않아도 타입 인자는 타입 동등성
검사에 참여한다. 따라서 이미 `Tag(Bool)`인 값은 `Tag(Int)` 자리에 사용할 수 없다.

Generic struct의 spread는 생성할 타입과 같은 선언 및 일치하는 타입 인자를 사용해야
한다. 모든 필드를 명시적으로 덮어쓰더라도 타입 인자를 바꾸는 업데이트는 허용하지
않는다. 필드에 나타나지 않는 phantom 타입 인자에도 같은 규칙을 적용한다.

---

## Slots

`Slots(a)`는 [compiler 소유 runtime 모듈](modules.md#compiler-소유-runtime-모듈)만
쓰는 고정 길이 불변 배열이다. 패키지 `std` 밖에서는 보이지 않으며 사용자 언어의
collection이 아니다. 타입 identity는 compiler 소유이며 같은 이름의 source 선언과
구별된다. IR에서는 원소 타입이 `a`인 [`adt.array`](ir.md#adt-배열)다. 원소가
aggregate이면 배열은 그 object에 대한 managed 참조를 담는다.

연산은 `std::runtime::Slots`의 compiler intrinsic이며 각각 `adt` 배열 operation
하나로 낮춘다. 위치와 길이는 `Nat`이고 lowering이 `arith.index_castui`로
`core.index`로 바꾼다.

| 연산 | 뜻 | `adt` operation |
| --- | --- | --- |
| `empty() -> Slots(a)` | 원소가 없는 배열 | `adt.array_of` |
| `len(slots: Slots(a)) -> Nat` | 원소 수 | `adt.array_len` |
| `get(slots: Slots(a), at: Nat) -> a` | `at`번째 원소 | `adt.array_get` |
| `insert(slots: Slots(a), at: Nat, value: a) -> Slots(a)` | `at` 앞에 `value`를 넣은 새 배열 | `adt.array_splice` (`start = end = at`) |
| `remove(slots: Slots(a), at: Nat) -> Slots(a)` | `at`번째 원소를 뺀 새 배열 | `adt.array_splice` (`end = at + 1`, 원소 없음) |
| `replace(slots: Slots(a), at: Nat, value: a) -> Slots(a)` | `at`번째 원소를 `value`로 바꾼 새 배열 | `adt.array_splice` (`end = at + 1`) |

`Slots`를 만든 뒤 바꾸는 연산은 없다. `insert`, `remove`, `replace`는 원래 배열을
그대로 두고 새 배열을 만든다. `get`, `remove`, `replace`는 `at < len`을,
`insert`는 `at ≤ len`을 요구하며, target은 이를 어긴 실행을 trap으로 끝낸다.

---

## Field Access

### UFCS와 괄호 생략

Struct 필드는 자동으로 getter 함수를 생성한다. UFCS와 결합하여 인자가 없으면 괄호 생략이 가능하다:

```rust
struct User {
    name: String
    age: Int
}

// 자동 생성되는 함수
// User::name : fn(User) -> String
// User::age  : fn(User) -> Int

// 필드 접근 (괄호 생략)
user.name      // User::name(user)
user.age       // User::age(user)

// 일반 함수도 인자 없으면 괄호 생략 가능
option.is_some // Option::is_some(option)
```

### Uniform Access Principle

필드 접근과 함수 호출이 동일한 문법을 사용하므로, 구현 변경이 API를 깨지 않는다:

```rust
// v1: 저장된 필드
struct User {
    name: String
    first_name: String
    last_name: String
}

user.name  // 필드 접근

// v2: 계산된 값으로 변경
struct User {
    first_name: String
    last_name: String
}

mod User {
    fn name(self: User) -> String {
        self.first_name <> " " <> self.last_name
    }
}

user.name  // 함수 호출 - 호출 코드 변경 불필요
```

### 인자가 있을 때

인자가 있는 함수는 괄호 필수:

```rust
option.map(fn(x) x + 1)    // OK
string.split(",")          // OK
```

---

## Field Update

### Setter와 Modifier

각 필드에 대해 `::set`과 `::modify` 함수도 자동 생성된다:

```rust
struct User {
    name: String
    age: Int
}

// 자동 생성되는 함수
// User::name         : fn(User) -> String
// User::name::set    : fn(User, String) -> User
// User::name::modify : fn(User, fn(String) -> String) -> User

// 사용 예시
user.name::set("Jane")           // User::name::set(user, "Jane")
user.age::modify(fn(n) n + 1)    // User::age::modify(user, fn(n) n + 1)
```

생성 규칙:

- 이름 있는 struct 필드마다 getter `T::f`, setter `T::f::set`, modifier
  `T::f::modify`를 struct와 동명인 모듈 `T`에 만든다. 위치 필드와 enum variant의
  필드에는 만들지 않는다.
- Generic struct에서는 세 함수 모두 struct의 타입 매개변수를 그대로 받는다.
  Setter와 modifier는 타입 인자를 바꾸지 않는다(spread와 같은 규칙).
  `struct Box(a) { value: a }`이면 `Box::value::set : fn(Box(a), a) -> Box(a)`이다.
- Modifier는 콜백의 ability row에 대해 다형적이다:
  `T::f::modify : fn(T, fn(F) ->{e} F) ->{e} T`. Getter와 setter는 순수하다.
- `T::f`는 값 namespace의 getter이면서 `set`·`modify`를 담는 모듈 `T::f`이기도
  하다. Companion `mod T`가 `f::set`이나 `f::modify`를 직접 선언하면 생성된
  함수와 충돌하는 중복 정의 오류이다.
- 세 함수의 path에서 struct와 필드는 각각 component 하나다. `T`의 path에 `f`를
  덧붙인 것이 getter이고, 거기에 `set`이나 `modify`를 덧붙인 것이 setter와
  modifier다.
- 생성된 `set`·`modify`는 경로로만 호출한다. 비한정 UFCS `x.set(v)`의 후보가
  아니다.
- 세 함수에는 source 선언이 없다. Name resolution은 struct의 필드에서 이름을
  등록하고, type checking은 필드 타입에서 각 함수의 type scheme을 만든다. AST에서
  IR로 내리는 단계가 이름 있는 필드마다 getter·setter·modifier의 본문을 일반 함수
  정의로 내보내며, 모든 사용은 일반 호출이나 함수 참조이다. 쓰이지 않는 정의는
  [source-logical global DCE](optimizations.md#적용-위치)가 지운다.

### Spread vs Setter

```rust
// Spread - 여러 필드 변경에 적합
User { ..user, name: "Jane", age: 31 }

// Setter - 체이닝에 적합
user
    .name::set("Jane")
    .age::modify(fn(n) n + 1)
```

---

## Pattern Matching

### 기본 패턴

```rust
case opt {
    .Some(x) -> x
    .None -> 0
}

case result {
    .Ok { value } -> value
    .Error { error } -> panic(error)
}
```

### Destructuring

```rust
// Struct destructuring
let User { name, age } = user

// 일부 필드만 (나머지 무시)
let User { name, .. } = user
let Point { x, .. } = point

// 필드 이름 변경
let User { name: user_name, age: user_age } = user

// 혼합
let User { name, age: user_age, .. } = user
```

### 패턴 매칭에서 `..`

```rust
case user {
    User { name: "Admin", .. } -> "admin user"
    User { age, .. } && age < 18 -> "minor"
    User { name, .. } -> "user: " <> name
}

case result {
    .Ok { value, .. } -> handle_value(value)
    .Error { error, .. } -> handle_error(error)
}
```

### 패턴 필드 검증

중괄호 생성자 패턴 `Name { field: pattern, .. }`은 필드를 이름으로 매칭한다. 필드를
쓰는 순서는 자유롭다. `Point { y, x }`는 `Point { x, y }`와 같은 값을 같은 이름에
바인딩한다. `name`만 쓴 필드는 `name: name`의 줄임이다.

- `..`가 없으면 모든 선언 필드를 써야 한다. `..`가 있으면 생략한 필드는 어떤 값과도
  매칭한다. `..`는 마지막에 한 번만 쓸 수 있다. 필드 없이 `Name { .. }`로 쓰면
  모든 필드를 무시하고 생성자만 매칭한다.
- 알 수 없는 필드, 중복 필드, 누락 필드는 레코드 생성과 같은 규칙으로 타입 검사
  단계에서 진단한다. 진단은 패턴 전체의 span을 사용한다.
- 중괄호 형식은 모든 필드에 이름이 있는 생성자와 필드가 없는 생성자에만 쓸 수 있다.
  위치 기반 필드를 가진 variant는 소괄호 형식으로 매칭한다.
- 소괄호 생성자 패턴은 생성자의 필드 수와 같은 수의 하위 패턴을 가져야 한다. 괄호
  없이 쓴 생성자 이름은 필드가 없는 패턴이므로, 필드가 있는 생성자에 쓰면 오류다.
- 생성자 자리에 생성자가 아닌 이름이 오면 오류다.

### 리터럴 패턴

리터럴 패턴은 scrutinee가 그 리터럴과 같을 때 매칭된다. 같음의 기준은 그 타입의
`==`와 같다.

- `Nat`, `Int`, `Bool`, `Rune`은 값으로 비교한다. `Rune`은 코드포인트를 비교한다.
- `Float`는 `==`와 같은 `oeq`로 비교한다. NaN은 어떤 리터럴 패턴과도 매칭되지 않고,
  `0.0` 패턴은 `-0.0`과도 매칭된다.
- `String`과 `Bytes`는 prelude가 정의한 그 타입의 `==`로 비교한다. `String`은 rope
  모양과 무관하게 내용으로 비교한다.
- `Nil`은 값이 하나뿐이므로 항상 매칭된다. 빈 튜플 `#()`은 없다.

```rust
case command {
    "quit" -> 0
    "help" -> 1
    _ -> 2
}
```

### 망라성과 도달 불가 arm

모든 `case`는 망라적이어야 한다. scrutinee가 가질 수 있는 값 중 어떤 arm에도
매칭되지 않는 값이 있으면 오류이며, 진단은 scrutinee의 타입과 누락된 값을 나타내는
예시 패턴을 최대 세 개 보여준다.

```rust
case opt {
    .None -> 0
    .Some(True) -> 1
}
// error: non-exhaustive case expression on `std::Option(Bool)`: missing patterns: .Some(False)
```

검사는 타입 검사가 소유하며, 패턴 행렬의 usefulness로 판정한다. 검사는 이름 해석과
[암묵적 생성자](type-inference.md#암묵적-생성자) 해석이 확정한 생성자를 본다.
예시 패턴의 enum variant는 암묵적 생성자 형식으로 보인다. 그 타입은 진단이 보이는
scrutinee의 타입과 상위 패턴의 필드 타입으로 정해진다. 중첩 패턴도 모두
분석하므로 `.Some(True)`는 `.Some(_)`를 덮지 않는다. 튜플 scrutinee는 각 요소를 한
열로 보고 함께 분석한다.

- 유한한 구성자 집합으로 다루는 것: 튜플, enum variant, struct(구성자 하나),
  `Bool`(`True`, `False`), `Nil`, 리스트 길이(`[]`, `[a, b]`, `[a, ..rest]`).
  enum은 모든 variant가 나타나야 완전하다.
- 숫자, 문자열, bytes, rune 리터럴은 도메인이 무한한 것으로 다루므로, 이를 덮으려면
  `_`나 바인딩 패턴이 필요하다.
- guard가 있는 arm은 망라성에 기여하지 않는다.
- 앞선 guard 없는 arm들이 이미 모두 덮는 arm은 도달할 수 없으며,
  `unreachable pattern` 경고를 받는다.
- 분석할 수 없는 패턴을 가진 arm은 guard가 있는 arm처럼 망라성에 기여하지
  않는다. 나머지 arm만으로 망라되면 그 `case`는 망라적이다.
- 망라성을 증명하지 못한 `case`는 망라적이지 않은 것으로 보고 오류로 거부한다.
  분석할 수 없는 arm이 누락된 값을 덮을 수도 있는 경우와, 검사가 정해진 단계
  한도 안에 끝나지 않는 경우가 여기에 해당한다. 진단은 `_` arm을 추가하라고
  안내한다.

망라적인 `case`에서는 guard 없는 arm들이 모든 값을 덮으므로, 마지막 guard 없는
arm 뒤의 arm은 도달할 수 없고 `unreachable pattern` 경고를 받는다. lowering은
이 arm들을 버리고 마지막 guard 없는 arm을 패턴 검사 없이 실행한다. 따라서 모든
arm이 실패하는 경로는 lowering 결과에 남지 않으며, 망라성 판정은 중첩 패턴까지
정확해야 한다.

---

## Named vs Positional Fields

### Enum에서의 선택

Enum variant는 positional 또는 named field 중 선택:

```rust
// Positional - 필드가 1-2개이고 의미가 명확할 때
enum Option(a) {
    None
    Some(a)
}

// Named - 필드가 여러 개이거나 의미를 명확히 할 때
enum Result(a, e) {
    Ok { value: a }
    Error { error: e }
}

// 같은 enum에서 혼합 가능
enum Expr {
    Lit(Int)
    Var(String)
    BinOp { op: String, lhs: Expr, rhs: Expr }
}
```

### 패턴 매칭

```rust
case expr {
    .Lit(n) -> n
    .Var(name) -> lookup(name)
    .BinOp { op: "+", lhs, rhs } -> eval(lhs) + eval(rhs)
    .BinOp { op, lhs, rhs } -> apply_op(op, eval(lhs), eval(rhs))
}
```

---

## Summary

| 문법 | 의미 |
| ---- | ---- |
| `struct Name { field: Type }` | Product type 선언 |
| `enum Name { Variant }` | Sum type 선언 |
| (타입 선언에서 개행 시 쉼표 생략 가능) | `struct`, `enum` 내부에서만 적용 |
| `Type(a, b)` | 타입 파라미터 |
| `Name { field: value }` | Struct/named variant 생성 |
| `Enum::Variant(value)` | Positional variant 생성 |
| `.Variant(value)`, `.Variant { field: value }` | 암묵적 생성자 |
| `Name { field }` | Shorthand (변수명 = 필드명) |
| `Name { ..x, field: value }` | Spread (복사 후 일부 변경) |
| `x.field` | 필드 접근 (UFCS, 괄호 생략) |
| `x.field::set(v)` | 필드 설정 |
| `x.field::modify(f)` | 필드 변환 |
| `let Name { field, .. } = x` | Destructuring (나머지 무시) |
