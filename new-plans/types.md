# Tribute Type System

> 이 문서는 Tribute의 타입 선언 문법, record, UFCS 규칙을 정의한다.

## Type Declarations

Tribute는 `struct`와 `enum` 두 키워드로 타입을 선언한다.

### Nominal Identity

Every `struct` and `enum` declaration introduces a distinct nominal type
identity. Identity comes from the resolved declaration, not from the displayed
type name alone. Two declarations with the same short name remain different
types, including declarations in different nested modules.

Qualified and locally unqualified references to one declaration resolve to the
same identity. Type display remains source-oriented and may use the same short
spelling for different declarations; equality, unification, method receiver
selection, and specialization must still compare declaration identity.

특수화된 nominal 타입은 해석된 선언의 qualified name을 기준으로 타입 인자를
맹글링한 normalized symbol을 사용한다. 예를 들어 서로 다른 선언
`A::Token(Nat)`과 `B::Token(Nat)`은 각각 `A::Token$Nat`, `B::Token$Nat`이다.
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

Source-level functions have exactly one logical result. `Unit` and `Never` are
ordinary logical result types rather than absence-of-result markers: a Unit
function has the logical result list `[core.nil]`, and a CPS-transformed
function has the logical result list `[core.never]`. General TrunkIR also supports
empty function result lists, distinct from both
logical types. An empty list alone does not prove physical CPS: that requires
the Cps calling convention together with the exact empty result list.

각 타겟은 callable contract를 독립적으로 소유한다. 특히 네이티브 타겟은
`clif.func_sig`로 순서 있는 0개 이상의 결과 목록을 표현할 수 있지만, 이 사실이
소스 함수의 단일 결과나 공통 `func.func_sig` 계약을 넓히지는 않는다.

### Struct (Product Type)

```rust
struct User {
    name: Text
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
    Var(Text)
    BinOp { op: Text, lhs: Expr, rhs: Expr }
}

// 한 줄이면 쉼표 필수
enum Ordering { Less, Equal, Greater }
```

### 타입 파라미터

타입 파라미터는 소괄호를 사용한다:

```rust
struct Box(a) { value: a }
enum Option(a) { None, Some(a) }
enum Result(a, e) { Ok { value: a }, Error { error: e } }

// 사용
let box: Box(Int) = Box { value: 42 }
let opt: Option(Text) = Some("hello")
let res: Result(Int, Text) = Ok { value: 42 }
```

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

Positional field는 소괄호, named field는 중괄호:

```rust
// Positional
let some = Some(42)
let none = None
let tree = Branch(Leaf(1), Leaf(2))

// Named
let ok = Ok { value: 42 }
let error = Error { error: "something went wrong" }
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

레코드 생성은 이름 해석으로 확정한 nominal struct 선언의 필드를 기준으로 검사한다.
명시한 필드는 해당 선언에 속해야 하며, 같은 필드를 두 번 지정할 수 없다.
Spread가 없으면 모든 선언 필드를 지정해야 한다. Spread는 생략한 필드를 제공하지만,
알 수 없는 필드나 중복 필드를 허용하지는 않는다.

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

## Field Access

### UFCS와 괄호 생략

Struct 필드는 자동으로 getter 함수를 생성한다. UFCS와 결합하여 인자가 없으면 괄호 생략이 가능하다:

```rust
struct User {
    name: Text
    age: Int
}

// 자동 생성되는 함수
// User::name : fn(User) -> Text
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
    name: Text
    first_name: Text
    last_name: Text
}

user.name  // 필드 접근

// v2: 계산된 값으로 변경
struct User {
    first_name: Text
    last_name: Text
}

mod User {
    fn name(self: User) -> Text {
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
    name: Text
    age: Int
}

// 자동 생성되는 함수
// User::name         : fn(User) -> Text
// User::name::set    : fn(User, Text) -> User
// User::name::modify : fn(User, fn(Text) -> Text) -> User

// 사용 예시
user.name::set("Jane")           // User::name::set(user, "Jane")
user.age::modify(fn(n) n + 1)    // User::age::modify(user, fn(n) n + 1)
```

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
    Some(x) -> x
    None -> 0
}

case result {
    Ok { value } -> value
    Error { error } -> panic(error)
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
    User { age, .. } if age < 18 -> "minor"
    User { name, .. } -> "user: " <> name
}

case result {
    Ok { value, .. } -> handle_value(value)
    Error { error, .. } -> handle_error(error)
}
```

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
    Var(Text)
    BinOp { op: Text, lhs: Expr, rhs: Expr }
}
```

### 패턴 매칭

```rust
case expr {
    Lit(n) -> n
    Var(name) -> lookup(name)
    BinOp { op: "+", lhs, rhs } -> eval(lhs) + eval(rhs)
    BinOp { op, lhs, rhs } -> apply_op(op, eval(lhs), eval(rhs))
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
| `Variant(value)` | Positional variant 생성 |
| `Name { field }` | Shorthand (변수명 = 필드명) |
| `Name { ..x, field: value }` | Spread (복사 후 일부 변경) |
| `x.field` | 필드 접근 (UFCS, 괄호 생략) |
| `x.field::set(v)` | 필드 설정 |
| `x.field::modify(f)` | 필드 변환 |
| `let Name { field, .. } = x` | Destructuring (나머지 무시) |
