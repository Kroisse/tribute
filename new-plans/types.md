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

The representation is not a source contract. In particular, `Empty` and `Cons`
are not public constructors or patterns. M1 construction uses list literals and
the canonical persistent prepend operation:

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
expose the representation of `tail`. This is the only general source-level List
construction operation required by M1.

M1 exposes sequence-view patterns:

```rust
[]                    // exactly empty
[x, y]                // exactly two elements
[head, ..tail]        // at least one element; tail is a List(a)
[first, second, ..]   // at least two elements; ignore the remainder
```

Exact patterns require the stated length. Prefix-rest patterns require at least
the prefix length and bind the remaining sequence without copying or mutation.
Element subpatterns are matched left to right.

The initial backend representation may be a simpler persistent linked sequence.
Full RRB-tree layout, logarithmic concatenation, and transient/uniqueness
optimization are future work. Those changes must not alter source syntax,
`List(a)` identity, or shared `list.*` IR contracts.

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
list.len       // List::len(list)
list.is_empty  // List::is_empty(list)
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
list.map(fn(x) x + 1)      // OK
list.filter(fn(x) x > 0)   // OK
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
