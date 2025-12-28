# Tribute Type System

> 이 문서는 Tribute의 타입 선언 문법, record, UFCS 규칙을 정의한다.

## Type Declarations

Tribute는 `struct`와 `enum` 두 키워드로 타입을 선언한다.

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
    True
    False
}

// 제네릭 enum - positional fields
enum Option(a) {
    None
    Some(a)
}

enum List(a) {
    Empty
    Cons(a, List(a))
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
let cons = Cons(1, Cons(2, Empty))

// Named
let ok = Ok { value: 42 }
let error = Error { error: "something went wrong" }
```

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
    Var(String)
    BinOp { op: String, lhs: Expr, rhs: Expr }
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
|------|------|
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
