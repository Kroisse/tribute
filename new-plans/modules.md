# Tribute Module System & Name Resolution

> 이 문서는 design.md를 보완하여 모듈 시스템, 네임스페이스, 이름 해소 규칙을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목 | 선택 | 대안 (채택하지 않음) |
|------|------|----------------------|
| 모듈 구분자 | `::` | `.` (UFCS와 충돌), `/` (나눗셈과 모호), `:` (타입 어노테이션과 충돌) |
| 메서드 호출 스타일 | UFCS (`.`) | Pipe (`\|>`) |
| 타입 선언 | `struct` / `enum` | 단일 `type` 키워드 |
| Ad-hoc polymorphism | 없음 (명시적 전달) | Typeclass, Trait, Implicits |
| 이름 해소 | Type-directed (use 범위 내) | 전역 suffix resolution (Unison) |
| Glob use | 미지원 | `use std::collections::*` |

### Glob Use 미지원 이유

`use std::collections::*` 같은 glob use를 지원하지 않는다:

- **명시성 저하**: 어떤 이름이 어디서 왔는지 파일만 보고 알 수 없음
- **취약한 의존성**: use한 모듈에 새 함수가 추가되면 기존 코드와 이름 충돌 가능
- **Type-directed resolution으로 충분**: 어차피 `xs.map(f)`처럼 쓰면 타입으로 해소됨

Gleam도 glob import를 지원하지 않으며, Rust/Haskell 커뮤니티에서도 explicit import가 권장된다.

### 설계 원칙

1. **명시성**: 이름이 어디서 오는지 파일 상단 use만 보면 파악 가능
2. **단순성**: Typeclass 없이도 실용적인 코드 작성 가능
3. **친숙함**: C/Rust/TypeScript 개발자에게 낯설지 않은 문법

---

## Module Syntax

### 모듈 선언과 Use

```rust
// std/collections/list.trb

pub enum List(a) {
    Empty
    Cons(a, List(a))
}

pub mod List {
    pub fn empty() -> List(a) {
        Empty
    }

    pub fn singleton(x: a) -> List(a) {
        Cons(x, Empty)
    }

    pub fn map(xs: List(a), f: fn(a) -> b) -> List(b) {
        case xs {
            Empty -> Empty
            Cons(h, t) -> Cons(f(h), map(t, f))
        }
    }

    pub fn filter(xs: List(a), p: fn(a) -> Bool) -> List(a) {
        case xs {
            Empty -> Empty
            Cons(h, t) -> case p(h) {
                True -> Cons(h, filter(t, p))
                False -> filter(t, p)
            }
        }
    }
}
```

### Use 문법

```rust
// 모듈 use
use std::collections::List

// 여러 모듈 use
use std::collections::{List, Option, Result}

// 별칭
use std::collections::List as L

// 특정 함수만 use (선택적)
use std::collections::List::{map, filter}
```

---

## Namespace Rules

### 타입과 동명 네임스페이스

타입 선언(`struct`, `enum`)은 암묵적으로 동명의 네임스페이스를 생성한다. 생성자는 자동으로 해당 네임스페이스에 포함된다.

```rust
pub enum Option(a) {
    None
    Some(a)
}

// 위 선언은 아래를 암묵적으로 생성:
// - Option::None: Option(a)
// - Option::Some: fn(a) -> Option(a)
```

`pub mod` 블록으로 관련 함수를 같은 네임스페이스에 추가할 수 있다:

```rust
pub mod Option {
    pub fn map(opt: Option(a), f: fn(a) -> b) -> Option(b) {
        case opt {
            None -> None
            Some(x) -> Some(f(x))
        }
    }

    pub fn unwrap_or(opt: Option(a), default: a) -> a {
        case opt {
            None -> default
            Some(x) -> x
        }
    }
}
```

### 네임스페이스 분리

타입/생성자 네임스페이스와 값 네임스페이스는 분리된다 (Unison 스타일):

```rust
// OK: 같은 이름이 타입과 값으로 공존 가능
enum List(a) { ... }
let list = List::empty()  // 값 `list`와 타입 `List`는 다른 네임스페이스
```

---

## UFCS (Uniform Function Call Syntax)

`.` 연산자는 UFCS를 위해 사용된다. `x.f(y, z)`는 `f(x, y, z)`로 해석된다.

### 괄호 생략

인자가 없는 함수는 괄호를 생략할 수 있다:

```rust
// 괄호 생략 가능
list.len           // List::len(list)
list.is_empty      // List::is_empty(list)
option.is_some     // Option::is_some(option)

// 인자가 있으면 괄호 필수
list.map(fn(x) x + 1)
string.split(",")
```

이 규칙 덕분에 struct 필드 접근과 함수 호출이 동일한 문법을 사용한다:

```rust
struct User { name: Text, age: Int }

// 필드 접근도 UFCS (자동 생성된 getter 함수)
user.name    // User::name(user)
user.age     // User::age(user)
```

### 기본 사용

```rust
use std::collections::List

fn process(xs: List(Int)) -> List(Int) {
    xs.map(fn(x) x * 2).filter(fn(x) x > 0)
    // 위는 아래와 동일:
    // List::filter(List::map(xs, fn(x) x * 2), fn(x) x > 0)
}
```

### Pipe Operator를 지원하지 않는 이유

UFCS와 pipe는 같은 문제(함수 체이닝)를 해결한다. 둘 다 지원하면:

- 학습할 문법 증가
- 스타일 불일치 논쟁
- 도구/파서 복잡도 증가

`::`을 모듈 구분자로 사용하므로 `.`이 UFCS 전용으로 남아, Gleam처럼 pipe에 의존할 필요가 없다.

### 기존 예제 업데이트

```rust
// Before (pipe 스타일, 더 이상 사용하지 않음)
fn process(data: List(Int)) -> Int {
    data
    |> list.filter(fn(x) x > 0)
    |> list.map(fn(x) x * 2)
    |> list.fold(0, fn(a, b) a + b)
}

// After (UFCS 스타일)
fn process(data: List(Int)) -> Int {
    data
        .filter(fn(x) x > 0)
        .map(fn(x) x * 2)
        .fold(0, fn(a, b) a + b)
}
```

---

## Name Resolution

### Type-Directed Resolution

함수 이름이 여러 모듈에서 use된 경우, 첫 번째 인자의 타입으로 해소한다.

```rust
use std::collections::List
use std::collections::Option

fn example(xs: List(Int), opt: Option(Text)) {
    xs.map(fn(x) x + 1)     // List::map 선택 (xs: List)
    opt.map(fn(s) s.len)    // Option::map 선택 (opt: Option)
}
```

### Resolution 규칙

1. **Qualified name**: `List::map(xs, f)` — 항상 명시적으로 지정된 함수 사용
2. **UFCS**: `xs.map(f)` — 첫 번째 인자 타입에 맞는 함수 검색
3. **Unqualified**: `map(xs, f)` — use된 모듈 중 타입이 맞는 함수 검색

### 모호성 처리

```rust
use std::collections::List
use some::other::List as OtherList  // 다른 List 타입

fn ambiguous(xs: List(Int), ys: OtherList(Int)) {
    xs.map(fn(x) x + 1)  // OK: std::collections::List::map
    ys.map(fn(x) x + 1)  // OK: some::other::List::map

    // 만약 타입으로 해소할 수 없으면 컴파일 에러 + 명시적 지정 요구
}
```

### Use 범위 제한

Unison과 달리, name resolution은 **use된 모듈 범위 내에서만** 동작한다:

```rust
// 이렇게 하면 안 됨: 전체 codebase 검색
fn bad_example(xs: List(Int)) {
    xs.some_function(...)  // 에러: some_function을 찾을 수 없음
}

// 이렇게 해야 함: 명시적 use
use std::collections::List
use some::module  // some_function이 정의된 모듈

fn good_example(xs: List(Int)) {
    xs.some_function(...)  // OK: use된 모듈에서 검색
}
```

이 제한으로 인해:
- 파일 상단 use만 보면 의존성 파악 가능
- 에러 메시지가 명확함
- IDE 자동완성이 빠름

---

## No Typeclass / Trait

Tribute는 typeclass나 trait을 지원하지 않는다.

### 이유

1. **Algebraic effects가 대부분의 use case 해결**: `Monad`, `MonadIO` 등이 불필요
2. **복잡도 감소**: coherence, orphan rule, overlapping instances 등 고려 불필요
3. **명시성 향상**: "마법" 감소, 코드 읽기 쉬움
4. **실용적 충분성**: Gleam, Unison 경험상 typeclass 없이도 충분

### 대안 패턴

```rust
// Typeclass 방식 (Haskell)
sort :: Ord a => [a] -> [a]
sort myList

// Tribute 방식: 명시적 함수 전달
fn sort(xs: List(a), compare: fn(a, a) -> Ordering) -> List(a) { ... }

// 사용
sort(my_list, Int::compare)
sort(my_list, Text::compare)

// 또는 특화된 함수 제공
fn sort_by(xs: List(a), key: fn(a) -> k, compare: fn(k, k) -> Ordering) -> List(a)
```

### 흔한 패턴들의 대체

| Typeclass 용도 | Tribute 대안 |
|---------------|--------------|
| `Show` | `fn show(x: T) -> Text`을 명시적 전달, 또는 type-directed resolution |
| `Eq` | `fn eq(a: T, b: T) -> Bool` 명시적 전달 |
| `Ord` | `fn compare(a: T, b: T) -> Ordering` 명시적 전달 |
| `Functor`/`Monad` | Ability system + type-directed `map`, `flat_map` |
| `Numeric` literals | 타입 어노테이션 또는 suffix (`42i64`, `3.14f32`) |

---

## Complete Example

```rust
// app/main.trb

use std::collections::{List, Option}
use std::io::Console

struct User {
    name: Text
    age: Int
}

fn main() ->{Console} Nil {
    let numbers = List::of(1, 2, 3, 4, 5)

    let result = numbers
        .filter(fn(x) x > 2)
        .map(fn(x) x * 10)
        .fold(0, fn(acc, x) acc + x)

    Console::println("Result: " <> Int::to_string(result))
    
    // Record 생성과 업데이트
    let user = User { name: "Alice", age: 30 }
    let older = User { ..user, age: 31 }
    
    Console::println("User: " <> older.name)
}

// 명시적 함수 전달 예시
fn sort_and_print(items: List(Text)) ->{Console} Nil {
    let sorted = List::sort(items, Text::compare)
    sorted.each(fn(item) {
        Console::println(item)
    })
}
```

---

## File-Based Modules (Package System)

### Package 개념

**Package**는 Tribute의 컴파일 단위다:

- 파일명이 모듈명이 됨 (`foo.trb` → `foo`)
- 하위 모듈은 동명 디렉토리에 위치 (Rust 2018 스타일)

### 최상위 모듈 (Root Module)

패키지의 최상위 모듈은 관례적으로:

| 파일 | 용도 |
|------|------|
| `lib.trb` | 라이브러리 패키지의 루트 |
| `main.trb` | 실행 파일 패키지의 루트 |

```
my_library/
  src/
    lib.trb           // 라이브러리 루트 (pkg::)
    utils.trb         // pkg::utils
    utils/
      math.trb        // pkg::utils::math

my_app/
  src/
    main.trb          // 실행 파일 루트
    config.trb        // pkg::config
```

`lib.trb`와 `main.trb`가 동시에 존재하면 라이브러리와 실행 파일을 모두 제공하는 패키지가 된다.

### 모듈 이름 관례

- **기본**: 소문자 (`math`, `utils`, `api`)
- **타입과 동명일 때**: PascalCase (`List`, `Option`) — 타입이 암묵적으로 동명 네임스페이스 생성

```rust
mod utils              // 소문자 (일반 모듈)
mod api                // 소문자

pub enum List(a) { ... }   // PascalCase (타입)
pub mod List { ... }       // 타입과 동명이므로 PascalCase
```

### 모듈 선언

파일 기반 모듈은 명시적 `mod` 선언이 필요하다:

```rust
// src/lib.trb
mod utils           // src/utils.trb 로드 (private)
pub mod api         // src/api.trb 로드 (public)

// src/utils.trb
pub mod math        // src/utils/math.trb 로드
pub mod string      // src/utils/string.trb 로드
```

인라인 모듈과 파일 기반 모듈의 차이:

```rust
// 인라인 모듈 (본문 있음)
pub mod helpers {
    pub fn double(x: Int) -> Int { x * 2 }
}

// 파일 기반 모듈 (본문 없음 → 파일에서 로드)
mod utils
pub mod api
```

### 경로 키워드

| 키워드 | 설명 | 예시 |
|--------|------|------|
| `pkg` | 현재 패키지 루트 | `use pkg::utils::math` |
| `super` | 부모 모듈 | `use super::sibling` |
| `self` | 현재 모듈 | `use self::internal` |

```rust
// src/utils/math.trb
use super::string::format    // utils::string::format
use pkg::api::Response       // api::Response (패키지 루트에서)
```

### 가시성 (Visibility)

| 수식자 | 범위 |
|--------|------|
| (없음) | 현재 모듈 내부만 |
| `pub(super)` | 부모 모듈까지 |
| `pub(pkg)` | 패키지 내부 전체 |
| `pub` | 공개 (외부 패키지에서도 접근 가능) |

```rust
// src/internal/utils.trb

fn private_helper() -> Int { 42 }           // 이 모듈에서만

pub(super) fn parent_visible() -> Int { 42 } // internal/ 및 하위에서

pub(pkg) fn package_internal() -> Int { 42 } // 이 패키지 내에서

pub fn public_api() -> Int { 42 }            // 어디서든
```

### Re-export

```rust
// 내부 모듈의 타입을 공개 API로 노출
pub use pkg::internal::PublicType
pub use pkg::internal::{TypeA, TypeB}

// 별칭으로 re-export
pub use pkg::internal::LongTypeName as Short
```

### 완전한 예시

```rust
// src/lib.trb
mod internal           // private 모듈
pub mod api            // public 모듈

// 주요 타입들을 루트에서 re-export
pub use pkg::internal::Config
pub use pkg::api::{Request, Response}


// src/internal/mod.trb (또는 src/internal.trb)
pub(pkg) struct Config {
    debug: Bool
    timeout: Int
}


// src/api.trb
use pkg::internal::Config
use super::internal    // 또는 pkg::internal

pub struct Request {
    path: Text
    config: Config
}

pub struct Response {
    status: Int
    body: Text
}

pub fn handle(req: Request) -> Response {
    // ...
}
```

---

## Open Questions

1. **Prelude**: 자동 use되는 기본 타입/함수 범위
2. **조건부 컴파일**: `#[cfg(...)] mod foo` 문법 및 지원 범위
3. **매니페스트**: `Tribute.toml` 형식 및 필수 여부
4. **lang-examples 구조**: 각 예제를 개별 폴더의 `main.trb`로 재구성
