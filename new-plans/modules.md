# Tribute Module System & Name Resolution

> 이 문서는 design.md를 보완하여 모듈 시스템, 네임스페이스, 이름 해소 규칙을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목 | 선택 | 대안 (채택하지 않음) |
| ---- | ---- | -------------------- |
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
- **Type-directed resolution으로 충분**: 어차피 `option.map(f)`처럼 쓰면 타입으로 해소됨

Gleam도 glob import를 지원하지 않으며, Rust/Haskell 커뮤니티에서도 explicit import가 권장된다.

### 설계 원칙

1. **명시성**: 이름이 어디서 오는지 파일 상단 use만 보면 파악 가능
2. **단순성**: Typeclass 없이도 실용적인 코드 작성 가능
3. **친숙함**: C/Rust/TypeScript 개발자에게 낯설지 않은 문법

### Compiler-owned builtin bindings

파일 기반 module tree가 구성되기 전에도 compiler-owned binding은 resolver namespace에
virtual export로 주입할 수 있다. 현재 `std::io::Io`가 이 경로를 사용한다. 사용자에게는
`use std::io::Io`로 보이지만 사용자 AST나 prelude AST에 synthetic declaration을
추가하지 않으며, semantic identity는 source declaration과 구분된다.

이 경로는 일반 standard-library module loading을 대신하지 않는다. Source로 구현할 수
있는 표준 타입과 함수는 장기적으로 file-based module system을 사용한다. 다만 source
syntax와 backend-independent lowering이 canonical identity를 요구하는 opaque `List`와
ambient/intrinsic 성질처럼 compiler가 보장해야 하는 metadata는 builtin registry에 둔다.

LSP는 import된 compiler-owned binding을 completion과 hover에 노출한다. 다만 이동할
source declaration이 없으므로 go-to-definition과 rename target은 제공하지 않는다.
로컬 declaration은 같은 short name의 builtin import보다 우선하며, 이 규칙은 effect
annotation을 포함한 모든 name-resolution 경로에서 동일하게 적용한다. 이 shadowing은
syntax-owned semantics를 바꾸지 않는다. 예를 들어 로컬 `List`는 annotation에서
선택될 수 있지만 `[1, 2]`의 canonical builtin `List` identity를 capture하지 못하며,
두 타입은 unify되지 않는다.

### Prelude와 패키지 `std`

prelude는 패키지 `std`의 루트 모듈이다. 선언의 식별자는 그 선언이 속한 패키지와
모듈 경로다. 그래서 prelude 선언의 식별자와 IR symbol은 `std::Option`,
`std::Option::map`, `std::Int::+`처럼 `std`로 시작하고, 사용자 패키지의 같은 경로
선언과 섞이지 않는다. prelude 안에서 `pkg`는 `std`를 가리킨다.

모든 모듈은 prelude 루트 항목을 짧은 이름으로 본다. 값과 타입(`Option`, `Some`,
`String`)뿐 아니라 그 항목의 namespace도 경로의 첫 segment가 된다
(`Option::map`, `Int::to_string`, `abilities::Throw`). 사용자 선언이 같은 이름을 가지면
그 선언이 우선하고, prelude 항목은 `std::Option`처럼 패키지 경로로 가리킨다. UFCS는
method index에서 후보를 모으므로 이름이 가려져도 prelude method를 찾는다.

진단은 타입을 식별 경로로 표시한다. prelude 타입은 `std::Option(Int)`처럼 보인다.

---

## Module Syntax

### 모듈 선언과 Use

표준 라이브러리의 canonical root는 `std`다. List의 함수 namespace는
`std::collections::List`이며, public prepend 선언의 canonical path는
`std::collections::List::prepend`다. Prelude는 같은 선언을 가리키는 짧은
`List` binding을 제공한다. `List::prepend`와 완전한 경로는 이 binding을
선택할 때 동일한 선언을 참조하며, 별도 wrapper나 선언 복제를 만들지 않는다.
Compiler-owned List 타입도 같은 조회 경로에서 사용할 수 있지만 source 타입으로
재선언하지 않는다.

Canonical declaration path와 scope 안의 조회 이름은 구분한다. 사용자 `mod List`는
짧은 prelude binding을 shadow할 수 있지만 표준 선언의 canonical path나 이미
해석된 prelude 내부 참조를 바꾸지 않는다. Import alias는 참조하는 선언의
identity를 바꾸지 않으며, 이후 단계는 이름 해석이 선택한 선언을 보존한다.

```rust
// List(a)는 compiler-owned opaque nominal type이다.
// Empty/Cons constructor나 backend layout은 이 namespace에 export하지 않는다.
// Source-visible dynamic construction is specified as a sequence operation.
// std::collections 내부:
pub mod List {
    pub fn prepend(value: a, tail: List(a)) -> List(a) { ... }
}
```

현재 List public surface는 literal, sequence-view pattern,
`List::prepend(value, tail) -> List(a)`로 제한된다. Prelude의 public wrapper는
registry-verified private compiler intrinsic을 호출하고, shared pipeline은 그
private call만 `list.prepend`로 lower한다. 같은 source-visible 조회 경로를 가진 일반
source function은 intrinsic이 아니다. Private intrinsic declaration은 별도 source
API가 아니다.
Compiler lowering에 필요한 `list.empty`, `list.prepend`, `list.is_empty`,
`list.head`, `list.tail`도 shared IR operation이며 source API가 아니다. 이
operation들은 sequence 의미만 갖고 target layout이나 variant tag를 노출하지
않는다. Native와 Wasm은 각자 private layout으로 lower한다.

`extern "intrinsic"`은 compiler-reserved directive다. Prelude와 사용자 선언 모두
선언의 패키지 경로를 intrinsic identity로 요청하며(`std::Int::+`), 사용자 패키지의
선언은 `std`가 가진 identity를 요청할 수 없다. 병합한 AST 전체에서
monomorphization 전에 지원되는 identity인지 검사한다. 미지원 directive는 미사용
선언이라도 각 source span에서 진단한다. 지원되는 identity와 typechecked complete
signature가 registry metadata와 함께 검증된 선언만 intrinsic lowering에 진입한다.
Ordinary source function이나 `extern "C"`는 같은 symbol을 사용해도 intrinsic이 아니며,
reserved ABI 문자열만으로 signature 검증을 우회할 수 없다. 이 directive는 일반적인
symbol uniqueness 규칙을 완화하지 않는다.

`extern "C"` 함수의 IR symbol은 선언한 이름이다. C linkage는 하나의 평평한
이름공간이므로, 모듈 안에서 선언해도 모듈 경로를 붙이지 않는다. 소스에서는 여전히
모듈 경로로 그 함수를 가리킨다.

### Use 문법

```rust
// 모듈 use
use std::collections::List

// 여러 모듈 use
use std::collections::{List, Option, Result}

// 별칭
use std::collections::List as L

// 특정 함수만 use (선택적)
use std::collections::List::prepend
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
enum Option(a) { None, Some(a) }
let option = Option::None  // 값 `option`과 타입 `Option`은 다른 네임스페이스
```

---

## UFCS (Uniform Function Call Syntax)

`.` 연산자는 UFCS를 위해 사용된다. `x.f(y, z)`는 `f(x, y, z)`로 해석된다.

### 괄호 생략

인자가 없는 함수는 괄호를 생략할 수 있다:

```rust
// 괄호 생략 가능
user.name          // User::name(user)

// 인자가 있으면 괄호 필수
option.map(fn(x) x + 1)
string.split(",")
```

이 규칙 덕분에 struct 필드 접근과 함수 호출이 동일한 문법을 사용한다:

```rust
struct User { name: String, age: Int }

// 필드 접근도 UFCS (자동 생성된 getter 함수)
user.name    // User::name(user)
user.age     // User::age(user)
```

### 기본 사용

```rust
use std::collections::Option

fn increment(opt: Option(Int)) -> Option(Int) {
    opt.map(fn(x) x + 1)
    // 위는 아래와 동일:
    // Option::map(opt, fn(x) x + 1)
}
```

### Pipe Operator를 지원하지 않는 이유

UFCS와 pipe는 같은 문제(함수 체이닝)를 해결한다. 둘 다 지원하면:

- 학습할 문법 증가
- 스타일 불일치 논쟁
- 도구/파서 복잡도 증가

`::`을 모듈 구분자로 사용하므로 `.`이 UFCS 전용으로 남아, Gleam처럼 pipe에 의존할 필요가 없다.

### UFCS 체이닝

다음 `List::filter`, `List::map`, `List::fold`는 해소 방식을 보여 주는
illustrative API이며 별도 public API 절에서 확정되기 전까지 예시에 불과하다.

```rust
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

함수 이름이 여러 함수를 가리킬 수 있으면 호출의 인자와 결과 타입으로 해소한다.
이 절의 `List::map`은 해소 규칙만 설명하는 illustrative API이며 별도 public API
절에서 확정되지 않았다.

```rust
use std::collections::List
use std::collections::Option

fn example(xs: List(Int), opt: Option(String)) {
    xs.map(fn(x) x + 1)     // List::map 선택 (xs: List)
    opt.map(fn(s) s.len)    // Option::map 선택 (opt: Option)
}
```

### Resolution 규칙

1. **Qualified name**: `List::map(xs, f)` — 항상 명시적으로 지정된 함수 사용
2. **UFCS**: `xs.map(f)` — 그 이름의 함수 중 호출의 타입에 맞는 함수 검색.
   Qualified UFCS `x.a::b(y)`는 스코프의 `a::b`와 스코프 안 모듈 `m`의 `m::a::b`를
   후보로 같은 규칙을 따른다([syntax.md](syntax.md#call-and-ufcs)).
3. **Unqualified**: `map(xs, f)` — use된 모듈 중 타입이 맞는 함수 검색

비한정 이름이 고를 후보는 그 이름으로 `use`한 함수들이다. 한 스코프의 여러
`use`가 같은 이름에 서로 다른 함수를 주면 그 이름은 그 함수들을 모두 가리킨다.
`f(x, y)`와 `x.f(y)`는 후보를 모으는 범위만 다르다. 그중 하나를 고르는 방식과,
고른 뒤 인자를 검사하고 인자 수와 타입의 오류를 보고하는 방식은 같다.

### 후보 선택

호출은 후보 중에서 자신의 타입에 맞는 함수를 고른다. 후보가 맞으려면:

- 매개변수의 수가 인자의 수와 같고 (UFCS의 receiver는 첫 번째 인자다),
- 각 매개변수의 타입이 그 자리 인자의 타입과 맞으며,
- 결과 타입이 호출의 결과에 기대되는 타입과 맞아야 한다.

두 타입은 같은 nominal 선언이거나 같은 primitive이면 맞는다. 함수 타입과 tuple은
항목 수가 같으면 맞는다. 후보의 타입 변수는 어떤 타입과도 맞는다. 아직 정해지지
않은 인자나 결과의 타입은 어느 후보도 배제하지 않는다. Effect row는 선택에 쓰지
않는다.

후보가 하나뿐이면 고를 것이 없다. 호출은 그 함수로 검사하고, 맞지 않는 인자는
보통의 타입 오류다. 후보가 여럿이고 맞는 후보가 정확히 하나이면 그 함수를 고른다. 그 뒤의 검사는 고른 함수의
선언으로 하므로, 맞는다고 본 타입의 세부(generic 인자, 함수의 매개변수)가 다르면
보통의 타입 오류가 된다. 맞는 후보가 여럿이면 타입이 더 정해질 때까지 기다리고,
함수 본문을 다 푼 뒤에도 여럿이면 오류다. 맞는 후보가 없어도 오류다. 다만 인자의
타입은 맞고 수만 다른 후보가 하나이면 그 함수의 인자 수 오류로 보고한다.

Receiver는 첫 번째 인자이며 다른 인자와 다르지 않다. Receiver의 타입이 정해지지
않았어도 나머지 인자와 결과가 후보를 하나로 좁히면 그 함수를 고르고, receiver의
타입은 고른 함수에서 추론된다.

필드를 읽는 `x.f`도 호출이다. Struct `T`의 필드 `f`는 getter 함수 `T::f`를 만들고,
`x.f`는 `f(x)`이다. 비한정 UFCS의 후보에는 그 이름의 함수들과 함께, 그 이름의
필드를 가진 모든 struct의 getter가 들어간다. 따라서 필드 `f`를 가진 struct가
하나뿐이고 같은 이름의 함수가 없으면 `x.f`는 `x`의 타입 없이도 정해진다.

Getter `T::f`는 `T`의 네임스페이스에 선언된 함수다. 한 네임스페이스에서 한 이름은
한 함수이므로, `T`의 companion 모듈은 필드와 같은 이름의 함수를 매개변수와 상관없이
선언할 수 없다. 다른 모듈은 같은 이름의 함수를 선언할 수 있다. 그 함수가 `T`를
첫 번째 인자로 받으면 `x.f`의 후보는 getter와 그 함수 둘이고, 다른 호출과 같은
규칙으로 고른다. 둘 다 맞으면 모호하며 `x.T::f`처럼 경로로 구분한다:

```rust
struct Name { text: String }

pub mod Name {
    pub fn text(name: Name, suffix: String) -> String { ... }  // Error: Name::text는 필드의 getter
}

mod audit {
    pub fn text(name: Name) -> String { ... }                  // OK: audit::text
}

fn show(name: Name) -> String {
    name.text           // Error: Name::text와 audit::text 모두 맞음
    name.Name::text     // OK
    name.audit::text    // OK
}
```

```rust
use a::pair   // fn pair(x: A, y: Int) -> Int
use b::pair   // fn pair(x: A, y: String) -> Int

pair(A { n: +1 }, +2)       // a::pair: 두 번째 인자가 Int
pair(A { n: +1 }, "two")    // b::pair
A { n: +1 }.pair("two")     // b::pair
fn(x) { x.pair("two") }     // b::pair: x는 A로 추론된다
```

그 스코프가 같은 이름을 선언하거나 지역 변수로 묶으면 이름은 그 선언이나 변수를
가리키며, `use`한 함수들은 후보가 아니다. 같은 함수를 여러 번 `use`하면 후보는
하나다.

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

여러 함수를 `use`한 이름은 호출해야 쓸 수 있다. 호출하지 않고 값으로 쓰면
오류이며, 함수를 경로로 지정해야 한다. 인자 없는 호출도 같다:

```rust
use a::size   // fn size(x: A) -> Nat
use b::size   // fn size(x: B) -> Nat
use a::make   // fn make() -> Nat
use b::make

size(A { n: 1 })    // OK: a::size
size(B { n: 2 })    // OK: b::size
size(1)             // Error: 어느 size도 Nat 하나를 받지 않는다
make()              // Error: 고를 인자가 없다. a::make()로 지정
apply(x, size)      // Error: 값으로 쓰면 고를 인자가 없다. a::size로 지정
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
sort(my_list, String::compare)

// 또는 특화된 함수 제공
fn sort_by(xs: List(a), key: fn(a) -> k, compare: fn(k, k) -> Ordering) -> List(a)
```

### 흔한 패턴들의 대체

| Typeclass 용도 | Tribute 대안 |
| -------------- | ------------ |
| `Show` | `fn show(x: T) -> String`을 명시적 전달, 또는 type-directed resolution |
| `Eq` | `fn eq(a: T, b: T) -> Bool` 명시적 전달 |
| `Ord` | `fn compare(a: T, b: T) -> Ordering` 명시적 전달 |
| `Functor`/`Monad` | Ability system + type-directed `map`, `flat_map` |
| `Numeric` literals | 타입 어노테이션 또는 suffix (`42i`, `1e3f`) |

---

## Complete Example

다음 예제의 `List::of`, `filter`, `map`, `fold`, `sort`, `each`는 전체 module/UFCS
구성을 보여 주기 위한 illustrative API이며 별도 public API 절에서 확정되지 않았다.

```rust
// app/main.trb

use std::collections::{List, Option}
use std::io::{Io, print_line}

struct User {
    name: String
    age: Int
}

fn main() ->{Io} Nil {
    let numbers = List::of(1, 2, 3, 4, 5)

    let result = numbers
        .filter(fn(x) x > 2)
        .map(fn(x) x * 10)
        .fold(0, fn(acc, x) acc + x)

    print_line("Result: " <> Int::to_string(result))
    
    // Record 생성과 업데이트
    let user = User { name: "Alice", age: 30 }
    let older = User { ..user, age: 31 }
    
    print_line("User: " <> older.name)
}

// 명시적 함수 전달 예시
fn sort_and_print(items: List(String)) ->{Io} Nil {
    let sorted = List::sort(items, String::compare)
    sorted.each(fn(item) {
        print_line(item)
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
| ---- | ---- |
| `lib.trb` | 라이브러리 패키지의 루트 |
| `main.trb` | 실행 파일 패키지의 루트 |

```text
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

pub enum Option(a) { ... }   // PascalCase (타입)
pub mod Option { ... }       // 타입과 동명이므로 PascalCase
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
| ------ | ---- | ---- |
| `pkg` | 현재 패키지 루트 | `use pkg::utils::math` |
| `super` | 부모 모듈 | `use super::sibling` |
| `self` | 현재 모듈 | `use self::internal` |

```rust
// src/utils/math.trb
use super::string::format    // utils::string::format
use pkg::api::Response       // api::Response (패키지 루트에서)
```

경로 키워드는 경로의 첫 segment에만 올 수 있다. 키워드로 시작하는 경로는 패키지
루트 기준의 경로 하나를 가리킨다. `pkg`는 루트, `self`는 현재 모듈, `super`는 현재
모듈의 부모다. 현재 모듈은 경로가 쓰인 inline 모듈이거나, 그 경로를 담은 파일의
모듈이다. 이 경로는 현재 모듈을 기준으로 다시 해석하지 않는다. 두 단계 위의 모듈은
`pkg`로 시작하는 경로로 가리킨다.

이 규칙은 `use` 경로, 값 경로(호출, 생성자, 패턴, handler arm의 ability), 타입과
effect annotation에 똑같이 적용한다. 패키지 루트에서 쓴 `super`, 그리고 첫
segment가 아닌 위치의 키워드는 이름 해석 오류다.

### 인라인 모듈의 이름 범위

인라인 모듈은 파일 기반 모듈과 같은 범위를 갖는다. 모듈 안에서 보이는 이름은 그
모듈의 선언, 그 모듈의 `use`, prelude뿐이다. 감싼 모듈이나 패키지 루트의 항목은
자동으로 보이지 않으며, `use super::x`나 `pkg::x`로 가져온다. 그래서 인라인 모듈을
파일로 옮겨도 이름의 뜻이 바뀌지 않는다.

경로의 첫 segment도 같은 범위에서 찾는다. 그 모듈의 하위 모듈이나 선언, 그 모듈이
`use`로 가져온 이름, prelude가 제공하는 namespace(`Option`, `std` 등)가 대상이다.
`use` 경로도 같다. 패키지 루트는 자신의 선언과 `use`, prelude를 본다.

예외로, 같은 이름의 타입이나 ability 옆에 선언된 companion 모듈은 그 이름으로 자기
타입을 본다. 경로에는 예외가 없다. 모듈 안에서 모듈 자신의 이름은 경로를 시작하지
않으며, 모듈의 항목과 생성자는 이름만으로 쓴다. 그래서
[타입과 동명 네임스페이스](#타입과-동명-네임스페이스)의 companion 모듈은 자기 타입과
생성자를 그대로 쓴다.

```rust
ability State(s) { op get() -> s }

enum Shape { Circle(Nat) }

mod Shape {
    pub fn radius(shape: Shape) -> Nat {   // companion: 옆의 enum Shape
        case shape { Circle(r) -> r }
    }
}

mod counter {
    use super::State                       // 감싼 모듈의 항목은 가져온다

    pub fn read() ->{State(Int)} Int { State::get() }
}
```

prelude도 같은 규칙을 따른다. prelude의 모듈은 다른 prelude 항목을 `use super::…`나
`pkg::…`로 가져온다.

### 가시성 (Visibility)

| 수식자 | 범위 |
| ------ | ---- |
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
    path: String
    config: Config
}

pub struct Response {
    status: Int
    body: String
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
