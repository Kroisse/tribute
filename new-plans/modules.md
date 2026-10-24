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
virtual export로 주입할 수 있다. 현재 `std::io::Io`와 runtime 전용 타입
`std::runtime::Slots`가 이 경로를 사용한다. 사용자에게는
`use std::io::Io`로 보이지만 사용자 AST나 표준 라이브러리 AST에 synthetic
declaration을 추가하지 않으며, semantic identity는 source declaration과 구분된다.
Virtual export는 그 binding을 내보내는 패키지(`std`)의 이름공간에 속하며, 일반
항목과 같은 [가시성](#가시성-visibility) 규칙을 따른다. 다른 패키지는 `pub`인 virtual
export(`std::io::Io`)만 [의존 패키지](#prelude와-패키지-std) 경로나 prelude 짧은
이름으로 본다. `std::runtime`의 virtual export(`Slots(a)`, `Dispatch`)는 그 모듈이
`pub(pkg)`이므로 패키지 `std` 밖에서 보이지 않는다.

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

**패키지의 루트.** 각 패키지는 자기 루트 모듈에서 해석한다. 모듈 경로는 그 패키지의
루트를 기준으로 하며, `pkg`는 언제나 경로가 쓰인 패키지의 루트다. 패키지 `std`의 루트
모듈은 표준 라이브러리 source이고, 그 안에서 `pkg`는 `std`의 루트를 가리킨다. 패키지
루트의 이름 범위와 경로 규칙은 모든 패키지에 같다.

**선언의 식별자.** 선언의 식별자는 그 선언이 속한 패키지와 모듈 경로다. 식별 경로는
이 둘로부터 한 곳에서 만든다. 식별자는 텍스트 하나가 아니라 component의 path다.
둘러싼 모듈의 이름들이 바깥쪽부터 차례로 오고 선언의 이름이 마지막에 온다.
`std::Int::+`는 component `std`, `Int`, `+`의 path다. Component가 없는 path는 루트
모듈을 가리킨다. 한정된 참조는 표기한 텍스트를 `::`로 다시 나누지 않고 component
단위로 해석하며, component 하나의 이름에 `::`가 들어 있어도 그 이름은 component
하나다.

식별 경로와 IR symbol의 철자는 다음과 같다.

- 의존 패키지의 선언은 패키지 이름으로 시작한다. 패키지 `std`의 선언은
  `std::Option`, `std::Option::map`, `std::Int::+`다.
- 컴파일하는 패키지의 선언은 자기 패키지 component를 생략하고 모듈 경로로
  시작한다(`main`, `geometry::area`). 이 생략은 컴파일하는 패키지에만 적용한다.
  패키지 `std`는 언제나 의존 패키지로 해석하므로, `std` 선언의 식별 경로와 IR
  symbol, runtime 항목 경로, `tribute.types` key는 언제나 `std::…`다.
- 생략한 경로의 첫 component가 의존 패키지의 이름이면 생략하지 않고 `pkg`로
  시작한다. 사용자 패키지의 `mod std { pub fn f() … }`의 `f`는 `pkg::std::f`이고,
  패키지 `std`의 선언과 식별자나 symbol을 공유하지 않는다. `pkg`는 경로 키워드이므로
  어떤 모듈의 이름도 될 수 없어 이 경로는 다른 선언과 겹치지 않는다.

**의존 패키지.** 패키지는 의존하는 패키지를 의존성 표로 안다. 모든 패키지는 `std`에
의존한다. 의존 패키지의 이름은 모든 모듈에서 경로의 첫 segment가 될 수 있고, 그 경로는
의존 패키지의 루트에서 이어서 해석한다(`std::io::print_line`). 의존 패키지의 항목은
그 패키지의 가시성 규칙을 따르므로 `pub` 항목만 보인다.

- 값 경로와 타입·ability annotation에서는 모듈 안의 이름(그 모듈의 선언과 `use`)이
  의존 패키지 이름보다 우선한다. 사용자의 `mod std`가 있으면 `std::f`는 그 모듈의
  `f`다.
- `use` 경로의 첫 segment가 모듈 안의 이름과 의존 패키지 이름 둘 다에 해당하면 이름
  해석 오류다. 모듈 쪽은 `self::std`나 `pkg::std`로, 패키지 쪽은 `::std`로 쓴다.
- 선행 `::`로 시작하는 경로(`::std::Option`)는 의존 패키지만 가리킨다. 모듈 안의
  이름이 패키지 이름을 가려도 이 경로로 패키지에 닿는다. 의존성 표에 없는 이름이면
  이름 해석 오류다.

**Prelude.** prelude는 패키지 `std`의 모듈 `std::prelude::v0`다. 모든 패키지의 모든
모듈은 패키지 `std` 자신의 모듈을 포함해 `std::prelude::v0`의 `pub` 항목을 짧은
이름으로 본다. 그 항목은 대개 표준 라이브러리 선언의 re-export다. 타입(`Option`,
`String`)뿐 아니라 그 항목의 namespace도 경로의 첫 segment가 된다
(`Option::map`, `Option::Some`, `Int::to_string`). Re-export는 대상의 namespace를
함께 가져오기 때문이다.

- 모듈 안의 이름이 prelude 짧은 이름보다 우선한다. 가려진 prelude 항목은
  `std::Option`이나 `::std::Option`처럼 패키지 경로로 가리킨다.
- 가려진 prelude 함수와 prelude 타입의 method는 호출의 마지막 순위 후보로
  남는다([후보의 순위](#후보의-순위)). 사용자의 `Option`이 있어도 `std::Option`
  값의 `x.map(f)`는 타입이 맞는 `std::Option::map`을 고른다. 경로로 지정하려면
  `x.std::Option::map(f)`로 쓴다.
- `std::prelude::v0`의 `use`는 prelude 짧은 이름으로 시작하지 않고 경로 키워드로
  쓴다(`pub use pkg::Option`). Prelude가 자기 자신에 기대지 않게 하기 위해서다.
- Prelude에 둘 이름은 사용자 코드가 짧은 이름으로 쓰는 표준 타입과, 타입과 같은
  이름의 namespace(`Int`, `Bytes`)다. Compiler 소유 `List`의 짧은 이름도 prelude의
  re-export다. 같은 선언을 가리키므로 식별 경로는 바뀌지 않는다.
- Enum variant는 prelude에 두지 않는다. Variant는 enum의 namespace에만 있으므로
  ([types.md](types.md#enum-variant의-이름)), prelude 타입의 variant는
  `Option::Some`이나 [암묵적 생성자](type-inference.md#암묵적-생성자) `.Some`으로
  가리킨다. 그래서 `Result::Error`와 `std::io::Error`처럼 이름이 같은 variant와
  타입이 prelude에서 충돌하지 않는다.
- 표준 라이브러리의 모듈(`std::abilities`, `std::io`, `std::collections`)은
  prelude에 두지 않는다. 그 항목은 `std::io::print_line`처럼 패키지 경로로
  가리키거나 `use`로 가져온다.

진단은 타입을 식별 경로로 표시한다. 패키지 `std`의 타입은 `std::Option(Int)`처럼
보인다.

### Compiler 소유 runtime 모듈

Compiler가 생성한 code가 부르는 runtime 함수 중 target에 묶이지 않는 것은 Tribute
source로 쓴다. 이 code는 패키지 `std`의 `pub(pkg)` 모듈 `std::runtime`과 그 하위 모듈에
둔다. 예를 들어 evidence runtime은 `std::runtime::evidence`다.
[가시성](#가시성-visibility) 규칙에 따라 패키지 `std` 밖에서는 이 모듈과 그 항목을
이름으로 가리킬 수 없다. 사용자 프로그램은 이 code를 부르지 않는다.

Source로 표현할 수 없는 runtime 전용 타입(`Slots(a)`, `Dispatch`)은
[compiler-owned builtin binding](#compiler-owned-builtin-bindings)으로
`std::runtime` 안에만 virtual export한다. Virtual export도 export된 모듈의
가시성을 따르므로 패키지 `std` 밖에서는 보이지 않는다.

**Runtime 항목.** Lowering이 부르는 runtime 함수와 lowering이 만드는 runtime
타입을 runtime 항목이라 한다. Runtime 항목의 정체성은 그 선언의 canonical 경로다
(`std::runtime::evidence::extend`). 별도의 역할 이름이나 source 표시는 없으며,
`std::runtime` 안에 있다는 것이 compiler 소유 code라는 표시다. Compiler는
lowering이 쓰는 runtime 항목 경로의 집합을 가진다.

- Frontend는 집합의 경로마다 선언이 있고 그 종류(본문이 있는 함수, struct)가
  맞는지 검사한다. 선언이 없거나 종류가 다르면 compiler 오류다. Signature와
  field는 source 선언이 유일한 정의이며, compiler는 이를 따로 적어 두지 않는다.
  항목을 쓰는 lowering이 IR에서 자신이 의존하는 타입을 확인하고, 맞지 않으면
  compiler 오류로 보고한다.
- `std::runtime` 안의 다른 선언은 runtime 항목이 아니다. Runtime 항목이 부르는
  helper일 수 있다.
- Virtual export한 compiler 소유 타입(`Slots(a)`, `Dispatch`)과
  `std::runtime::Slots`의 compiler intrinsic은 runtime 항목이 아니다. 그 IR
  형태는 compiler가 정하므로 `tribute.types`로 게시하지 않으며, intrinsic은
  아래 `extern "intrinsic"` directive 규칙을 따른다.
- Lowering은 runtime 함수를 canonical 경로에서 만든 IR symbol로 찾는다. 패키지
  `std`의 선언 식별자는 사용자 패키지의 선언과 섞이지 않으므로(위
  [Prelude와 패키지 `std`](#prelude와-패키지-std)) 이 symbol은 패키지 `std`의 정의만
  가리킨다. 경로에서 symbol을 만드는 일은 한 곳에서 하며, 찾은 정의가 본문이 있는
  함수가 아니면 compiler 오류다. 이 조회는 `std::runtime`의 runtime 항목에만
  허용된다. 다른 선언의 의미, ownership, 호출 규약을 symbol 철자로 추측하지
  않는다는 규칙은 그대로다.
- Runtime 타입의 IR 표현은 pass를 거치며 바뀌므로 이름으로 찾지 않는다.
  Frontend는 `std::String`처럼 root `core.module`의 dictionary 속성
  `tribute.types`([IR 계약](ir.md)에 정의)에 canonical 경로를 key로 그
  `TypeRef`를 게시하고, lowering은 이 항목을 읽는다.

**제약.** `std::runtime`과 하위 모듈에서 본문이 있는 source 함수는 runtime
항목이든 helper든 다음을 지킨다. 본문이 없는 compiler intrinsic 선언에는 적용하지
않는다. Frontend가 검사한다.

- Ability row가 비어 있어 evidence를 받지 않고 중단되지 않는다. 빈 row는 `->{}`로
  명시한다. Row를 생략한 함수는 row 변수로 일반화되므로 거부한다. Handler를
  설치하거나 operation을 수행하지 않으므로 `effect.*`를 만들지 않으며, 그 함수를
  부르는 lowering이 그 함수 자신을 다시 lowering 대상으로 삼지 않는다.
- 단형(monomorphic)이다. Type parameter를 갖지 않는다.
- 반복은 tail position의 재귀로만 쓴다. `Direct` route의 proper tail call은
  stack을 늘리지 않는다.
- Closure 값은 저장하고 돌려줄 수 있지만 부르지 않는다.

**생존 범위.** Runtime 함수 항목은 아직 아무도 부르지 않아도 그 항목을 쓰는
lowering 단계가 끝날 때까지 dead function 제거의 root다. 그 단계 뒤에는 일반
함수처럼 참조되지 않으면 제거된다. 단, 그 호출을 만들 연산이 생길 수 없는
module에서는 처음부터 root가 아니다. Evidence runtime 함수를 부르는 `effect`
연산은 도달 가능한 `perform`, `handle`, `evidence_plan`에서만 생기므로, 이것이
없는 module은 shared lowering 전에 evidence runtime 함수를 제거한다.
`tribute.types`의 runtime 타입 항목은 타입 식별 metadata로 representation/ABI
경계 출구 뒤에도 남는다.

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
해석된 패키지 `std` 내부 참조를 바꾸지 않는다. Import alias는 참조하는 선언의
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
`List::prepend(value, tail) -> List(a)`로 제한된다. 패키지 `std`의 public wrapper는
registry-verified private compiler intrinsic을 호출하고, shared pipeline은 그
private call만 `list.prepend`로 lower한다. 같은 source-visible 조회 경로를 가진 일반
source function은 intrinsic이 아니다. Private intrinsic declaration은 별도 source
API가 아니다.
Compiler lowering에 필요한 `list.empty`, `list.prepend`, `list.is_empty`,
`list.head`, `list.tail`도 shared IR operation이며 source API가 아니다. 이
operation들은 sequence 의미만 갖고 target layout이나 variant tag를 노출하지
않는다. Native와 Wasm은 각자 private layout으로 lower한다.

`extern "intrinsic"`은 compiler-reserved directive다. 패키지 `std`와 사용자 선언 모두
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

타입 선언(`struct`, `enum`)은 암묵적으로 동명의 네임스페이스를 생성한다. Enum의
variant는 이 네임스페이스에만 선언되며, enum을 둘러싼 모듈에는 이름이 생기지
않는다([types.md](types.md#enum-variant의-이름)).

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
            .None -> .None
            .Some(x) -> .Some(f(x))
        }
    }

    pub fn unwrap_or(opt: Option(a), default: a) -> a {
        case opt {
            .None -> default
            .Some(x) -> x
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

### 중복 선언

한 모듈(패키지 루트, 파일 모듈, 인라인 `mod` 본문) 안에서 `fn`, `extern fn`,
`struct`, `enum`, `ability`는 이름 하나를 한 번만 선언할 수 있다. 같은 이름을 다시
선언하면 나중 선언이 오류이며, 먼저 선언한 것이 그 이름의 정의로 남는다. 오류가 난
선언은 이후 단계에 전달되지 않는다. 타입과 같은 이름의 companion `mod`, `use`,
다른 모듈의 같은 이름은 중복이 아니다.

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
2. **UFCS**: `xs.map(f)` — 호출 위치의 스코프가 가리키는 `map`과 스코프 안 모듈
   `m`의 `m::map` 중 호출의 타입에 맞는 함수 검색. Qualified UFCS `x.a::b(y)`는
   스코프의 `a::b`와 스코프 안 모듈 `m`의 `m::a::b`를 후보로 같은 규칙을 따른다
   ([syntax.md](syntax.md#call-and-ufcs)).
3. **Unqualified**: `map(xs, f)` — use된 모듈 중 타입이 맞는 함수 검색

비한정 이름이 고를 후보는 그 이름으로 `use`한 함수들이다. 한 스코프의 여러
`use`가 같은 이름에 서로 다른 함수를 주면 그 이름은 그 함수들을 모두 가리킨다.
`f(x, y)`와 `x.f(y)`는 후보를 모으는 범위만 다르다. 그중 하나를 고르는 방식과,
고른 뒤 인자를 검사하고 인자 수와 타입의 오류를 보고하는 방식은 같다. 후보는 가져온
방식에 따라 [순위](#후보의-순위)를 가진다.

UFCS의 후보는 호출 위치의 스코프에서만 찾는다. `x.f(y)`의 후보는 스코프가
가리키는 함수 `f`(여러 `use`가 준 함수 전부)와, 스코프 안의 이름 `m`마다 `m::f`이다.
타입의 이름은 그 companion 네임스페이스를 가리키므로, 스코프에 있는 타입 `T`의
`T::f`가 여기에 든다. 스코프 밖의 모듈은 receiver 타입을 선언한 모듈이라도 찾지
않는다. 다른 모듈의 타입에 메서드를 쓰려면 그 타입을 `use`한다:

```rust
mod shapes {
    pub struct Point { x: Nat }
    pub mod Point {
        pub fn double(p: Point) -> Nat { p.x * 2 }
    }
}

mod a {
    fn f(p: super::shapes::Point) -> Nat { p.double }   // Error: Point가 스코프에 없음
}

mod b {
    use super::shapes::Point
    fn f(p: Point) -> Nat { p.double + p.x }            // OK
}
```

스코프에 없는 모듈의 선언은 호출의 후보를 바꾸지 않는다.

### 암묵적 생성자의 후보

[암묵적 생성자](type-inference.md#암묵적-생성자) `.Name`은 그 자리의 타입이 정해지면
그 타입의 생성자를 쓴다. 타입이 정해지지 않았을 때만 스코프의 타입에서 후보를
모은다. 후보는 스코프 안의 타입 이름 `T`마다 `T`의 생성자 `Name`이다. 경로
`T::Name`으로 쓸 수 있는 생성자이며, 쓰는 자리에서 보이는 것만 든다. 순위는
UFCS의 [후보의 순위](#후보의-순위)를 따른다.

1. **스코프의 타입**: 호출 위치의 스코프가 선언이나 `use`로 가리키는 타입.
2. **Prelude**: prelude의 짧은 이름이 가리키는 타입. 스코프의 선언이나 `use`가 그
   짧은 이름을 가려도 이 순위의 후보로 남는다.

같은 타입을 여러 이름으로 가리켜도 후보는 하나다. 고르는 방식과 기다리는 방식,
모호성 오류는 [후보 선택](#후보-선택)과 같다.

```rust
use std::io::Error

fn f() -> Result(Int, Error) {
    .Error { error: .EndOfFile }    // 기대 타입 Result의 Error, 그 필드 타입 Error의
}                                   // EndOfFile
```

### 후보의 순위

호출의 후보는 그 함수를 가져온 방식에 따라 순위를 가진다. 순위는 `use` 선언의
순서와 상관없다.

1. **직접 가리키는 함수**: 호출 위치의 스코프가 그 이름으로 가리키는 함수. 그 스코프의
   선언과, 그 이름으로 `use`한 함수다.
2. **스코프 안 이름의 namespace**: 스코프 안의 이름 `m`마다 `m::f`. UFCS에만 있는
   순위다. 모듈의 함수, 타입 `T`의 companion namespace의 함수, struct field의
   getter가 여기에 든다.
3. **Prelude**: prelude의 짧은 이름이 가리키는 함수와, UFCS에서 prelude 이름 `p`의
   `p::f`. 스코프의 선언이나 `use`가 그 짧은 이름을 가려도 이 순위의 후보로 남는다.

Qualified UFCS `x.a::b(y)`의 후보는 첫 segment를 가져온 방식으로 순위를 가진다.
스코프가 그 선언이나 `use`로 직접 가리키는 `a`의 `a::b`는 1순위, 스코프 안 이름
`m`의 `m::a::b`는 2순위, prelude 이름인 `a`의 `a::b`와 prelude 이름 `p`의
`p::a::b`는 3순위다. 경로 키워드나 의존 패키지 이름으로 시작하는 경로는 그 경로가
가리키는 함수 하나를 1순위로 둔다.

호출은 가장 높은 순위부터 [후보 선택](#후보-선택) 규칙을 적용한다. 그 순위에 맞는
후보가 있으면 그 순위 안에서 고르며, 맞는 후보가 여럿이면 그 순위 안의 모호성이다.
그 순위에 맞는 후보가 하나도 없을 때만 다음 순위로 내려간다. 아직 정해지지 않은
타입은 후보를 배제하지 않으므로, 높은 순위의 후보가 타입으로 배제될 때까지 낮은
순위로 내려가지 않는다. 비한정 호출 `f(x)`에서 지역 변수로 묶은 이름은 함수 후보를
찾지 않고 그 변수를 가리킨다.

높은 순위의 후보는 타입이 맞는 낮은 순위의 후보를 가린다. 예를 들어 직접 `use`한
함수는 같은 이름의 getter보다 먼저 고른다. 이 둘을 구분하려면 경로로 쓴다.

### 후보 선택

호출은 후보 중에서 자신의 타입에 맞는 함수를 고른다. 후보가 맞으려면:

- 매개변수의 수가 인자의 수와 같고 (UFCS의 receiver는 첫 번째 인자다),
- 각 매개변수의 타입이 그 자리 인자의 타입과 맞으며,
- 결과 타입이 호출의 결과에 기대되는 타입과 맞아야 한다.

두 타입은 같은 nominal 선언이거나 같은 primitive이면 맞는다. 함수 타입과 tuple은
항목 수가 같으면 맞는다. 후보의 타입 변수는 어떤 타입과도 맞는다. 아직 정해지지
않은 인자나 결과의 타입은 어느 후보도 배제하지 않는다. Ability row는 선택에 쓰지
않는다.

모든 순위를 합쳐 후보가 하나뿐이면 고를 것이 없다. 호출은 그 함수로 검사하고, 맞지
않는 인자는 보통의 타입 오류다. 그렇지 않으면 [순위](#후보의-순위)마다 고른다. 한
순위에서 맞는 후보가 정확히 하나이면 그 함수를 고른다. 그 뒤의 검사는 고른 함수의
선언으로 하므로, 맞는다고 본 타입의 세부(generic 인자, 함수의 매개변수)가 다르면
보통의 타입 오류가 된다. 맞는 후보가 여럿이면 타입이 더 정해질 때까지 기다리고,
함수 본문을 다 푼 뒤에도 여럿이면 오류다. 어느 순위에도 맞는 후보가 없으면 오류다.
다만 모든 순위를 통틀어 인자의 타입은 맞고 수만 다른 후보가 하나이면 그 함수의 인자
수 오류로 보고한다.

Receiver는 첫 번째 인자이며 다른 인자와 다르지 않다. Receiver의 타입이 정해지지
않았어도 나머지 인자와 결과가 후보를 하나로 좁히면 그 함수를 고르고, receiver의
타입은 고른 함수에서 추론된다.

필드를 읽는 `x.f`도 호출이다. Struct `T`의 필드 `f`는 getter 함수 `T::f`를 만들고,
`x.f`는 `f(x)`이다. Getter는 `T`의 네임스페이스에 있으므로 `T`가 스코프에 있을 때
`x.f`의 후보가 된다. 스코프의 타입 중 필드 `f`를 가진 struct가 하나뿐이고 같은
이름의 다른 후보가 없으면 `x.f`는 `x`의 타입 없이도 정해진다.

Getter `T::f`는 `T`의 네임스페이스에 선언된 함수다. 한 네임스페이스에서 한 이름은
한 함수이므로, `T`의 companion 모듈은 필드와 같은 이름의 함수를 매개변수와 상관없이
선언할 수 없다. 다른 모듈은 같은 이름의 함수를 선언할 수 있다. 그 함수가 `T`를
첫 번째 인자로 받고 `x.f`의 스코프에서 후보가 되면 후보는 getter와 그 함수 둘이고,
다른 호출과 같은 규칙으로 고른다. 모듈 이름을 통해 후보가 된 함수는 getter와 같은
순위이므로, 둘 다 맞으면 모호하며 `x.T::f`처럼 경로로 구분한다. 그 함수를 직접
`use`했다면 더 높은 순위이므로 getter보다 먼저 고른다:

```rust
struct Name { text: String }

pub mod Name {
    pub fn text(name: Name, suffix: String) -> String { ... }  // Error: Name::text는 필드의 getter
}

mod audit {
    pub fn text(name: Name) -> String { ... }                  // OK: audit::text
}

fn show(name: Name) -> String {
    name.text           // Error: 같은 순위의 Name::text와 audit::text 모두 맞음
    name.Name::text     // OK
    name.audit::text    // OK
}

mod report {
    use super::{Name, audit::text}

    fn show(name: Name) -> String {
        name.text       // audit::text: 직접 use한 함수가 getter보다 먼저다
    }
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
가리키며, `use`한 함수들은 후보가 아니다. 지역 변수가 아니라면 prelude의 같은 이름
함수는 [마지막 순위](#후보의-순위)의 후보로 남는다. 같은 함수를 여러 번 `use`하면
후보는 하나다.

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

여러 함수를 `use`한 이름은 쓰이는 자리의 타입으로 고른다. 인자 없는 호출은 결과에
기대되는 타입으로 고른다. 호출하지 않고 값으로 쓰면 그 자리에 기대되는 함수
타입의 매개변수와 결과를, 호출의 인자와 결과처럼 써서 고른다. 그 타입이 하나로
좁히지 못하면 모호성 오류이며 함수를 경로로 지정한다:

```rust
use a::size   // fn size(x: A) -> Nat
use b::size   // fn size(x: B) -> Nat
use a::make   // fn make() -> Nat
use b::make   // fn make() -> String

size(A { n: 1 })    // OK: a::size
size(B { n: 2 })    // OK: b::size
size(1)             // Error: 어느 size도 Nat 하나를 받지 않는다
count(make())       // OK: count가 Nat을 받으면 a::make
let _ = make()      // Error: 결과 타입이 정해지지 않아 모호하다. a::make()로 지정
apply(A { n: 1 }, size)   // OK: apply가 fn(A) -> b를 기대하므로 a::size
let f = size
apply(A { n: 1 }, f)      // OK: f가 쓰이는 자리의 타입으로 a::size
let g = size              // Error: g가 어디에도 쓰이지 않아 모호하다. a::size로 지정
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
| 선행 `::` | 의존 패키지의 루트 | `use ::std::io::print_line` |

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
ability annotation에 똑같이 적용한다. 패키지 루트에서 쓴 `super`, 그리고 첫
segment가 아닌 위치의 키워드는 이름 해석 오류다.

### 인라인 모듈의 이름 범위

인라인 모듈은 파일 기반 모듈과 같은 범위를 갖는다. 모듈 안에서 보이는 이름은 그
모듈의 선언, 그 모듈의 `use`, prelude뿐이다. 감싼 모듈이나 패키지 루트의 항목은
자동으로 보이지 않으며, `use super::x`나 `pkg::x`로 가져온다. 그래서 인라인 모듈을
파일로 옮겨도 이름의 뜻이 바뀌지 않는다.

경로의 첫 segment도 같은 범위에서 찾는다. 그 모듈의 하위 모듈이나 선언, 그 모듈이
`use`로 가져온 이름, prelude가 제공하는 namespace(`Option` 등), 의존 패키지
이름(`std`)이 대상이다. `use` 경로도 같다. 패키지 루트는 자신의 선언과 `use`,
prelude, 의존 패키지 이름을 본다.

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

패키지 `std`도 같은 규칙을 따른다. `std`의 모듈은 다른 `std` 항목을 prelude 짧은
이름으로 쓰거나 `use super::…`나 `pkg::…`로 가져온다.

### 가시성 (Visibility)

| 수식자 | 범위 |
| ------ | ---- |
| (없음), `pub(self)` | 선언한 모듈과 그 하위 모듈 |
| `pub(super)` | 부모 모듈과 그 하위 모듈 |
| `pub(pkg::a::b)` 등 모듈 경로 | 그 모듈과 그 하위 모듈 |
| `pub(pkg)` | 패키지 내부 전체 |
| `pub` | 공개 (외부 패키지에서도 접근 가능) |

```rust
// src/internal/utils.trb

fn private_helper() -> Int { 42 }           // 이 모듈에서만

pub(super) fn parent_visible() -> Int { 42 } // internal/ 및 하위에서

pub(pkg::internal) fn internal_only() -> Int { 42 } // internal/ 및 하위에서

pub(pkg) fn package_internal() -> Int { 42 } // 이 패키지 내에서

pub fn public_api() -> Int { 42 }            // 어디서든
```

패키지 `std`도 같은 규칙을 따른다. 항목과 참조하는 모듈이 같은 패키지인지는 경로의
철자가 아니라 선언이 속한 패키지로 정한다. 다른 패키지에서는 `pub` 항목만 보이므로,
사용자 패키지는 `std`의 `pub` 항목만 보고 `pub(pkg)`인 `std::runtime`은 보지 못한다.

가시성 수식자는 함수, extern 함수, `struct`, `enum`, `ability`, `mod`, `use`에 붙는다.

**제한된 가시성의 경로.** `pub(…)`의 괄호 안은 모듈 경로이며, 그 모듈과 그 하위 모듈에서
항목이 보인다. 경로는 [경로 키워드](#경로-키워드)로 시작하고 `use` 경로와 같이 해석한다.
`pub(pkg)`, `pub(super)`, `pub(self)`는 segment가 하나인 경우다. 경로는 선언한 모듈
자신이거나 그 조상 모듈이어야 한다. 그렇지 않은 모듈, 모듈이 아닌 이름, 존재하지 않는
경로는 이름 해석 오류다. 패키지 루트에서 쓴 `pub(super)`도 루트 위를 가리키므로 오류다.

**경로의 모든 모듈.** 항목은 그 항목 자신과 그 항목을 둘러싼 모든 모듈이 참조하는
모듈에서 보일 때만 보인다. 그래서 `pub(pkg) mod runtime` 안의 `pub fn`은 패키지 밖에서
보이지 않는다. 이 규칙은 이름 해석의 모든 경로에 똑같이 적용한다. `use` 경로, 값
경로(호출, 생성자, 패턴, handler arm의 ability), 타입과 ability annotation, UFCS 후보가
모두 그렇다. 보이지 않는 함수는 UFCS 후보가 되지 않는다. 보이지 않는 항목을 이름으로
가리키면 이름 해석 오류이며, 진단은 그 항목이나 그 항목을 가린 모듈의 식별 경로와
선언된 가시성을 표시한다.

**타입의 구성원.** 가시성 수식자는 선언 단위로 붙는다. Enum의 variant는 enum의
가시성을 따르고, struct의 field와 field 함수(`T::f`, `T::f::set`, `T::f::modify`)는
struct의 가시성을 따른다. Ability의 operation은 ability의 가시성을 따른다.

**Companion 모듈.** 타입과 같은 이름의 companion 모듈은 별개의 선언이며 자기 가시성을
가진다. `T::f`처럼 경로가 companion 모듈을 지나면 그 모듈의 가시성이 적용되고, 타입의
가시성은 타입과 생성자에만 적용된다. 그래서 공개 타입의 companion 모듈도 함수를
공개하려면 `pub mod`로 선언한다.

**`use`와 re-export.** `use`의 대상은 `use`를 선언한 모듈에서 보여야 한다. 가져온
이름은 그 `use`의 가시성을 가진다. 비공개가 아닌 `use`는 가져온 이름을 선언한 모듈의 이름
공간에 더한다. 그래서 `pub mod facade { pub use pkg::source::f }`의 `facade::f`는
경로로 가리킬 수 있고, 패키지 루트의 `pub use`는 패키지 밖에서도 보인다. 이 경로의
가시성은 대상이 아니라 그 `use`가 정한다. Re-export는 대상의 식별을 바꾸지 않으므로,
re-export한 경로로 쓴 타입과 ability annotation도 대상 선언을 가리킨다.
`pub use`처럼 대상보다 넓은 가시성으로 다시 내보낼 수는 없으며, 이는 이름 해석
오류다. 대상의 가시성에는 대상을 둘러싼 모듈의 가시성도
포함되므로, private 모듈의 `pub` 항목을 바깥에 `pub use`로 공개하는 facade 방식은 쓸 수
없다. 공개할 항목은 공개 모듈에 둔다.

### Re-export

```rust
// 공개 모듈 internal의 타입을 루트 경로로도 노출
pub use pkg::internal::PublicType
pub use pkg::internal::{TypeA, TypeB}

// 별칭으로 re-export
pub use pkg::internal::LongTypeName as Short
```

Re-export한 이름은 대상을 가리킨다. 그 이름 뒤로 이어지는 경로는 대상에서, 대상의
namespace를 따라 끝까지 이어진다. 모듈 `m`이 `pub use std as library`를 선언하면
`m::library::Option::Some`은 패키지 `std`의 `Option::Some`이다. 경로가 지나는
re-export도 [가시성](#가시성-visibility) 규칙을 따르므로, 쓰는 자리에서 보이지 않는
re-export를 지나는 경로는 그 re-export에서 멈춘다.

### 완전한 예시

```rust
// src/lib.trb
pub mod internal       // public 모듈 (re-export 대상을 담는다)
pub mod api            // public 모듈

// 주요 타입들을 루트에서 re-export
pub use pkg::internal::Config
pub use pkg::api::{Request, Response}


// src/internal/mod.trb (또는 src/internal.trb)
pub struct Config {
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

1. **Prelude 버전**: `std::prelude::v0` 다음 버전을 언제, 어떻게 고르는지
2. **조건부 컴파일**: `#[cfg(...)] mod foo` 문법 및 지원 범위
3. **매니페스트**: `Tribute.toml` 형식 및 필수 여부
4. **lang-examples 구조**: 각 예제를 개별 폴더의 `main.trb`로 재구성
