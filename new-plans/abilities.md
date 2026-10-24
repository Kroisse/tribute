# Tribute Ability System

> 이 문서는 Tribute의 ability (algebraic effect) 시스템 문법과 의미론을 정의한다.

## 용어

Tribute는 Unison의 선례를 따라 algebraic effect를 **ability**라고 부른다.

## 함수 타입과 Ability

### 기본 문법

```rust
fn(a) ->{E} b
```

- `{E}`: 이 함수가 수행할 수 있는 ability 집합
- 화살표 `->`와 ability `{E}`가 붙어있어 "이 ability들을 수행하며 b를 반환"이라는 인과관계가 명확함

### Ability Annotation 생략

Ability annotation을 생략하면 fresh한 ability 변수가 생성된다:

```rust
fn(a) -> b       // fn(a) ->{e} b (fresh e)
```

타입 선언(struct·enum 필드, ability operation 시그니처) 안의 함수 타입은 예외로,
생략하면 빈 row이고 row 변수를 쓸 수 없다([선언 안의 함수 타입](types.md#선언-안의-함수-타입)).

함수 선언의 effect는 본문에서 추론하지 않는다. Effect를 생략한 함수의 본문은
concrete ability를 수행할 수 없으며, ability를 수행하는 함수는 row에 명시한다.

따라서 고차 함수에서 내부 함수의 effect를 전파하려면 **같은 변수를 명시**해야 한다:

```rust
// 올바른 선언: 같은 변수 e 사용
fn map(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)
fn filter(xs: List(a), p: fn(a) ->{e} Bool) ->{e} List(a)
fn fold(xs: List(a), init: b, f: fn(b, a) ->{e} b) ->{e} b
```

### 순수 함수 (Ability 없음)

```rust
fn(a) ->{} b    // 빈 ability 집합을 명시한 pure 함수
```

순수 함수가 필요한 경우 명시적으로 `{}`를 표기한다:

```rust
// Map 생성 시 eq, hash는 순수해야 함
fn new(eq: fn(k, k) ->{} Bool, hash: fn(k) ->{} Int) -> Map(k, v)

// memoization은 순수 함수에만 의미 있음
fn memoize(f: fn(a) ->{} b) ->{} fn(a) ->{} b
```

참고: `Map`은 생성 시점에 `eq`, `hash`를 받아 내부에 보관한다. 이후 `insert`, `get` 등에서는 전달할 필요 없다:

```rust
let users = Map::new(String::eq, String::hash)
let users = users.insert("alice", alice).insert("bob", bob)
users.get("alice")  // eq, hash 불필요
```

### 특정 Ability 명시

```rust
fn(String) ->{Http} Response             // Http만
fn(String) ->{Http, Async} Response      // Http와 Async
```

### Host Interface

Program 밖의 기능은 host가 처리하는 ability로 연다. 그런 ability도 일반 ability이며,
program 안에서 처리되지 않고 host가 부르는 entrypoint(`main` 등)에 남으면 host가
처리한다. Entrypoint에 남은 row들이 program의 host interface이고, 규칙은
[host-interface.md](host-interface.md)가 정한다. Compiler가 특별히 아는 ambient
ability는 없다. 표준 입출력도 패키지 `std`의 source에 선언한 ability `std::io::Io`와 그 위의
`std::io::Read`, `std::io::Write`로 제공한다([io.md](io.md)).

```rust
use std::abilities::Throw
use std::io::{Read, Write, Error, print, print_line, read_line}

fn greet(name: String) ->{Write} Nil {
    print("Hello, ")
    print_line(name)
}

fn prompt() ->{Read, Write, Throw(Error)} String {
    print("Name: ")
    read_line()
}
```

## 함수 정의 문법

함수 타입과 동일한 구조를 사용한다:

```rust
// 특정 ability
fn fetch(url: String) ->{Http} Response {
    Http::get(url)
}

// 여러 ability
fn fetch_all(urls: List(String)) ->{Http, Async} List(Response) {
    urls.map(fn(url) url.get.await)
}

// 순수 함수
fn add(x: Int, y: Int) ->{} Int {
    x + y
}
```

## 람다 문법

### 기본 형태

```rust
fn(<args>) <expr>
```

### Block Expression

`{ ... }`는 block expression이다. 여러 문장을 순차 실행하고 마지막 값을 반환한다:

```rust
{
    let x = 1
    let y = 2
    x + y
}  // 3을 반환
```

### 람다와 Block

`fn(args) expr`에서 `expr`이 block이면 자연스럽게 여러 문장을 가진 람다가 된다:

```rust
fn(x) x + 1              // 단일 표현식

fn(x) {                  // block expression
    let y = x + 1
    y * 2
}
```

### 사용 예시

```rust
xs.map(fn(x) x * 2)
xs.filter(fn(x) x > 0)
xs.fold(0, fn(a, b) a + b)
opt.or_else(fn() fallback())
```

### Thunk 타입

인자 없는 함수 (thunk)의 타입:

```rust
fn or_else(opt: Option(a), fallback: fn() ->{e} a) ->{e} a {
    case opt {
        Some(x) -> x
        None -> fallback()
    }
}
```

## Ability 호출

Ability operation은 일반 함수처럼 호출한다. 별도의 `perform` 키워드 없이
`Ability::operation(args)` 형태를 사용한다:

```rust
fn fetch(url: String) ->{Http} Response {
    Http::get(url)  // perform 없이 직접 호출
}
```

타입 시그니처가 이미 `->{E}` 형태로 ability 정보를 담고 있으므로, 호출 시점에 별도 표기는 불필요하다.

### UFCS와 Ability

Ability operation도 UFCS를 통해 메서드처럼 호출할 수 있다. 인자가 없으면 괄호 생략 가능:

```rust
// 명시적 호출
Http::get(url)
Async::await(promise)

// UFCS (괄호 생략)
url.get
promise.await

// 체이닝
url.get.await
```

## Ability 정의

### Operation 종류

Ability operation은 두 종류로 선언한다:

- **`fn`**: Tail-resumptive operation. Handler가 값을 반환하면 자동으로 resume된다.
  Continuation을 캡처하지 않으므로 일반 함수 호출과 동등한 성능.
  **선언부가 보증(guarantee)이다**: `fn`으로 선언된 operation은 모든 handler에서
  반드시 `fn`으로 구현해야 한다.
- **`op`**: General operation. Handler에서 `resume` 키워드를 통해
  resume 여부와 시점을 제어한다. `-> Never`를 반환하면 절대 resume하지 않는
  abort/exception 패턴을 표현한다.
  `op`으로 선언된 operation은 handler에서 반드시 `op`으로 구현한다.

```rust
ability Http {
    fn get(url: String) -> Response
    fn post(url: String, body: String) -> Response
}

ability Async {
    fn await(promise: Promise(a)) -> a
}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Logger {
    op log(msg: String) -> Nil
}

ability Fail {
    op fail(msg: String) -> Never  // never resumes
}
```

### Operation의 Ability Row 규칙

Ability operation은 **자신이 속한 ability의 effect만** 암시적으로 수행한다.
Operation 시그니처에 ability annotation을 적을 수 없다:

```rust
// ✅ OK: effect는 암시적으로 부여됨
ability Stream(a) {
    op emit(value: a) -> Nil
}

// ❌ 불가: ability annotation 자체가 문법 오류
ability Stream(a) {
    op emit(value: a) ->{Stream(a)} Nil
}

// ❌ 불가: 다른 ability를 참조
ability Reporter {
    op report(message: String) ->{Reporter, Write} Nil
}
```

다른 effect가 필요한 경우, operation 자체가 아닌 **handler 구현부**에서 처리한다:

```rust
// operation은 자신의 ability만 수행
ability Reporter {
    op report(message: String) -> Nil
}

// handler가 Write effect를 사용하여 구현
fn run_reporter(comp: fn() ->{e, Reporter} a) ->{e, Write} a {
    handle comp() {
        op Reporter::report(message) {
            print_line(message)
            resume Nil
        }
    }
}
```

이 규칙은 ability를 **순수한 효과 선언**으로 유지하고,
구체적인 효과 해석은 handler에 위임하기 위한 것이다.
Koka, Unison 등 기존 algebraic effect 언어들도 동일한 제약을 채택하고 있다.

## Handler

### `handle` 표현식

`handle expr { arms }`는 computation을 실행하고 handler arm으로 결과를 처리한다.

Handler arm은 함수 정의와 대칭적인 구조를 가진다. Ability 선언에서 `fn`/`op`으로
operation을 정의하듯, handler에서도 `fn`/`op` 키워드로 각 operation의 구현을 작성한다:

```rust
handle computation() {
    do result { result }
    op Logger::log(msg) {
        print_line(msg)
        resume Nil
    }
    op State::get() { run_state(fn() resume state, state) }
}
```

computation의 실행 결과는 두 가지 중 하나:

- **완료**: computation이 값을 반환함 → `do value { expr }` 매칭
- **Suspend**: ability operation에서 멈춤 → `fn`/`op` handler arm 매칭

`do` arm은 생략할 수 있다. 생략하면 `do value { value }`, 즉 body의 결과를 그대로
handle의 결과로 내는 identity completion을 쓴다:

```rust
handle computation() {
    op Logger::log(msg) {
        print_line(msg)
        resume Nil
    }
}
```

### Handler Arm 종류

| arm | 대상 | 의미 |
| --- | ---- | ---- |
| `do value { expr }` | completion | Computation 완료, 결과값 바인딩 (생략 시 identity) |
| `fn Op(args) { body }` | `fn` operation만 | Tail-resumptive: body의 반환값이 resume 값 |
| `op Op(args) { body }` | `op` operation만 | body에서 `resume`으로 명시적 resume |

Handler arm은 `handle` 표현식 내에서만 사용할 수 있다.

**`do` arm 생략:**

- `handle`의 `do` arm은 없거나 하나다. 둘 이상이면 typechecking 단계의 컴파일 에러다.
- `do` arm이 없는 `handle`의 결과 타입은 body의 결과 타입이다. 따라서 값을 내는
  `op` arm, 곧 `resume`하지 않고 끝나는 arm의 결과도 body의 결과 타입이어야 한다.
  `resume`의 결과도 같은 타입이다.
- 생략은 결과를 바꾸지 않는 completion에 한한다. Body의 결과를 다른 타입이나 값으로
  바꾸려면 `do` arm을 적는다.

**Handler arm과 선언의 관계:**

- `fn`으로 선언된 operation → handler에서 반드시 `fn`으로 구현 (`op`으로 구현 시 컴파일 에러)
- `op`으로 선언된 operation → handler에서 반드시 `op`으로 구현 (`fn`으로 구현 시 컴파일 에러)
- Arm의 종류는 선언의 종류와 같다. 꼬리에서 바로 `resume`하는 `op` arm을 빠른 경로로
  내리는 것은 최적화가 정하며, 언어 계약에 속하지 않는다.

**Handler arm의 완전성:**

- `handle` 표현식이 어떤 ability에 대해 `fn`/`op` arm을 하나라도 가지면, 그
  ability가 선언한 모든 operation에 대한 arm을 가져야 한다. 빠진 arm이 있으면
  typechecking 단계의 컴파일 에러이며, 진단은 ability와 빠진 operation을 모두
  나열한다. Parametrized ability(`State(s)`)는 ability 선언 단위로 판단하고,
  이름이 같아도 다른 module의 ability는 서로 다른 ability로 판단한다.
- 한 `handle`은 operation마다 arm을 최대 하나 가진다. 같은 operation의 arm이 둘
  이상이면 typechecking 단계의 컴파일 에러다.
- 빠진 arm은 암묵적으로 바깥 handler로 전달(forward)되거나 계산을 중단(abort)하거나
  trap하지 않는다. 일부 operation만 처리하는 handler는 허용되지 않는다.
- Arm이 하나도 없는 ability는 그 `handle`이 처리하지 않는다. 그 label은 결과
  row에 그대로 남고, 그 operation은 바깥 handler로 간다. `do` arm만 있는
  `handle`도 허용된다.

### `fn` Operation Handler

`fn` operation은 continuation을 캡처하지 않는다. Handler body의 반환값이 곧
resume 값이 된다. 함수 정의와 동일한 구조:

```rust
// 선언
ability Clock {
    fn now_millis() -> Int
}

// handler (구현)
fn fixed_clock(comp: fn() ->{e, Clock} a, now: Int) ->{e} a {
    handle comp() {
        fn Clock::now_millis() { now } // Int 반환 → resume now
    }
}
```

`fn` handler arm에서는 `resume`을 사용할 수 **없다** (컴파일 에러).

`fn` handler arm은 계산을 멈추거나(suspend) 버릴(abort) 수 없다. `fn` operation은
어떤 handler 아래에서도 continuation을 handler에 넘기지 않고 perform 지점으로
반환한다고 선언이 보증하고, `fn` arm은 그 operation의 구현이기 때문이다. 따라서 `fn`
arm 본문은 다음을 할 수 없다. `Throw`와 `-> Never` operation도 포함한다.

- `op` operation을 직접 perform한다.
- Ability row에 `op` operation을 선언한 ability가 있는 callable을 호출한다. 그런
  callable은 호출 도중 멈출 수 있다.

직접 perform은 operation 단위로 판단하므로, `fn`과 `op`를 함께 선언한 ability의 `fn`
operation은 직접 perform할 수 있다. 호출은 callee의 ability row로 판단하므로 그 ability를
요구하는 함수는 호출할 수 없다. Arm 안의 `handle`이 처리하는 label은 그 handle 안에서
제한을 받지 않는다. Lambda 본문은 lambda를 호출하는 곳에서 그 타입의 row로 판단한다.
Arm 정의의 타입에 드러나는 위반은 typechecking 단계의 컴파일 에러이며, 진단은 arm이
처리하는 operation과 arm이 닿는 금지된 ability를 나열한다.

계산을 멈추거나 중단해야 하는 handler는 그 operation을 `op`으로 선언하고 `op` arm으로
처리한다. 이 보증을 구현하는 방법, 곧 `fn` perform 지점이 continuation frame을 만들지
않는 dispatch는 [CPS 전략](cps-effects.md#fn-operation-direct-dispatch)이 정한다.

Row 변수로 들어오는 label도 같은 규칙을 따른다. `fn` arm이 effect-polymorphic
callable을 호출하면, 그 row 변수를 고정한 인스턴스의 label에도 `op` operation을
선언한 ability가 올 수 없다. 이 위반은 arm의 정의만으로는 드러나지 않는다. 정의에서는 그
row가 row를 생략한 순수 함수의 호출과 같은 열린 row이기 때문이다. 그래서 인스턴스마다
callee의 호출 규약을 정한 뒤 그 호출을 컴파일 에러로 거부한다
([`fn` arm 호출 검사](cps-effects.md#fn-arm-호출-검사)).

```rust
ability Stop { op stop() -> Never }

fn limited(comp: fn() ->{e, Counter} Int) ->{e, Stop} Int {
    handle comp() {
        fn Counter::next() { Stop::stop() } // ❌ 컴파일 에러: `fn` arm에서 `op` 수행
    }
}
```

### `op` Operation Handler와 `resume`

`op` operation의 handler body에서는 `resume` 키워드로 computation을 재개한다.
`resume`은 일반 함수처럼 호출한다:

```rust
op State::get() { resume current_state }     // resume with value
op State::set(v) { resume Nil }              // resume with unit
```

#### `resume`의 사용 규칙

`op -> T` handler body에서 `resume`은 **최대 1회** 호출할 수 있다 (affine).
호출하지 않으면 continuation은 암묵적으로 drop된다:

```rust
// 1회 호출: 정상 resume
op State::get() { resume state }

// 0회 호출: continuation 암묵적 drop (abort)
op SomeOp::cancel() { fallback_value }
```

항상 resume하지 않는 operation은 `-> Never`로 선언하는 것이 좋다.
`-> Never`는 continuation 캡처 자체를 생략하는 최적화를 가능하게 한다.

#### `resume`과 handler 선택

Handler arm은 처리한 label이 빠진 row로 검사된다. 따라서 arm이 직접 수행하는
operation은 그 handle 바깥의 handler가 처리한다. 재개된 계산의 handler는 `resume`이
놓인 위치가 정한다:

- **Arm 본문의 `resume`:** 재개된 계산은 같은 handle 아래에서 계속된다. Arm 본문의
  row는 continuation의 effect를 더하지 않으며, 처리한 label은 계속 이 handler에 묶인다.
- **Arm 안 lambda의 `resume`:** lambda의 row는 continuation의 effect를 담는다.
  재개된 계산의 label은 그 lambda를 호출한 곳이 묶은 handler, 예를 들어 arm 안에서
  새로 설치한 handler로 간다.

```rust
fn counter(comp: fn() ->{e, State(Nat)} a) ->{e, State(Nat)} a {
    handle comp() {
        // State::get()은 바깥 State로 간다. resume 뒤의 State는 이 handler가 받는다.
        op State::get() { resume State::get() + 1 }
        op State::set(v) { resume Nil }
    }
}
```

아래 `run_state`의 `run_state(fn() resume state, state)`는 두 번째 규칙을 쓴다.
재개된 계산은 새로 설치한 `run_state`의 handler로 이어진다.

### 기본 예시: State (op)

```rust
fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        op State::get() { run_state(fn() resume state, state) }
        op State::set(v) { run_state(fn() resume Nil, v) }
    }
}
```

### Abort 패턴 (op -> Never)

`-> Never`를 반환하는 operation은 절대 resume하지 않는다. Handler body에서
`resume`을 사용할 수 없으며, continuation 캡처도 발생하지 않는다:

```rust
ability Fail {
    op fail(msg: String) -> Never
}

fn run_maybe(comp: fn() ->{e, Fail} a) ->{e} Option(a) {
    handle comp() {
        do result { Some(result) }
        op Fail::fail(msg) { None }   // resume 없음, drop 불필요
    }
}
```

### 여러 Ability 처리

중첩하거나 하나의 handler에서 여러 ability를 처리할 수 있다:

```rust
fn run_logger(comp: fn() ->{e, Logger} a) ->{e, Write} a {
    handle comp() {
        op Logger::log(msg) {
            print_line(msg)
            resume Nil
        }
    }
}
```

## 전체 예시

```rust
use std::abilities::Exception
use std::io::{Io, Write, stdio, print_line}

ability Logger {
    op log(msg: String) -> Nil
}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn counter() ->{Logger, State(Int)} Nil {
    let n = State::get()
    Logger::log("Count: " <> Int::to_string(n))
    State::set(n + 1)
}

fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        op State::get() { run_state(fn() resume state, state) }
        op State::set(v) { run_state(fn() resume Nil, v) }
    }
}

fn run_logger(comp: fn() ->{e, Logger} a) ->{e, Write} a {
    handle comp() {
        op Logger::log(msg) {
            print_line(msg)
            resume Nil
        }
    }
}

fn main() ->{Io, Exception} Nil {
    with stdio
    fn() {
        fn() {
            counter()
            counter()
            counter()
        }.run_state(0)
    }.run_logger()
}
```

## 요약

### Operation 종류

| 선언 | resume | continuation 캡처 | handler arm | 용례 |
| ---- | ------ | ----------------- | ----------- | ---- |
| `fn now_millis() -> Int` | 1회, 암시적 | 없음 | `fn`만 가능 | Clock, Config |
| `op get() -> s` | 0~1회, `resume` (affine) | 있음 | `op`만 가능 | State, Stream, Coroutine |
| `op fail(msg: String) -> Never` | 0회 | 없음 | `op`만 가능 | Fail, Exception |

### Operation 규칙

| 규칙 | 설명 |
| ---- | ---- |
| 암시적 effect | Operation은 자신이 속한 ability의 effect를 암시적으로 수행 |
| ability annotation 금지 | Operation 시그니처에 ability annotation을 적을 수 없음 |
| 다른 effect 필요 시 | Handler 구현부에서 처리 |
| 선언과 arm 일치 | `fn`/`op`으로 선언 → 모든 handler에서 같은 키워드로 구현 |
| `fn` arm의 effect | `fn` arm 본문은 `op` operation을 perform하거나 그런 ability를 요구하는 callable을 호출할 수 없음 |

### 전체 문법 요약

| 문법 | 의미 |
| ---- | ---- |
| `fn(a) -> b` | `fn(a) ->{e} b` (fresh e) |
| `fn(a) ->{} b` | 빈 ability row를 명시한 pure 함수 타입 |
| `fn(a) ->{E} b` | ability E를 수행하는 함수 타입 |
| `fn(a) ->{e, E} b` | ability E + 나머지 e |
| `Nil` | Unit 타입/값 |
| `{ ... }` | block expression |
| `fn(x) expr` | 람다 |
| `handle expr { ... }` | effect handling |
| `do value { expr }` | completion arm (handle 내, 생략 시 identity) |
| `fn Op(args) { body }` | `fn` operation handler (handle 내) |
| `op Op(args) { body }` | `op` operation handler (handle 내) |

## Checked effect instances across compilation

Effect labels use the exact resolved ability declaration and ordered type
arguments; row label order does not select an instance. Accumulation preserves
open-tail union relationships through let/function generalization. Handler
subtraction consumes the selected instance without constraining every incoming
tail to contain it. Reached handler and perform identities must have resolved
semantic type arguments before IR type erasure. A physical anyref representation
is not evidence that an unresolved semantic argument is valid.
