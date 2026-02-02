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

### Effect Annotation 생략

Effect annotation을 생략하면 fresh한 ability 변수가 생성된다:

```rust
fn(a) -> b       // fn(a) ->{e} b (fresh e)
```

따라서 고차 함수에서 내부 함수의 effect를 전파하려면 **같은 변수를 명시**해야 한다:

```rust
// 올바른 선언: 같은 변수 e 사용
fn map(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)
fn filter(xs: List(a), p: fn(a) ->{e} Bool) ->{e} List(a)
fn fold(xs: List(a), init: b, f: fn(b, a) ->{e} b) ->{e} b
```

### 순수 함수 (Ability 없음)

```rust
fn(a) ->{} b    // 빈 ability 집합 = 순수 함수
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
let users = Map::new(Text::eq, Text::hash)
let users = users.insert("alice", alice).insert("bob", bob)
users.get("alice")  // eq, hash 불필요
```

### 특정 Ability 명시

```rust
fn(Text) ->{Http} Response             // Http만
fn(Text) ->{Http, Async} Response      // Http와 Async
```

## 함수 정의 문법

함수 타입과 동일한 구조를 사용한다:

```rust
// 특정 ability
fn fetch(url: Text) ->{Http} Response {
    Http::get(url)
}

// 여러 ability
fn fetch_all(urls: List(Text)) ->{Http, Async} List(Response) {
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

Ability operation은 일반 함수처럼 호출한다. 별도의 `perform` 키워드 없이 `Ability::operation(args)` 형태를 사용한다:

```rust
fn fetch(url: Text) ->{Http} Response {
    Http::get(url)  // perform 없이 직접 호출
}

fn greet() ->{Console} Nil {
    Console::print("Hello!")
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

```rust
ability Http {
    fn get(url: Text) -> Response
    fn post(url: Text, body: Text) -> Response
}

ability Async {
    fn await(promise: Promise(a)) -> a
}

ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

ability Console {
    fn print(msg: Text) -> Nil
    fn read() -> Text
}
```

## Handler

### `handle` 표현식

`handle expr { arms }`는 computation을 실행하고 handler arm으로 결과를 처리한다:

```rust
handle computation() {
    { result } -> result                    // 완료 시
    { State::get() -> k } -> k(42)          // suspend 시
}
```

computation의 실행 결과는 두 가지 중 하나:

- **완료**: computation이 값을 반환함 → `{ value }` 패턴 매칭
- **Suspend**: ability operation에서 멈춤 → `{ Op(args) -> k }` 패턴 매칭

### Handler 패턴

| 패턴 | 의미 |
| ---- | ---- |
| `{ value }` | Computation 완료, 결과값 바인딩 |
| `{ Operation(args) -> k }` | Suspend, continuation `k` 바인딩 |

Handler 패턴은 `handle` 표현식 내에서만 사용할 수 있다.

### Continuation

Continuation `k`는 일반 함수처럼 호출한다:

```rust
{ State::get() -> k } -> k(current_state)  // resume
{ State::set(v) -> k } -> k(Nil)            // resume with unit
```

#### Linear Type

Continuation은 **linear type**이다. 반드시 한 번 사용하거나 명시적으로 버려야 한다:

```rust
// 사용: 함수로 호출
{ State::get() -> k } -> k(state)

// 버림: drop 함수 사용
{ Fail::fail(msg) -> k } -> {
    drop(k)
    None
}
```

사용하지도 버리지도 않으면 타입 에러:

```rust
// 컴파일 에러: k가 사용되지 않음
{ Fail::fail(msg) -> k } -> None
```

### 기본 예시

```rust
fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() k(state), state)
        { State::set(v) -> k } -> run_state(fn() k(Nil), v)
    }
}
```

### Abort 패턴

Continuation을 사용하지 않으면 명시적으로 버려야 한다:

```rust
fn run_maybe(comp: fn() ->{e, Fail} a) ->{e} Option(a) {
    handle comp() {
        { result } -> Some(result)
        { Fail::fail(msg) -> k } -> {
            drop(k)
            None
        }
    }
}
```

### 여러 Ability 처리

중첩하거나 하나의 handler에서 여러 ability를 처리할 수 있다:

```rust
fn run_console(comp: fn() ->{e, Console} a) ->{e, IO} a {
    handle comp() {
        { result } -> result
        { Console::print(msg) -> k } -> {
            IO::write(stdout, msg)
            run_console(fn() k(Nil))
        }
        { Console::read() -> k } -> {
            let input = IO::read(stdin)
            run_console(fn() k(input))
        }
    }
}
```

## 전체 예시

```rust
ability Console {
    fn print(msg: Text) -> Nil
    fn read() -> Text
}

ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{Console, State(Int)} Nil {
    let n = State::get()
    Console::print("Count: " <> Int::to_string(n))
    State::set(n + 1)
}

fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() k(state), state)
        { State::set(v) -> k } -> run_state(fn() k(Nil), v)
    }
}

fn run_console(comp: fn() ->{e, Console} a) ->{e, IO} a {
    handle comp() {
        { result } -> result
        { Console::print(msg) -> k } -> {
            IO::write(stdout, msg)
            run_console(fn() k(Nil))
        }
        { Console::read() -> k } -> {
            let input = IO::read(stdin)
            run_console(fn() k(input))
        }
    }
}

fn main() ->{IO} Nil {
    fn() {
        fn() {
            counter()
            counter()
            counter()
        }.run_state(0)
    }.run_console()
}
```

## 요약

| 문법 | 의미 |
| ---- | ---- |
| `fn(a) -> b` | `fn(a) ->{e} b` (fresh e) |
| `fn(a) ->{} b` | 순수 함수 타입 |
| `fn(a) ->{E} b` | ability E를 수행하는 함수 타입 |
| `fn(a) ->{e, E} b` | ability E + 나머지 e |
| `Nil` | Unit 타입/값 |
| `{ ... }` | block expression |
| `fn(x) expr` | 람다 |
| `handle expr { ... }` | effect handling |
| `{ value }` | completion 패턴 (handle 내) |
| `{ Op(args) -> k }` | suspend + continuation 패턴 (handle 내) |
