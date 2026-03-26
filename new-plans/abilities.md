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

Ability operation은 일반 함수처럼 호출한다. 별도의 `perform` 키워드 없이
`Ability::operation(args)` 형태를 사용한다:

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

### Operation 종류

Ability operation은 두 종류로 선언한다:

- **`fn`**: Tail-resumptive operation. Handler가 값을 반환하면 자동으로 resume된다.
  Continuation을 캡처하지 않으므로 일반 함수 호출과 동등한 성능.
  **선언부가 보증(guarantee)이다**: `fn`으로 선언된 operation은 모든 handler에서
  반드시 `fn`으로 구현해야 한다.
- **`op`**: General operation. Handler에서 `resume` 키워드를 통해
  resume 여부와 시점을 제어한다. `-> Never`를 반환하면 절대 resume하지 않는
  abort/exception 패턴을 표현한다.
  `op`으로 선언된 operation은 handler에서 `fn`으로도 구현할 수 있다
  (tail-resumptive가 충분한 경우).

```rust
ability Http {
    fn get(url: Text) -> Response
    fn post(url: Text, body: Text) -> Response
}

ability Async {
    fn await(promise: Promise(a)) -> a
}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Console {
    fn print(msg: Text) -> Nil
    fn read() -> Text
}

ability Fail {
    op fail(msg: Text) -> Never  // never resumes
}
```

### Operation의 Effect Row 규칙

Ability operation은 **자신이 속한 ability의 effect만** 암시적으로 수행한다.
Operation 시그니처에 effect annotation을 적을 수 없다:

```rust
// ✅ OK: effect는 암시적으로 부여됨
ability Stream(a) {
    op emit(value: a) -> Nil
}

// ❌ 불가: effect annotation 자체가 문법 오류
ability Stream(a) {
    op emit(value: a) ->{Stream(a)} Nil
}

// ❌ 불가: 다른 ability를 참조
ability Cache(k, v) {
    fn get(key: k) ->{Cache(k, v), IO} v
}
```

다른 effect가 필요한 경우, operation 자체가 아닌 **handler 구현부**에서 처리한다:

```rust
// operation은 자신의 ability만 수행
ability Cache(k, v) {
    fn get(key: k) -> v
    fn put(key: k, value: v) -> Nil
}

// handler가 IO effect를 사용하여 구현
fn run_cache(comp: fn() ->{e, Cache(k, v)} a) ->{e, IO} a {
    handle comp() {
        do result { result }
        fn Cache::get(key) { IO::read_from_disk(key) }
        fn Cache::put(key, value) { IO::write_to_disk(key, value) }
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
    fn Console::print(msg) { IO::write(stdout, msg) }
    op State::get() { run_state(fn() resume state, state) }
}
```

computation의 실행 결과는 두 가지 중 하나:

- **완료**: computation이 값을 반환함 → `do value { expr }` 매칭
- **Suspend**: ability operation에서 멈춤 → `fn`/`op` handler arm 매칭

### Handler Arm 종류

| arm | 대상 | 의미 |
| --- | ---- | ---- |
| `do value { expr }` | completion | Computation 완료, 결과값 바인딩 |
| `fn Op(args) { body }` | `fn`/`op` operation | Tail-resumptive: body의 반환값이 resume 값 |
| `op Op(args) { body }` | `op` operation만 | body에서 `resume`으로 명시적 resume |

Handler arm은 `handle` 표현식 내에서만 사용할 수 있다.

**Handler arm과 선언의 관계:**

- `fn`으로 선언된 operation → handler에서 반드시 `fn`으로 구현 (`op`으로 구현 시 컴파일 에러)
- `op`으로 선언된 operation → handler에서 `op` 또는 `fn`으로 구현 가능

### `fn` Operation Handler

`fn` operation은 continuation을 캡처하지 않는다. Handler body의 반환값이 곧
resume 값이 된다. 함수 정의와 동일한 구조:

```rust
// 선언
ability Console {
    fn print(msg: Text) -> Nil
    fn read() -> Text
}

// handler (구현)
fn run_console(comp: fn() ->{e, Console} a) ->{e, IO} a {
    handle comp() {
        do result { result }
        fn Console::print(msg) { IO::write(stdout, msg) }  // Nil 반환 → resume Nil
        fn Console::read() { IO::read(stdin) }              // Text 반환 → resume input
    }
}
```

`fn` handler arm에서는 `resume`을 사용할 수 **없다** (컴파일 에러).

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

### 기본 예시: State (op)

```rust
fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        do result { result }
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
    op fail(msg: Text) -> Never
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
fn run_console(comp: fn() ->{e, Console} a) ->{e, IO} a {
    handle comp() {
        do result { result }
        fn Console::print(msg) { IO::write(stdout, msg) }
        fn Console::read() { IO::read(stdin) }
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
    op get() -> s
    op set(value: s) -> Nil
}

fn counter() ->{Console, State(Int)} Nil {
    let n = State::get()
    Console::print("Count: " <> Int::to_string(n))
    State::set(n + 1)
}

fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() resume state, state) }
        op State::set(v) { run_state(fn() resume Nil, v) }
    }
}

fn run_console(comp: fn() ->{e, Console} a) ->{e, IO} a {
    handle comp() {
        do result { result }
        fn Console::print(msg) { IO::write(stdout, msg) }
        fn Console::read() { IO::read(stdin) }
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

### Operation 종류

| 선언 | resume | continuation 캡처 | handler arm | 용례 |
| ---- | ------ | ----------------- | ----------- | ---- |
| `fn print(msg: Text) -> Nil` | 1회, 암시적 | 없음 | `fn`만 가능 | Console, Reader, Logger |
| `op get() -> s` | 0~1회, `resume` (affine) | 있음 | `op` 또는 `fn` | State, Stream, Coroutine |
| `op fail(msg: Text) -> Never` | 0회 | 없음 | `op` 또는 `fn` | Fail, Exception |

### Operation 규칙

| 규칙 | 설명 |
| ---- | ---- |
| 암시적 effect | Operation은 자신이 속한 ability의 effect를 암시적으로 수행 |
| effect annotation 금지 | Operation 시그니처에 effect annotation을 적을 수 없음 |
| 다른 effect 필요 시 | Handler 구현부에서 처리 |
| `fn` 선언 보증 | `fn`으로 선언 → 모든 handler에서 `fn`으로 구현 필수 |
| `op` handler 유연성 | `op`으로 선언 → handler에서 `fn` 또는 `op`으로 구현 가능 |

### 전체 문법 요약

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
| `do value { expr }` | completion arm (handle 내) |
| `fn Op(args) { body }` | `fn` operation handler (handle 내) |
| `op Op(args) { body }` | `op` operation handler (handle 내) |
