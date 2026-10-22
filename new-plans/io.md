# Basic I/O

> 이 문서는 표준 입출력 ability `std::io::Read`, `std::io::Write`, `std::io::Io`와
> 기본 표준 입출력 API를 정의한다.

## Goals

- 외부 입출력을 ability row에서 추적한다.
- `print`, `print_line`, `read_line`을 일반 함수 API로 제공한다.
- 입출력도 다른 ability와 같은 규칙을 따른다. Compiler가 특별히 아는 ambient
  ability는 없으며, 표준 입출력 ability는 모두 패키지 `std`의 source에 선언한다.
- 일반 코드는 handle 없는 `Read`/`Write`를 요구한다. Handle을 다루는 `Io`는 그것을
  stream에 묶는 handler와 여러 stream을 함께 다루는 코드에만 나타난다.
- Native와 Wasm이 같은 source API와 같은 host 경계를 사용한다.

두 층의 구성 원칙은 [std-abilities.md](std-abilities.md)가, host interface의 일반
규칙은 [host-interface.md](host-interface.md)가 정한다.

## Public API

The `std::io` module exports:

```rust
pub ability Read {
    op read(max: Nat) -> Result(Bytes, Error)
}

pub ability Write {
    op write(bytes: Bytes) -> Result(Nat, Error)
}

pub ability Io {
    fn stdout() -> Option(OutStream)
    fn stderr() -> Option(OutStream)
    fn stdin() -> Option(InStream)
    fn write(out: OutStream, bytes: Bytes) -> Result(Nat, Error)
    fn read(input: InStream, max: Nat) -> Result(Bytes, Error)
    fn close_output(out: OutStream) -> Result(Nil, Error)
    fn close_input(input: InStream) -> Result(Nil, Error)
}

pub struct SystemError {
    code: Int
    message: String
}

pub enum Error {
    EndOfFile
    InvalidEncoding
    Closed
    System(SystemError)
}

pub fn write_all(bytes: Bytes) ->{Write} Result(Nil, Error)

pub fn to_output(out: OutStream, app: fn() ->{e, Write} a) ->{e, Io} a
pub fn from_input(input: InStream, app: fn() ->{e, Read} a) ->{e, Io} a
pub fn stdio(app: fn() ->{e, Read, Write} a) ->{e, Io} a

pub fn using_output(out: OutStream, body: fn(OutStream) ->{e} a) ->{e, Io} a
pub fn using_input(input: InStream, body: fn(InStream) ->{e} a) ->{e, Io} a

pub fn print(message: String) ->{Write} Nil
pub fn print_line(message: String) ->{Write} Nil
pub fn read_line() ->{Read, Throw(Error)} String
```

`OutStream`과 `InStream`은 [host resource](host-interface.md#host-resource) 타입이다.
Tribute source는 이 값을 만들 수 없고 `Io`의 operation 결과로만 얻는다. 선언 문법은
[host-interface.md](host-interface.md#미정-사항)의 미정 사항이다.

이 ability들은 일반 module export다. 사용자가 정의한 다른 module의 `ability Io {}`는
이름이 같아도 별개의 ability다.

API 이름은 Tribute의 일반 함수 naming을 따른다. 개행을 추가하는 함수는
`println`이 아니라 기존 이름인 `print_line`을 사용한다.

## `Read`와 `Write`

`Read`와 `Write`는 대상을 모르는 byte 입출력이다. 어디서 읽고 어디에 쓸지는 그
ability를 처리하는 handler가 정한다. `fn() ->{e, Write} a` 같은 계산은 그 자체로
출력을 만드는 생산자이며, 값처럼 넘기고 handler로 소비한다.

- `Write::write`는 `bytes`의 앞부분을 쓰고 쓴 byte 수를 돌려준다. 빈 `Bytes`가
  아니면 1 이상이다. `write_all`은 모두 쓸 때까지 반복한다.
- `Read::read`는 `max`바이트 이하를 돌려준다. 빈 `Bytes`는 입력의 끝이다. `max`가
  0이면 빈 `Bytes`를 돌려주며 끝을 뜻하지 않는다.
- 닫힌 대상은 `Err(Error::Closed)`, 그 밖의 실패는 `Err(Error::System)`이다.

두 ability의 operation은 `op`다. Handler는 resume하지 않아 생산자를 멈출 수 있고,
결과를 돌려주기 전에 다른 `op`를 수행하거나 기다릴 수 있다. 이 성질 때문에 조기
종료, 비동기 입출력, 실패의 `throw`를 같은 ability로 다룬다. 그 대가로 `Read`나
`Write`를 요구하는 함수는 중단될 수 있는 함수가 된다
([Calling Convention](#calling-convention)).

Handler는 대상을 바꾸거나 변환을 끼운다.

```rust
fn report(lines: List(String)) ->{Write} Nil { ... }   // 대상을 모른다

fn main() ->{Io} Nil {
    with stdio                  // Read/Write를 표준 stream에 묶는다
    report(["a", "b"])
}

fn archive(out: OutStream, lines: List(String)) ->{Io} Nil {
    with to_output(out)         // 이 stream에 쓴다
    with gzip                   // Write를 받아 압축해 바깥 Write로 보낸다
    report(lines)
}
```

`print`는 message를 UTF-8 `Bytes`로 바꿔 `write_all`로 쓴다. `print_line`은 끝에
`\n` 하나를 붙여 한 번에 쓴다. `read_line`은 [`read_line` Contract](#read_line-contract)를
따른다.

## `Io`

`Io`는 host resource handle로 표준 stream을 얻고 읽고 쓰는 ability다. Operation은
모두 `fn`이며 host가 처리하도록 설계한다.

- `stdout`, `stderr`, `stdin`은 host가 준 표준 stream을 돌려준다. Host가 그 stream을
  제공하지 않으면 `None`이다.
- `write`와 `read`는 `Write::write`, `Read::read`와 같은 계약을 따른다.
- `close_output`은 남은 출력을 flush한 뒤 stream을 끝내고, `close_input`은 입력
  stream을 끝낸다. 그 과정의 실패를 돌려준다. 이미 닫힌 stream에는 아무 일도 하지
  않고 `Ok`다.
- 닫힌 stream에 `write`나 `read`를 하면 host를 부르지 않고 `Err(Error::Closed)`다.

Stream의 수명은 [host resource의 수명](host-interface.md#수명)을 따른다. Close하지
않은 stream도 마지막 참조가 release되면 해제되지만 그 시점은 정하지 않는다.
`using_output`과 `using_input`은 `body`가 끝나면 stream을 close하고 `body`의 결과를
돌려준다. Close의 실패는 버리므로, 실패를 확인해야 하면 close operation을 직접
부른다.

이후 파일이나 소켓처럼 다른 ability가 돌려주는 stream도 같은 `OutStream`, `InStream`이며
같은 `Io` operation으로 읽고 쓴다.

### 묶는 handler

- `to_output(out, app)`은 `app`의 `Write::write`를 `Io::write(out, ..)`로 처리한다.
- `from_input(input, app)`은 `app`의 `Read::read`를 `Io::read(input, ..)`로 처리한다.
- `stdio(app)`은 `Read`를 stdin에, `Write`를 stdout에 묶는다. 표준 stream이 없을 때의
  동작은 [미정 사항](#미정-사항)이다.

이 handler들은 stream을 닫지 않는다. Stream의 수명은 handle을 가진 쪽이 정한다.

## Calling Convention

`Io`는 `fn` operation만 가지므로 `Io`만 요구하는 함수는 evidence를 받지만 중단되지
않는다. `Read`와 `Write`는 `op` operation을 가지므로, 이를 요구하는 함수는 중단될 수
있다. 각 경우의 호출 규약은 [CPS 전략](cps-effects.md#호출-규약)이 정한다. 묶는
handler의 arm은 `Io`의 `fn` operation만 부르므로, handler 안쪽의 host 호출은
continuation을 잡지 않는다.

## Entrypoint

`main`은 command 형태의 host가 부르는 [entrypoint](host-interface.md#entrypoint)이며
`Nil`을 반환해야 한다. `main`의 residual row는 host interface의 일부이며,
[host가 처리할 수 있는](host-interface.md#host가-처리하는-ability) ability만 남을 수 있다.

```rust
fn main() ->{Io} Nil {
    with stdio
    print_line("Hello")
}
```

`Write`도 그 조건을 지키므로 `main`에 남길 수 있다. 그때는 host가 `Write`를 직접
구현한다. `Write::write`는 `op`이므로 host interface에서 비동기 함수가 된다
([Host 경계](host-interface.md#host-경계)).

```rust
fn main() ->{Write} Nil {
    print_line("Hello")
}
```

Native와 Wasm entrypoint는 target에 맞는 initial evidence를 만들어 `main`에 전달한다.
현재 representation에서는 empty evidence로 충분하지만 이를 source-level ABI로 고정하지
않는다.

## `read_line` Contract

```rust
fn read_line() ->{Read, Throw(std::io::Error)} String
```

- 반환하는 문자열에서 line ending (`\n` 또는 `\r\n`)을 제거한다.
- 빈 줄은 빈 `String`을 정상 반환한다.
- EOF 전에 일부 문자를 읽었다면 line ending이 없어도 마지막 줄로 반환한다.
- 아무 문자도 읽지 못한 상태에서 EOF에 도달하면 `Error::EndOfFile`을 throw한다.
- 입력이 유효한 UTF-8이 아니면 `Error::InvalidEncoding`을 throw한다.
- OS 또는 host I/O 실패는 `Error::System`으로 throw한다.

EOF는 runtime failure와 같은 `Throw(Error)` 채널을 사용하므로 반환 타입을
`Option(String)`으로 만들지 않는다. `Abort`는 EOF를 나타내지 않는다.

## Host Boundary

표준 입출력 ability는 [host 경계](host-interface.md#host-경계)를 따라 target에
내려간다. Public API는 `extern "C"`, WASI import, runtime symbol 같은 target detail을
노출하지 않는다.

- `Io`의 operation은 WIT의 동기 함수가 되고, `OutStream`과 `InStream`은 그 interface의
  resource가 된다. Entrypoint에 남은 `Read`/`Write`의 operation은 비동기 함수가 된다.
- 이 interface는 WASI interface를 그대로 옮긴 것이 아니라 Tribute가 정한 계약이다.
  Wasm host는 이를 WASI의 표준 stream 위에 구현할 수 있다.
- Native에서는 `tribute-runtime`이 같은 interface를 C ABI 위에서 구현한다.
- 출력 buffering은 경계 아래, host 쪽에 둔다. Host는 쓰기를 모아 보낼 수 있고, close와
  program 종료 때 남은 출력을 flush한다. Tribute 쪽 handler는 buffering을 맡지 않는다.

## Validation

- `Read`, `Write`, `Io`가 일반 함수 호출을 통해 전파된다.
- `{Io}`만 요구하는 함수는 evidence를 받지만 중단되지 않는다. `{Write}`를 요구하는
  함수는 중단될 수 있다.
- `main ->{Io} Nil`과 `main ->{Write} Nil`은 허용된다.
- 사용자가 정의한 같은 이름의 ability는 `std::io`의 ability와 구별된다.
- `Write`를 처리하는 Tribute handler 아래에서 `print_line`은 host에 닿지 않는다.
- `Write` handler가 resume하지 않으면 생산자는 그 지점에서 멈춘다.
- 닫힌 stream의 `write`와 `read`는 host를 부르지 않고 `Err(Error::Closed)`다.
- 두 번째 close는 `Ok`이고, 닫힌 stream의 마지막 release는 drop을 다시 부르지 않는다.
- `print`는 개행을 추가하지 않고 `print_line`은 정확히 하나 추가한다.
- `read_line`은 빈 줄, 마지막 unterminated line, EOF, invalid UTF-8, system error를
  구분한다.
- Native와 Wasm 모두 동적 `String` I/O를 처리한다.

## 미정 사항

- **출력 실패와 표준 stream이 없을 때.** `print`와 `print_line`은 실패를 돌려주지
  않는다. `Write::write`가 실패하거나 `stdio`가 stdout을 얻지 못했을 때 출력을
  버릴지, trap할지, `print`가 `Throw(Error)`를 요구할지 정해야 한다.
- **표준 오류 출력.** 한 row에는 `Write`가 한 번만 나타날 수 있다. stdout과 stderr에
  함께 쓰는 코드를 `Write` 층에서 어떻게 표현할지 정해야 한다. 이름 붙인 instance
  ([중복 Label 금지](type-inference.md#중복-label-금지))나 별도 ability가 후보다.
  같은 문제는 파일 두 개에 함께 쓰는 코드에도 있으며, 지금은 `Io`를 직접 쓴다.
- **줄 단위 입력.** `read_line`은 한 줄보다 많이 읽은 byte를 다음 호출에 남겨야 한다.
  그 byte를 `Read` handler가 보관할지, 줄 단위 입력을 위한 ability나 handler를 따로
  둘지 정해야 한다.
- **범용 stream ability.** 원소 타입으로 일반화한 `Emit(a)`, `Ask(a)`를 두고 byte
  입출력을 그 instance로 볼지 정해야 한다.
