# Basic I/O

> 이 문서는 표준 입출력 host ability `std::io::Io`, 그 위의 일반 ability
> `std::io::Console`, 기본 표준 입출력 API를 정의한다.

## Goals

- 외부 입출력을 ability row에서 추적한다.
- `print`, `print_line`, `read_line`을 일반 함수 API로 제공한다.
- 입출력도 다른 ability와 같은 규칙을 따른다. Compiler가 특별히 아는 ambient
  ability는 없으며, `Io`와 `Console`은 모두 prelude source에 선언한 ability다.
- 일반 코드는 handle 없는 `Console`을 요구하고, host 권한인 `Io`는 handler를
  설치하는 곳에만 나타난다.
- Native와 Wasm이 같은 source API와 같은 host 경계를 사용한다.

두 층의 구성 원칙은 [std-host-abilities.md](std-host-abilities.md)가, host
ability의 일반 규칙은 [host-abilities.md](host-abilities.md)가 정한다.

## Public API

The `std::io` module exports:

```rust
// host ability (표시 문법은 host-abilities.md의 미정 사항)
pub ability Io {
    fn stdout() -> Option(OutStream)
    fn stderr() -> Option(OutStream)
    fn stdin() -> Option(InStream)
    fn write(out: OutStream, bytes: Bytes) -> Result(Nil, StreamError)
    fn read(input: InStream, max: Nat) -> Result(Bytes, StreamError)
}

pub ability Console {
    fn write(bytes: Bytes) -> Nil
    fn read_line() -> Result(String, Error)
}

pub struct SystemError {
    code: Int
    message: String
}

pub enum StreamError {
    Closed
    System(SystemError)
}

pub enum Error {
    EndOfFile
    InvalidEncoding
    System(SystemError)
}

pub fn console(app: fn() ->{e, Console} a) ->{e, Io} a

pub fn print(message: String) ->{Console} Nil
pub fn print_line(message: String) ->{Console} Nil
pub fn read_line() ->{Console, Throw(Error)} String
```

`OutStream`과 `InStream`은 host resource 타입이다
([Host Resource](std-host-abilities.md#host-resource)). Tribute source는 이 값을
만들 수 없고 `Io`의 operation 결과로만 얻는다.

`Io`와 `Console`은 일반 module export다. 사용자가 정의한 다른 module의
`ability Io {}`는 이름이 같아도 별개의 ability다.

API 이름은 Tribute의 일반 함수 naming을 따른다. 개행을 추가하는 함수는
`println`이 아니라 기존 이름인 `print_line`을 사용한다.

## `Io` Host Ability

`Io`는 표준 stream을 얻고 stream을 읽고 쓰는 host ability다. Host가 처리하며,
Tribute handler로 가로챌 수도 있다([Host Ability](host-abilities.md#host-ability)).

- `stdout`, `stderr`, `stdin`은 host가 준 표준 stream을 돌려준다. Host가 그 stream을
  제공하지 않으면 `None`이다.
- `write`는 `bytes`를 모두 쓰거나 실패한다. 일부만 쓴 결과는 돌려주지 않는다.
- `read`는 1바이트 이상 `max`바이트 이하를 돌려준다. 더 읽을 것이 없으면
  `Err(StreamError::Closed)`다. `max`가 0이면 빈 `Bytes`를 돌려준다.
- 그 밖의 host 실패는 `StreamError::System`이다.

이후 파일처럼 다른 host ability가 돌려주는 stream도 같은 `write`와 `read`로
다룬다.

## `Console`과 `console` Handler

`Console`은 handle이 없는 표준 입출력 ability다. `print`, `print_line`, `read_line`은
`Console`만 요구하므로, 이를 쓰는 코드는 host 권한을 갖지 않는다.

- `print`는 message를 UTF-8 `Bytes`로 바꿔 `Console::write`를 호출한다.
  `print_line`은 끝에 `\n` 하나를 붙여 한 번 호출한다.
- `read_line`은 `Console::read_line`의 `Err`를 `Throw(Error)`로 바꾼다.

`console`은 `Console`을 `Io`로 구현하는 표준 handler다. 설치할 때 표준 stream을
한 번 얻어 두고, `Console::write`는 stdout에, `Console::read_line`은 stdin에서 읽은
한 줄로 처리한다.

테스트나 다른 출력 대상이 필요한 코드는 `Console`을 처리하는 handler를 직접
씌운다. `Console`의 handler는 `Io`를 요구하지 않아도 된다.

## Calling Convention

`Io`와 `Console`은 `fn` operation만 가지므로, 둘 중 하나만 요구하는 함수는
`EvidenceDirect` 규약을 쓴다. `read_line`처럼 `Throw(Error)`를 함께 요구하면 더
강한 CPS 규약을 쓴다. 규약은 [CPS 전략](cps-effects.md#호출-규약)이 정한다.

## Entrypoint

`main`은 `Nil`을 반환해야 한다. `main`의 residual row에는
[host ability](host-abilities.md#host-ability)만 남을 수 있다. 처리되지 않은
`Throw(Error)`나 host ability가 아닌 ability가 남으면 frontend가 backend lowering
전에 거부한다.

```rust
fn main() ->{Io} Nil {
    with console
    print_line("Hello")
}
```

`Console`은 host ability가 아니므로 `main`이 처리하지 않고 남기면 거부한다.
`main`은 `console` 같은 handler로 `Console`을 처리하고 `Io`를 host interface로
남긴다.

Native와 Wasm entrypoint는 target에 맞는 initial evidence를 만들어
`EvidenceDirect` main에 전달한다. 현재 representation에서는 empty evidence로
충분하지만 이를 source-level ABI로 고정하지 않는다.

## `read_line` Contract

```rust
fn read_line() ->{Console, Throw(std::io::Error)} String
```

`Console::read_line`을 처리하는 handler는 다음 계약을 지킨다. `console` handler도
마찬가지다.

- 반환하는 문자열에서 line ending (`\n` 또는 `\r\n`)을 제거한다.
- 빈 줄은 빈 `String`을 정상 반환한다.
- EOF 전에 일부 문자를 읽었다면 line ending이 없어도 마지막 줄로 반환한다.
- 아무 문자도 읽지 못한 상태에서 EOF에 도달하면 `Error::EndOfFile`이다.
- 입력이 유효한 UTF-8이 아니면 `Error::InvalidEncoding`이다.
- OS 또는 host I/O 실패는 `Error::System`이다.

EOF는 runtime failure와 같은 `Throw(Error)` 채널을 사용하므로 `read_line`의 반환
타입을 `Option(String)`으로 만들지 않는다. `Abort`는 EOF를 나타내지 않는다.

## Host Boundary

`Io`는 [host 경계](host-abilities.md#host-경계)를 따라 target에 내려간다. Public
API는 `extern "C"`, WASI import, runtime symbol 같은 target detail을 노출하지
않는다.

- `Io`의 operation은 다른 host ability와 같은 방식으로 WIT interface의 함수가 되고,
  `OutStream`과 `InStream`은 그 interface의 resource가 된다.
- `Io`는 WASI interface를 그대로 옮긴 것이 아니라 Tribute가 정한 계약이다. Wasm
  host는 이 interface를 WASI의 표준 stream 위에 구현할 수 있다.
- Native에서는 `tribute-runtime`이 같은 interface를 C ABI 위에서 구현한다.

## Validation

- `Io`와 `Console`이 일반 함수 호출을 통해 전파된다.
- `{Console}`만 요구하는 함수와 호출은 `EvidenceDirect` ABI를 사용한다.
- `{Console, Throw(Error)}`는 CPS ABI로 승격된다.
- `main ->{Io} Nil`은 허용되고, `Console`이나 다른 non-host ability가 남은 `main`은
  거부된다.
- 사용자가 정의한 같은 이름의 ability는 `std::io`의 ability와 구별된다.
- `Console`을 처리하는 Tribute handler 아래에서 `print_line`은 host에 닿지 않는다.
- `print`는 개행을 추가하지 않고 `print_line`은 정확히 하나 추가한다.
- `read_line`은 빈 줄, 마지막 unterminated line, EOF, invalid UTF-8, system error를
  구분한다.
- Native와 Wasm 모두 동적 `String` I/O를 처리한다.

## 미정 사항

- **표준 stream이 없거나 쓰기가 실패할 때.** `Io::stdout()`이 `None`이거나
  `Io::write`가 실패했을 때 `console` handler가 출력을 버릴지, trap할지 정해야 한다.
  `Console::write`는 실패를 돌려주지 않으므로 이 선택은 handler에 있다.
- **표준 오류 출력.** `Console`에 stderr로 쓰는 operation을 둘지 정해야 한다.
- **줄 단위 입력의 구현.** `console` handler는 `Io::read`의 byte 단위 결과에서 줄을
  나눠야 한다. 호출 사이에 남은 byte를 보관할 handler 상태를 둘지, `Io`에 줄 단위
  operation을 둘지 정해야 한다.
