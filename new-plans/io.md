# Basic I/O

> 이 문서는 Tribute의 builtin ambient `Io` ability와 기본 표준 입출력 API를
> 정의한다.

## Goals

- 외부 입출력을 effect row에서 추적한다.
- `print`, `print_line`, `read_line`을 일반 함수 API로 제공한다.
- 런타임 경계인 `Io`와 사용자가 처리할 수 있는 algebraic ability를 구분한다.
- Native와 Wasm이 같은 source API와 target-independent I/O boundary를 사용한다.

표준 라이브러리는 별도의 `Console` ability를 제공하지 않는다. 테스트 가능한
추상화가 필요한 프로그램은 `Logger`, `Terminal` 같은 application-specific
ability를 정의하고 handler에서 `std::io` 함수를 호출할 수 있다.

## Public API

The `std::io` module exports:

```rust
pub ability Io {}

pub struct SystemError {
    code: Int
    message: String
}

pub enum Error {
    EndOfFile
    InvalidEncoding
    System(SystemError)
}

pub fn print(message: String) ->{Io} Nil
pub fn print_line(message: String) ->{Io} Nil
pub fn read_line() ->{Io, Throw(Error)} String
```

API 이름은 Tribute의 일반 함수 naming을 따른다. 개행을 추가하는 함수는
`println`이 아니라 기존 이름인 `print_line`을 사용한다.

## `Io`: Builtin Ambient Ability

`std::io::Io`는 operation이 없는 compiler-owned builtin ability다. 타입
시스템은 일반 effect label처럼 전파하고 effect annotation과 diagnostic에
표시한다.

```rust
fn greet(name: String) ->{Io} Nil {
    print("Hello, ")
    print_line(name)
}
```

`Io`의 ambient 성질은 다음과 같다.

- 사용자는 `Io`를 handler로 제거할 수 없다.
- `main`에 처리되지 않은 채 남을 수 있는 terminal effect다.
- `Io`를 요구하는 함수는 runtime/host I/O boundary를 호출할 수 있다.
- ambient ability를 선언하는 사용자 문법은 제공하지 않는다.
- 사용자가 정의한 `ability Io {}`는 별개의 일반 ability이며 ambient가 아니다.

Compiler는 이름 문자열만 비교하지 않고 canonical builtin identity
`std::io::Io`를 등록하여 이 성질을 판정한다.

`Io`의 semantic identity는 source ability identity와 구분되는 compiler-owned
builtin discriminator를 가진다. Resolver는 `std::io::Io`를 virtual builtin binding으로
노출하므로 `use std::io::Io`와 qualified effect annotation은 일반 module export처럼
동작하지만, 사용자 AST에 synthetic `ability` declaration을 삽입하지 않는다. 따라서
사용자가 선언한 `ability Io {}`는 이름이 같아도 builtin metadata를 얻지 않는다.

이 virtual binding은 아직 file-based module loader를 요구하지 않는다. 이후 기본 I/O
함수와 오류 타입은 embedded standard-library source로 제공할 수 있지만, `Io`의 ambient
성질은 source declaration이 아니라 builtin identity에 연결한다. Builtin identity는
frontend 판정용이며 runtime ability ID나 Evidence marker를 암시하지 않는다.

## Calling Convention

함수 effect와 호출 규약은 다음 세 단계로 분류한다.

| Convention | Parameters | Result | Requirement |
| ---------- | ---------- | ------ | ----------- |
| `Direct` | source parameters | source result | closed empty effect row or pure named worker |
| `EvidenceDirect` | evidence + source parameters | source result | `Io` or tail-resumptive `fn` effect |
| `Cps` | evidence + `done_k` + source parameters | source result를 직접 반환하지 않음 | general `op`, `Throw`, or another CPS effect |

규약을 합성할 때 `Direct < EvidenceDirect < Cps` 순서로 더 강한 규약이
우선한다.

명시적인 빈 effect annotation `->{}`는 닫힌 빈 effect row를 뜻하므로
`Direct`를 사용한다. Effect annotation을 생략한 함수와 달리 effect-polymorphic하지
않지만, 빈 row 자체는 evidence parameter를 요구하지 않는다.

Effect annotation을 생략한 `fn(...) -> T`의 semantic type은
`fn(...) ->{e} T`다. 열린 row의 간접 호출은 `Cps`지만, 정의에서 concrete residual
effect가 발견되지 않으면 해당 named definition은 별도의 `Direct` worker를 가질 수
있다.

`Io`만 요구하는 함수는 현재 `EvidenceDirect`를 사용한다. Evidence는 전달되지만
`Io` operation을 dispatch하기 위한 handler lookup은 수행하지 않는다. Evidence의
concrete representation과 ambient runtime capability 저장 여부는 backend 구현
세부사항이다. Ambient-only 함수에서 evidence를 완전히 제거하는 최적화는 후속
작업이다.

```text
print_line(ev, message) -> Nil
main(ev) -> Nil
```

`read_line`은 실패 시 `Throw(Error)`를 수행하므로 CPS 규약을 사용한다. 논리적
ABI에서 `done_k`와 함수의 control result는 `Never`이며, 현재 compatibility
lowering만 이를 `anyref` carrier로 표현한다.

```text
read_line(ev, done_k: fn(String) -> Never) -> Never
```

## Entrypoint

`main`은 `Nil`을 반환해야 한다. 모든 residual effect는 오류이지만 builtin
`std::io::Io`만 예외적으로 허용한다.

```rust
fn main() ->{Io} Nil {
    print_line("Hello")
}
```

Native와 Wasm entrypoint는 target에 맞는 initial evidence를 만들어
`EvidenceDirect` main에 전달한다. 현재 representation에서는 empty evidence로
충분하지만 이를 source-level ABI로 고정하지 않는다. 처리되지 않은 `Throw(Error)`나
사용자 ability가 남아 있으면 frontend가 backend lowering 전에 거부한다.

## `read_line` Contract

```rust
fn read_line() ->{Io, Throw(std::io::Error)} String
```

- 반환하는 문자열에서 line ending (`\n` 또는 `\r\n`)을 제거한다.
- 빈 줄은 빈 `String`을 정상 반환한다.
- EOF 전에 일부 문자를 읽었다면 line ending이 없어도 마지막 줄로 반환한다.
- 아무 문자도 읽지 못한 상태에서 EOF에 도달하면 `Error::EndOfFile`을 throw한다.
- 입력이 유효한 UTF-8이 아니면 `Error::InvalidEncoding`을 throw한다.
- OS 또는 host I/O 실패는 `Error::System`으로 변환한다.

EOF는 runtime failure와 같은 `Throw(Error)` 채널을 사용하므로 반환 타입을
`Option(String)`으로 만들지 않는다. `Abort`는 EOF를 나타내지 않는다.

## Runtime Boundary

Public API는 `extern "C"`, WASI import, `__print_line` 같은 target detail을
노출하지 않는다. Shared lowering은 동적 `Bytes`를 읽고 쓸 수 있는
target-independent I/O boundary를 생성한다.

- Native lowering은 `tribute-runtime`의 stdin/stdout ABI로 변환한다.
- Wasm lowering은 현재 backend가 지원하는 WASI 또는 custom host import로 변환한다.
- WASI preview2/component model 전환은 이 source API와 독립적인 후속 backend
  작업이다.

`String` 출력은 rope를 `Bytes` chunk로 순회하거나 flatten하여 처리할 수 있지만,
source-level `Io` 계약과 target-independent boundary는 어느 표현을 선택하든
동일해야 한다. Wasm도 문자열 literal뿐 아니라 동적으로 만든 `String`을 출력할
수 있어야 한다.

## Validation

- `Io`가 일반 함수 호출을 통해 전파된다.
- `{Io}` 함수와 호출은 `EvidenceDirect` ABI를 사용한다.
- `{Io, Throw(Error)}`는 CPS ABI로 승격된다.
- `main ->{Io} Nil`은 허용되고 다른 residual ability는 거부된다.
- 사용자가 정의한 같은 이름의 ability는 ambient로 취급되지 않는다.
- `print`는 개행을 추가하지 않고 `print_line`은 정확히 하나 추가한다.
- `read_line`은 빈 줄, 마지막 unterminated line, EOF, invalid UTF-8, system error를
  구분한다.
- Native와 Wasm 모두 동적 `String` I/O를 처리한다.
