# Standard Host Abilities

> 이 문서는 표준 라이브러리가 제공하는 host ability의 구성 원칙과, 그 위에 두는
> 일반 ability 층을 정한다. Host ability 자체의 규칙은
> [host-abilities.md](host-abilities.md)가, 표준 입출력 API의 세부는
> [io.md](io.md)가 정한다.

현재 구현과 target 지원 범위는 [capabilities.md](capabilities.md)가 기록한다. 이
문서의 내용은 설계 의도이며 지원을 주장하지 않는다.

## 목표

- 바깥 세계에 닿는 표준 기능을 모두 prelude source에 선언한 host ability로
  제공한다. Compiler가 특별히 아는 ambient ability는 두지 않는다.
- 대부분의 코드는 host ability가 아니라 그 위의 일반 ability를 요구한다. 그래서
  library 코드는 host 권한을 직접 갖지 않고, 테스트는 host 없이 일반 handler
  하나로 대체할 수 있다.
- Host 권한이 정해지는 곳은 그 일반 ability의 handler를 설치하는 곳, 보통
  `main`이다. `main`의 signature가 곧 program의 host interface다
  ([Host 경계](host-abilities.md#host-경계)).

## 두 층

표준 기능은 두 층으로 나눈다.

| 층 | 예 | 성질 |
| --- | --- | --- |
| Host ability | `std::io::Io` | host가 처리한다. Host resource handle을 다룬다. Operation은 실패를 `Result`로 돌려준다. |
| 일반 ability | `std::io::Console` | handle이 없는 고수준 operation이다. 표준 handler가 host ability로 구현한다. |

일반 ability 층의 표준 handler는 그 일반 ability를 요구하는 계산을 받아 host
ability를 요구하는 계산으로 바꾸는 일반 함수다.

```rust
pub fn console(app: fn() ->{e, Console} a) ->{e, Io} a
```

Program은 `main`에서 이 handler를 설치하고, 나머지 코드는 일반 ability만 적는다.

```rust
use std::io::{Io, Console, console, print_line}

fn main() ->{Io} Nil {
    with console
    greet("Tribute")
}

fn greet(name: String) ->{Console} Nil {
    print_line("Hello, \{name}")
}
```

`with`는 block의 나머지를 람다로 감싸 마지막 인자로 넘기는 문법이다
([With Statement](syntax.md#with-statement)). 위 `main`의 본문은
`console(fn() { greet("Tribute") })`와 같다.

테스트는 `Console`을 처리하는 handler를 직접 씌운다. 이때 `Io`는 등장하지 않는다.

```rust
fn capture(app: fn() ->{e, Console} a) ->{e} #(a, List(Bytes))
```

## Host Resource

Host ability의 operation은 host가 소유한 object를 가리키는 **host resource** 값을
주고받는다. 표준 입출력의 `OutStream`과 `InStream`이 그 예다.

- **Host만 만든다.** Tribute source는 host resource 값을 만들 수 없다. 값은 host
  ability operation의 결과로만 생긴다. 그래서 program 안의 모든 host resource는
  host의 handle 공간 하나에 속한다.
- **Tribute handler는 handle을 만들지 않는다.** Host ability를 처리하는 Tribute
  handler는 받은 handle을 바깥 handler로 넘기거나, 실패를 돌려주거나, 호출을
  관찰할 수 있다. 가짜 handle을 만들 수 없으므로 handle이 그것을 모르는 handler에
  닿는 일이 없다.
- **같은 handle을 여러 host ability가 쓴다.** 표준 출력에 쓰는 연산과 이후 파일에
  쓰는 연산은 같은 `OutStream`과 같은 `Io::write`를 쓴다. 출력 대상은 값으로
  넘긴다.

Host resource 타입의 선언 문법, 수명과 해제, 경계에서의 표현은
[host-abilities.md](host-abilities.md#미정-사항)의 미정 사항이다.

## 표준 Host Ability

| Ability | Module | 내용 |
| --- | --- | --- |
| `Io` | `std::io` | 표준 stream 획득과 stream 읽기·쓰기 ([io.md](io.md#io-host-ability)) |

표준 stream 획득은 `Io`에 함께 둔다. 표준 stream은 권한이 낮고 command 형태의
host가 항상 제공하므로 따로 나누지 않는다. Host가 표준 stream을 주지 않을 수
있으므로 획득 operation은 `Option`을 돌려준다.

파일 시스템처럼 권한이 큰 기능은 별도 host ability로 둔다. 그 ability도 host가
만든 handle을 돌려주고, stream 읽기·쓰기는 `Io`의 operation을 그대로 쓴다.

## 실패 표현

Host ability operation의 시그니처에는 다른 ability를 적을 수 없다
([Operation의 Ability Row 규칙](abilities.md#operation의-ability-row-규칙)). 그래서
host operation은 실패를 `Result`로 돌려준다. 표준 라이브러리의 wrapper 함수가
이를 `Throw`로 바꾸며, 호출 지점에서 잡을 수 있다.

```rust
pub fn read_line() ->{Console, Throw(Error)} String
```

## 미정 사항

- **Tribute가 만드는 resource.** Tribute handler가 자기 handle을 만들어 host
  ability를 흉내 내는 경우다. 이때는 같은 handle 공간에 여러 handler가 관여하므로,
  handle을 만든 handler와 쓰는 handler가 어긋나지 않게 해야 한다. Ability와 그
  resource를 scope 타입 매개변수로 묶고(`Io(s)`, `OutStream(s)`), handler가 fresh한
  `s`를 도입하는 방식을 검토한다. 그러면 operation은 label로 자기 handler에 가고,
  handle이 handler 범위를 벗어나는 것은 타입 오류가 된다. 여러 ability가 같은
  handle 공간을 쓴다는 선언(`ability Stdio(s) requires Io(s)`)도 이 범위에 둔다.
- **Default handler.** `main`까지 처리되지 않은 일반 ability에 entrypoint가 표준
  handler를 씌우는 선언이다. `main`의 host interface가 signature와 달라지고 handler
  설치 순서를 정해야 하므로 지금은 두지 않는다.
- **파일 시스템.** Host가 준 directory handle을 통해서만 파일에 닿는 구조(`open_at`,
  preopen)를 검토한다. 전역 root는 두지 않는다.
- **Terminal.** 표준 stream과 별도로, 없을 수 있는 terminal 기능을 둘지 정해야 한다.
- **그 밖의 후보.** 시계, 난수, 환경 변수와 인자처럼 표준으로 제공할 host ability의
  목록이다.
