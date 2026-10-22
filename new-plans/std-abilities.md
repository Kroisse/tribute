# Standard Abilities

> 이 문서는 표준 라이브러리가 바깥 세계에 닿는 기능을 어떤 ability로 나누어
> 제공하는지 정한다. Host interface의 규칙은 [host-interface.md](host-interface.md)가,
> 표준 입출력 API의 세부는 [io.md](io.md)가 정한다.

현재 구현과 target 지원 범위는 [capabilities.md](capabilities.md)가 기록한다. 이
문서의 내용은 설계 의도이며 지원을 주장하지 않는다.

## 목표

- 바깥 세계에 닿는 표준 기능을 모두 prelude source에 선언한 ability로 제공한다.
  Compiler가 특별히 아는 ambient ability는 두지 않는다.
- 대부분의 코드는 handle 없는 ability를 요구한다. 그래서 library 코드는 어떤 파일이나
  stream에 닿을지 스스로 정하지 않고, 테스트는 host 없이 handler 하나로 대체할 수 있다.
- Host 권한이 정해지는 곳은 handle을 다루는 ability가 처음 나타나는 곳, 보통 `main` 같은
  entrypoint다. Entrypoint의 signature가 곧 그 entrypoint가 요구하는 host 기능이다
  ([Entrypoint](host-interface.md#entrypoint)).

## 두 층

표준 기능은 두 층으로 나눈다. 두 층 모두 일반 ability 선언이며, host가 처리하는지는
선언이 아니라 entrypoint에 남느냐로 정해진다
([Host가 처리하는 ability](host-interface.md#host가-처리하는-ability)).

| 층 | 예 | 성질 |
| --- | --- | --- |
| 대상 없는 층 | `std::io::Read`, `std::io::Write` | handle이 없다. Operation은 `op`다. 대상은 handler가 정한다. |
| Handle 층 | `std::io::Io` | [host resource](host-interface.md#host-resource) handle을 다룬다. Operation은 `fn`이며 host가 처리하도록 설계한다. |

두 층은 묶는 handler로 잇는다. 묶는 handler는 대상 없는 ability를 요구하는 계산을
받아, 그 operation을 handle 층의 operation으로 처리한다.

```rust
pub fn to_output(out: OutStream, app: fn() ->{e, Write} a) ->{e, Io} a
pub fn stdio(app: fn() ->{e, Read, Write} a) ->{e, Io} a
```

```rust
use std::io::{Io, Write, stdio, print_line}

fn main() ->{Io} Nil {
    with stdio
    greet("Tribute")
}

fn greet(name: String) ->{Write} Nil {
    print_line("Hello, \{name}")
}
```

`with`는 block의 나머지를 람다로 감싸 마지막 인자로 넘기는 문법이다
([With Statement](syntax.md#with-statement)). 위 `main`의 본문은
`stdio(fn() { greet("Tribute") })`와 같다.

테스트는 `Write`를 처리하는 handler를 직접 씌운다. 이때 `Io`는 등장하지 않는다.

```rust
fn capture(app: fn() ->{e, Write} a) ->{e} #(a, Bytes)
```

Host가 대상 없는 ability를 직접 구현할 수 있다면 entrypoint가 그 ability를 그대로 남겨도
된다(`fn main() ->{Write} Nil`).

### 대상 없는 층의 operation이 `op`인 이유

`fn` operation의 handler arm은 continuation을 잡지 못하므로, 그 arm 안에서는 계산을
멈추거나 기다리거나 `throw`할 수 없다. 대상 없는 층은 다음을 handler로 표현하려고
`op`를 쓴다.

- **조기 종료.** 소비자 handler가 resume하지 않으면 생산자는 그 지점에서 멈춘다.
- **비동기.** Handler가 결과를 돌려주기 전에 다른 `op`를 수행해 기다릴 수 있다.
  Entrypoint에 남은 `op`는 host interface에서 비동기 함수가 된다.
- **변환.** 압축이나 tee처럼 받은 쓰기를 고쳐 바깥 `Write`로 보내는 handler를 쌓는다.

대가로 대상 없는 층을 요구하는 함수는 CPS 규약을 쓴다. Handle 층은 `fn`이므로 묶는
handler 안쪽의 host 호출은 continuation을 잡지 않는다.

## Handle 층

Handle 층의 operation은 host resource 값을 주고받는다. 표준 입출력의 `OutStream`과
`InStream`이 그 예다. Host resource는 host만 만들므로 handle이 그것을 모르는 handler에
닿는 일이 없고, 수명은 drop과 close가 정한다([수명](host-interface.md#수명)).

Handle 층의 타입은 구체 타입이다. Test가 가짜 대상을 만들 때는 handle 층이 아니라
대상 없는 층을 대체한다.

같은 stream 타입을 여러 ability가 쓴다. 표준 stream과 이후 파일, 소켓은 모두
`OutStream`과 `InStream`을 돌려주며, 읽기와 쓰기는 `Io`의 operation을 그대로 쓴다.
파일이나 소켓처럼 읽고 쓰는 resource는 stream 자체가 아니라 stream을 내주는 resource다.
그 resource의 ability가 읽기용과 쓰기용 stream을 따로 돌려주며, 각 stream은 따로 닫힌다
([자식 resource](host-interface.md#자식-resource)).

## 표준 Ability

| Ability | Module | 층 | 내용 |
| --- | --- | --- | --- |
| `Read` | `std::io` | 대상 없음 | byte 입력 ([io.md](io.md#read와-write)) |
| `Write` | `std::io` | 대상 없음 | byte 출력 ([io.md](io.md#read와-write)) |
| `Io` | `std::io` | handle | 표준 stream 획득과 stream 읽기·쓰기·close ([io.md](io.md#io)) |

표준 stream 획득은 `Io`에 함께 둔다. 표준 stream은 권한이 낮고 command 형태의 host가
항상 제공하므로 따로 나누지 않는다. Host가 표준 stream을 주지 않을 수 있으므로 획득
operation은 `Option`을 돌려준다.

파일 시스템처럼 권한이 큰 기능은 별도 ability로 둔다. 그 ability도 host가 만든 handle을
돌려주고, stream 읽기·쓰기는 `Io`의 operation을 그대로 쓴다.

## 실패 표현

Operation 시그니처에는 다른 ability를 적을 수 없다
([Operation의 Ability Row 규칙](abilities.md#operation의-ability-row-규칙)). 그래서
operation은 실패를 `Result`로 돌려준다. 표준 라이브러리의 wrapper 함수가 이를
`Throw`로 바꾸며, 호출 지점에서 잡을 수 있다.

```rust
pub fn read_line() ->{Read, Throw(Error)} String
```

## 미정 사항

- **Handle 층의 가짜 resource.** Test handler가 자기 stream을 만들어 `Io`를 흉내 내는
  경우다. 대상 없는 층을 대체하면 대부분 필요 없다. 조건과 후보는
  [host-interface.md](host-interface.md#미정-사항)에 있다.
- **Default handler.** Entrypoint까지 처리되지 않은 ability에 표준 handler를 씌우는
  선언이다. Host interface가 entrypoint의 signature와 달라지고 handler 설치 순서를
  정해야 하므로 지금은 두지 않는다.
- **파일 시스템.** Host가 준 directory handle을 통해서만 파일에 닿는 구조(`open_at`,
  preopen)를 검토한다. 전역 root는 두지 않는다. 파일 stream의 offset을 정하는 방식도
  함께 정한다.
- **Terminal.** 표준 stream과 별도로, 없을 수 있는 terminal 기능을 둘지 정해야 한다.
- **그 밖의 후보.** 시계, 난수, 환경 변수와 인자처럼 표준으로 제공할 ability의 목록이다.
- **Thunk 타입 축약.** `fn() ->{e} a`를 줄여 쓰는 문법이다. 새 문법보다 row 매개변수를
  받는 타입 alias로 해결하는 쪽을 검토한다.
