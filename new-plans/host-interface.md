# Host Interface

> 이 문서는 Tribute 프로그램이 바깥 세계와 만나는 경계를 정한다. 외부 함수
> 호출(FFI)은 따로 두지 않고, host가 handler를 공급하는 ability로 표현한다.

현재 구현과 target 지원 범위는 [capabilities.md](capabilities.md)가 기록한다. 이
문서의 내용은 설계 의도이며 지원을 주장하지 않는다.

## 목표

- 프로그램이 바깥 세계에 할 수 있는 일을 ability row로 드러낸다. 함수 시그니처를
  읽으면 그 코드가 닿을 수 있는 host 기능을 알 수 있다.
- Host가 공급하는 기능을 Tribute 쪽에서 handler로 가로챌 수 있게 한다. 테스트는
  host 없이 handler로 대신하고, 신뢰하지 않는 코드에는 더 좁은 handler를 씌운다.
- 브라우저, plugin host, 단독 실행 launcher가 같은 경계를 쓴다. 이들의 차이는 누가
  handler를 공급하느냐뿐이다([design.md](design.md#지향점)).

## Host가 처리하는 ability

프로그램 안에서 처리되지 않으면 host가 처리하는 ability가 있다.
[Entrypoint](#entrypoint)에 처리되지 않은 채 남은 그런 ability들이 프로그램의 host
interface다.

```rust
ability Clock {
    fn now_millis() -> Int
}

ability Kv {
    fn get(key: String) -> Option(Bytes)
    fn put(key: String, value: Bytes) -> Nil
}
```

Ability를 host가 처리할 수 있다고 표시하는 선언 문법은 아래 [미정 사항](#미정-사항)에
둔다. 표시 외에는 일반 ability와 같은 규칙을 따른다.

- **Operation은 모두 `fn`이다.** Host 경계는 continuation을 넘기지 않는다. 선언부가
  보증하므로([Operation 종류](abilities.md#operation-종류)) host가 처리하는 operation의
  호출은 continuation을 잡지 않는다. `op` operation을 가진 ability는 host가 처리할 수
  없다.
- **Tribute handler가 먼저 처리한다.** 이 operation도 가장 가까운 handler로
  dispatch된다. 프로그램 안에 handler가 없을 때만 host에 닿는다. 그래서 이런
  ability를 쓰는 코드는 host 없이도 handler를 씌워 실행하고 테스트할 수 있다.
- **Entrypoint의 residual row에 남을 수 있다.** Entrypoint에 처리되지 않은 채 남은
  ability들이 그 프로그램이 host에 요구하는 기능의 전부다. Host가 처리할 수 없는
  ability가 남으면 frontend가 거부한다.

## Entrypoint

Entrypoint는 host가 부르는 Tribute 함수다. `main`은 command 형태의 host가 부르는
entrypoint의 한 경우다. Plugin이나 embedding host는 여러 entrypoint를 부를 수 있다.

- 각 entrypoint의 residual row에는 host가 처리하는 ability만 남을 수 있다.
- 프로그램의 host interface는 모든 entrypoint의 residual row를 합친 것이다. 각
  entrypoint의 signature는 그 entrypoint가 닿는 host 기능을 따로 보여 준다.
- Entrypoint의 row는 본문에서 추론하지 않고 signature에 적는다. 그래서 처리를 잊은
  ability가 조용히 host 요구가 되지 않는다.

Entrypoint를 표시하는 방법과 entrypoint signature의 규칙은 아래
[미정 사항](#미정-사항)에 둔다.

## Host 경계

모든 entrypoint의 residual row를 합친 것이 프로그램의 host interface다. Host는 그 row에
있는 ability의 operation을 모두 구현해야 한다.

경계 계약은 WebAssembly Component Model의 interface 언어인 WIT로 표현한다. Host
interface의 ability 하나가 WIT interface 하나가 되고, operation은 그 interface의 함수가
된다. 그 의미는 이 문서가 정하고, WIT는 그 의미를 target에 실어 나르는 표현이다.
경계를 넘을 수 있는 것은 WIT로 표현할 수 있는 것으로 한정한다.

- **Wasm:** 프로그램을 component로 내보내고, host interface를 component의 import로,
  entrypoint를 export로 둔다. Host가 import를 채우지 못하면 실행 중이 아니라 instantiation에서 실패한다.
- **Native:** 같은 WIT interface를 C ABI 위에서 쓴다. 값의 배치는 Component Model의
  canonical ABI를 그대로 따르므로, 경계용으로 값을 펼치고 읽는 lowering을 두 타겟이
  공유한다. Launcher나 embedder는 WIT에서 생성한 binding을 구현하며, 빠진 함수는
  link에서 실패한다.

Entrypoint는 root evidence에 host handler를 두고 호출된다. Host handler의 각 arm은 대응하는
import를 호출한다. 그 사이에 Tribute handler가 없다고 증명된 호출은 evidence lookup
없이 import를 직접 호출해도 된다. 이 최적화는 dispatch 결과를 바꾸지 않는다.

## 경계를 넘는 값

Host operation의 parameter와 result는 WIT 타입으로 옮길 수 있는 타입으로 제한한다.

- `Bool`, `Float`, 정수(`Int`, `Nat`): 임의 정밀도 정수와 WIT의 고정 폭 정수 사이의
  변환은 [미정 사항](#미정-사항)에 둔다.
- `Bytes`, `String`: 경계에서 복사한다. Host는 호출이 끝난 뒤 Tribute 쪽 object를
  참조하지 않고, Tribute도 host가 넘긴 버퍼를 호출 뒤에 참조하지 않는다.
- 위 타입으로 만든 `Option`, struct, enum은 WIT의 `option`, `record`, `variant`로
  옮기고 경계에서 값으로 펼친다. 제네릭 타입은 경계에서 인스턴스마다 구체 타입이
  되며, 인스턴스마다 WIT 타입 하나가 생긴다.
- 실패는 operation의 결과 타입으로 돌려준다. 경계를 넘는 예외는 없다.

다음은 경계를 넘지 못한다.

- **재귀 타입.** WIT에는 재귀 타입이 없다. 트리 같은 재귀 데이터는 펼친 인코딩이나
  host resource를 거쳐 넘긴다.
- **함수와 closure.** WIT에는 함수 값이 없다.

Managed 참조의 소유권은 경계를 넘지 않는다. 그래서 host는 Tribute의 reference
counting이나 [비순환 object 그래프](rc.md#비순환-object-그래프) 계약에 참여하지 않는다.

## 표준 입출력과의 관계

[`std::io::Io`](io.md)는 compiler가 정한 target I/O boundary에 고정된 builtin
ambient ability다. Host interface는 같은 생각을 사용자가 정의한 ability로 일반화한다.
`Io`를 host interface에 두는 일반 ability로 다시 정의할지는 미정이다. 그 전까지 `Io`는
[io.md](io.md)의 규칙을 따른다. 특히 `Io`는 handler로 제거할 수 없다.

## 미정 사항

- **선언 문법.** Ability를 host가 처리할 수 있다고 표시하는 방법이다. 예를 들어
  attribute, keyword, 또는 module 단위 표시가 있다.
- **Entrypoint의 표시와 규칙.** `main` 밖의 entrypoint를 표시하는 방법이다. 예를 들어
  `export` 같은 표시, root module의 `pub` 함수, 별도의 world 선언이 있다. Entrypoint의
  parameter와 result가 경계를 넘을 수 있는 타입이어야 한다는 규칙, WIT export 이름,
  entrypoint마다 root evidence를 따로 만들지도 정해야 한다.
- **WIT 생성 규칙.** Host interface의 ability와 그 module path에서 WIT package,
  interface, 함수 이름을 만드는 방식이다. 구체 타입이 된 제네릭 인스턴스의 WIT 타입
  이름도 정해야 한다.
- **정수의 경계 표현.** WIT 정수는 고정 폭이다. 임의 정밀도 정수를 어느 폭으로
  넘길지, 범위를 넘는 값을 trap할지, 경계 전용 고정 폭 정수 타입을 따로 둘지 정해야
  한다.
- **Host가 잡는 resource.** 파일이나 소켓처럼 host가 소유한 object를 Tribute가
  가리키는 불투명 handle 타입이다. WIT의 `resource`로 옮긴다. Tribute 쪽 handle은
  RC로 관리하고, 마지막 참조가 사라지면 resource의 drop을 호출하는 방식을 검토한다.
- **Callback.** Host가 Tribute 함수를 다시 부르는 경로다. WIT에 함수 값이 없으므로
  Tribute가 export하는 resource로 감싼다. 이것도 entrypoint의 한 종류다. 그 함수가
  수행할 수 있는 effect와, 한 entrypoint가 실행 중일 때 host가 다른 entrypoint를 부르는
  재진입 규칙을 정해야 한다.
- **비동기 host 호출.** Host가 나중에 답하는 operation이다. Component Model의
  stackless async lift는 export가 event loop로 돌아갔다가 이벤트마다 다시 불리는
  구조다. CPS로 continuation을 heap에 두는 Tribute는 이 구조에 맞는다. Tribute 쪽
  scheduler handler가 continuation을 보관하고, 완료 이벤트에서 재개하는 방식을
  검토한다. Host operation 자체는 계속 `fn`이다.
- **`Io`의 위치.** 위 [표준 입출력과의 관계](#표준-입출력과의-관계)를 참고한다.
