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

Host가 처리하는 ability는 선언의 종류가 아니다. [Entrypoint](#entrypoint)의 residual
row에 남은 ability를 host가 처리하며, 그 row들이 program의 host interface다. Ability를
host가 처리하도록 따로 표시하는 문법은 없다.

```rust
ability Clock {
    fn now_millis() -> Int
}

ability Kv {
    fn get(key: String) -> Option(Bytes)
    fn put(key: String, value: Bytes) -> Nil
}
```

Entrypoint의 residual row에 남는 ability는 다음 조건을 모두 지켜야 한다. Frontend는
각 entrypoint에서 이 조건을 검사하고, 어긴 ability가 있으면 거부한다.

- **Operation의 타입은 경계를 넘을 수 있다.** Parameter와 result는
  [경계를 넘는 값](#경계를-넘는-값)으로 제한한다. 제네릭 ability는 entrypoint의 row에서
  구체 instance가 되므로 그 instance의 타입으로 검사한다.

`fn` operation과 `op` operation 모두 host에 맡길 수 있다. 둘은 경계에서 다른 종류의
함수가 된다([Operation 종류와 경계](#operation-종류와-경계)).

Ability가 host에 닿는 것은 작성자가 entrypoint의 row에 그 ability를 적었을 때뿐이다.
Entrypoint의 row는 본문에서 추론하지 않으므로, 처리를 잊은 ability가 조용히 host 요구가
되지 않는다.

- **Tribute handler가 먼저 처리한다.** Host interface의 operation도 가장 가까운
  handler로 dispatch된다. 프로그램 안에 handler가 없을 때만 host에 닿는다. 그래서 이런
  ability를 쓰는 코드는 host 없이도 handler를 씌워 실행하고 테스트할 수 있다.
- **같은 ability가 두 역할을 할 수 있다.** 어떤 program에서는 Tribute handler가
  처리하는 ability를 다른 program은 entrypoint에 남겨 host에게 맡길 수 있다. 예를 들어
  [`std::io::Write`](io.md#read와-write)는 표준 handler로 stream에 묶을 수도, host가
  직접 구현할 수도 있다.

## Entrypoint

Entrypoint는 host가 부르는 Tribute 함수다. `main`은 command 형태의 host가 부르는
entrypoint의 한 경우다. Plugin이나 embedding host는 여러 entrypoint를 부를 수 있다.

- 각 entrypoint의 residual row에는 host가 처리할 수 있는 ability만 남을 수 있다.
- 프로그램의 host interface는 모든 entrypoint의 residual row를 합친 것이다. 각
  entrypoint의 signature는 그 entrypoint가 닿는 host 기능을 따로 보여 준다.

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
  entrypoint를 export로 둔다. Host가 import를 채우지 못하면 실행 중이 아니라
  instantiation에서 실패한다.
- **Native:** 같은 WIT interface를 C ABI 위에서 쓴다. 값의 배치는 Component Model의
  canonical ABI를 그대로 따르므로, 경계용으로 값을 펼치고 읽는 lowering을 두 타겟이
  공유한다. Launcher나 embedder는 WIT에서 생성한 binding을 구현하며, 빠진 함수는
  link에서 실패한다.

### Operation 종류와 경계

| Operation | WIT 함수 | 경계에서 일어나는 일 |
| --- | --- | --- |
| `fn` | 동기 함수 | host가 결과를 돌려주면 그 값으로 계속한다. |
| `op` | `async` 함수 | 아래와 같다. |

`op` operation이 host에 닿으면 entrypoint의 host handler가 비동기 import를 부른다.

- **Continuation은 program 밖으로 나가지 않는다.** Host가 그 자리에서 결과를 돌려주면
  바로 resume한다. 아니면 handler는 continuation을 host가 준 호출 식별자로 보관하고
  host의 event loop로 돌아간다. 완료가 오면 보관한 continuation을 꺼내 resume한다.
  Host는 Tribute 함수나 continuation을 받지 않으므로 함수 값이 경계를 넘지 않는다는
  규칙은 그대로다.
- **Resume은 많아야 한 번이다.** 한 호출은 한 번만 완료된다. Tribute의 `resume`도
  많아야 한 번이므로([`resume`의 사용 규칙](abilities.md#resume의-사용-규칙)) 두 규칙이
  맞는다.
- **취소.** Program이 보관한 continuation을 resume하지 않고 버리면 그 호출을 취소한다.
  Host가 호출을 취소하면 보관한 continuation을 drop하며, 그 안의 값은 RC로 해제된다.
- **끝나지 않는 operation.** 결과 타입이 `Never`인 `op`는 host가 값을 받고 resume하지
  않는다. 예를 들어 entrypoint에 남은 `Throw(e)`는 처리되지 않은 오류를 host에 넘긴다.

Wasm에서 `op`의 import는 Component Model의 비동기 함수이며, entrypoint export는 callback
방식으로 event loop에 돌아간다. Native launcher도 같은 event loop를 둔다. Host는 비동기
import를 그 자리에서 완료해도 되며, 그러면 program은 기다리지 않고 계속한다.

### Root handler

Entrypoint는 root evidence에 host handler를 둔다. Host handler의 각 arm은 대응하는
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
- [Host resource](#host-resource) 값은 host가 만든 handle로 넘는다. 경계를 넘는
  handle은 모두 host가 이전에 돌려준 것이다.
- 실패는 operation의 결과 타입으로 돌려준다. 경계를 넘는 예외는 없다.

다음은 경계를 넘지 못한다.

- **재귀 타입.** WIT에는 재귀 타입이 없다. 트리 같은 재귀 데이터는 펼친 인코딩이나
  host resource를 거쳐 넘긴다.
- **함수와 closure.** WIT에는 함수 값이 없다.

Managed 참조의 소유권은 경계를 넘지 않는다. 그래서 host는 Tribute의 reference
counting이나 [비순환 object 그래프](rc.md#비순환-object-그래프) 계약에 참여하지 않는다.

## Host Resource

Host resource는 host가 소유한 object를 가리키는 불투명 handle 타입이다. 파일, stream,
소켓이 그 예다. WIT의 `resource`로 옮긴다.

- **Host만 만든다.** Tribute source는 host resource 값을 만들 수 없다. 값은 host
  operation의 결과로만 생긴다. 그래서 program 안의 모든 host resource는 host의
  handle 공간 하나에 속한다. Host resource를 쓰는 ability를 Tribute handler가
  처리하더라도, 그 handler는 handle을 만들지 못하고 받은 handle을 바깥 handler로
  넘기거나, 실패를 돌려주거나, 호출을 관찰할 수 있을 뿐이다.
- **소유권.** Tribute 쪽 handle object가 WIT의 `own`을 가진다. Host operation의
  parameter는 항상 `borrow`이고, resource를 돌려주는 result는 `own`이다. Tribute가
  handle의 소유권을 host로 넘기는 operation은 없다.
- **표현.** Handle object는 host handle을 담은 불변 RC object다. Managed 참조를 담지
  않으므로 [비순환 object 그래프](rc.md#비순환-object-그래프)를 만들지 않는다. 닫힘
  표시는 unmanaged flag다.

### 자식 resource

어떤 resource는 다른 resource에서 나온다. 파일이나 소켓이 내주는 읽기용·쓰기용 stream이
그 예다. 자식 resource의 handle object는 부모 resource의 handle object를 참조한다.

- 그래서 부모는 자식이 모두 release된 뒤에 release된다. Host가 자식보다 부모가 먼저
  drop되는 것을 오류로 다루더라도 drop 순서는 RC가 지킨다.
- 자식은 자기보다 먼저 만들어진 부모만 가리키므로
  [비순환 object 그래프](rc.md#비순환-object-그래프)를 지킨다.
- 자식은 따로 close한다. 읽기용 stream을 닫아도 쓰기용 stream은 열려 있다.

### 수명

- **Drop은 안전망이다.** 마지막 참조가 release되면 handle object의 release가
  host의 drop을 부른다. Drop은 handle을 더 쓸 수 없게 된 뒤 반드시 일어나지만, 그
  시점은 정하지 않는다. RC 해제 시점은 관찰할 수 없다는 전제를 host resource에도
  그대로 두므로, RC 최적화는 host resource의 release도 다른 값과 같이 옮길 수 있다.
  Drop은 host의 destructor만 부르고 Tribute code를 다시 부르지 않는다.
- **시점이 필요하면 close를 쓴다.** Resource를 정의하는 ability는 그 resource를
  지금 끝내는 close operation을 둔다. Close는 host 쪽에서 끝내는 데 필요한 일(stream의
  flush 등)을 하고 resource를 drop한 뒤, 그 과정의 실패를 `Result`로 돌려준다. Drop에는
  실패를 돌려줄 길이 없으므로 실패를 받는 곳은 close뿐이다.
- **닫힌 handle.** Close한 handle을 쓰는 operation은 host를 부르지 않고 실패를 돌려준다.
  Host의 handle table은 비운 칸을 다시 쓰므로 이 검사는 Tribute 쪽 닫힘 표시로 한다.
  이미 닫힌 handle을 다시 close하면 아무 일도 하지 않고 성공한다. 닫힌 handle의
  마지막 참조가 release되면 drop을 다시 부르지 않는다.
- **범위 helper.** Resource를 받아 계산이 끝나면 close하는 helper는 표준 라이브러리의
  일반 함수다. [`with` 문](syntax.md#with-statement)과 함께 쓴다. 범위를 벗어난
  handle을 쓰는 것은 막지 않으며, 그때는 닫힌 handle의 규칙을 따른다.
- **Target.** 결정적인 drop은 RC로 관리하는 표현에서 성립한다. RC가 없는 WasmGC
  표현에서는 handle이 close나 instance 종료 때만 해제된다.

Host resource에 linear type이나 affine type은 두지 않는다. Continuation은 resume되지
않고 drop될 수 있으므로([`resume`의 사용 규칙](abilities.md#resume의-사용-규칙))
그 안의 handle을 자동으로 해제하는 길이 어차피 필요하고, 연산마다 handle을 돌려받거나
별도의 borrow 규칙을 두는 비용이 그 이득보다 크다.

## 표준 입출력과의 관계

표준 입출력도 같은 규칙을 따른다. [`std::io::Io`](io.md#io)와
[`std::io::Read`, `std::io::Write`](io.md#read와-write)는 패키지 `std`의 source에 선언한
ability이며, Tribute handler로 가로챌 수 있다. Compiler가 특별히 아는 ambient
ability는 없다. 표준 ability의 두 층은 [std-abilities.md](std-abilities.md)가
정한다.

## 미정 사항

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
- **Host resource의 선언 문법.** Ability 안에서 resource 타입을 선언하는 문법과,
  그 타입의 WIT 이름을 정해야 한다.
- **Tribute가 정의하는 resource.** `handle` 식이 resource의 표현을 정하게 하면 host
  ability를 흉내 내는 handler가 자기 handle을 만들 수 있다. 그러나 handler는 동적으로
  찾으므로, 한 arm이 받는 handle의 표현이 그 handler의 것인지 정적으로 알 수 없다.
  서로 다른 구현자의 표현이 섞이지 않게 하려면 타입이 구현자를 가리키거나, 값이
  구현자를 기억해 그 구현자로 dispatch해야 한다. 타입으로 가리키는 후보는 handle
  타입을 ability의 타입 매개변수로 두는 것(`Output(o)`에서 host는 `o = OutStream`)과
  handler가 fresh한 scope 타입을 도입하는 것(`Io(s)`, `OutStream(s)`)이다. 앞의 것은
  instance가 다르면 다른 ability라는 기존 규칙만으로 표현이 섞이지 않는다. 이 가운데
  하나를 들이기 전까지 Tribute handler에게 host resource는 불투명하다.
- **Callback.** Host가 Tribute 함수를 다시 부르는 경로다. WIT에 함수 값이 없으므로
  Tribute가 export하는 resource로 감싼다. 이것도 entrypoint의 한 종류다. 그 함수가
  수행할 수 있는 effect와, 한 entrypoint가 실행 중일 때 host가 다른 entrypoint를 부르는
  재진입 규칙을 정해야 한다.
- **동시에 기다리는 호출.** `op` operation의 비동기 경계는 continuation 하나가 기다리는
  경우를 정한다. 여러 계산을 동시에 진행하고 여러 호출을 함께 기다리는 scheduler,
  그리고 기다리는 동안 host가 다른 entrypoint를 부르는 재진입 규칙을 정해야 한다.
- **열린 자식이 있는 부모의 close.** 자식 resource가 열린 채 부모를 close하면 실패를
  돌려줄지, 자식도 닫힌 것으로 표시할지, host의 규칙을 따를지 정해야 한다.
