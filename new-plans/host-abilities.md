# Host Abilities

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

## Host Ability

Host ability는 프로그램 안에서 처리되지 않으면 host가 처리하는 ability다.

```rust
ability Clock {
    fn now_millis() -> Int
}

ability Kv {
    fn get(key: String) -> Option(Bytes)
    fn put(key: String, value: Bytes) -> Nil
}
```

Ability를 host ability로 표시하는 선언 문법은 아래 [미정 사항](#미정-사항)에 둔다.
표시 외에는 일반 ability와 같은 규칙을 따른다.

- **Operation은 모두 `fn`이다.** Host 경계는 continuation을 넘기지 않는다. 선언부가
  보증하므로([Operation 종류](abilities.md#operation-종류)) host ability operation의
  호출은 continuation을 잡지 않는다. `op` operation을 가진 ability는 host ability가
  될 수 없다.
- **Tribute handler가 먼저 처리한다.** Host ability operation은 가장 가까운 handler로
  dispatch된다. 프로그램 안에 handler가 없을 때만 host에 닿는다. 그래서 host
  ability를 쓰는 코드는 host 없이도 handler를 씌워 실행하고 테스트할 수 있다.
- **`main`의 residual row에 남을 수 있다.** `main`에 처리되지 않은 채 남은 host
  ability들이 그 프로그램이 host에 요구하는 기능의 전부다. Host ability가 아닌
  residual effect는 지금처럼 frontend가 거부한다.

## Host 경계

`main`의 residual row가 프로그램의 host interface다. Host는 그 row에 있는 host
ability의 operation을 모두 구현해야 한다.

- **Wasm:** operation마다 module import 하나를 둔다. Import 이름은 ability의
  qualified path와 operation 이름에서 정한다. Host가 import를 채우지 못하면
  실행 중이 아니라 instantiation에서 실패한다.
- **Native:** operation마다 extern symbol 하나를 둔다. Launcher나 embedder가 그
  symbol을 제공하며, 빠진 symbol은 link에서 실패한다.

Entrypoint는 root evidence에 host handler를 둔다. Host handler의 각 arm은 대응하는
import나 extern을 호출한다. 그 사이에 Tribute handler가 없다고 증명된 호출은 evidence
lookup 없이 import를 직접 호출해도 된다. 이 최적화는 dispatch 결과를 바꾸지 않는다.

## 경계를 넘는 값

Host operation의 parameter와 result는 경계 타입으로 제한한다.

- `Bool`, `Float`, 정수(`Int`, `Nat`): 정수의 경계 표현은
  [미정 사항](#미정-사항)에 둔다.
- `Bytes`, `String`: 경계에서 복사한다. Host는 호출이 끝난 뒤 Tribute 쪽 object를
  참조하지 않고, Tribute도 host가 넘긴 버퍼를 호출 뒤에 참조하지 않는다.
- 위 타입으로 만든 `Option`과 struct·enum은 경계에서 값으로 펼친다.

Managed 참조의 소유권은 경계를 넘지 않는다. 그래서 host는 Tribute의 reference
counting이나 [비순환 object 그래프](rc.md#비순환-object-그래프) 계약에 참여하지 않는다.

## 표준 입출력과의 관계

[`std::io::Io`](io.md)는 compiler가 정한 target I/O boundary에 고정된 builtin
ambient ability다. Host ability는 같은 생각을 사용자가 정의한 ability로 일반화한다.
`Io`를 host ability의 한 사례로 다시 정의할지는 미정이다. 그 전까지 `Io`는
[io.md](io.md)의 규칙을 따른다. 특히 `Io`는 handler로 제거할 수 없다.

## 미정 사항

- **선언 문법.** Ability를 host ability로 표시하는 방법이다. 예를 들어 attribute,
  keyword, 또는 module 단위 표시가 있다.
- **Import 이름 규칙.** Qualified path를 Wasm import의 module과 field 이름으로 나누는
  방식과 native symbol mangling이다.
- **정수의 경계 표현.** 임의 정밀도 정수를 고정 폭으로 넘길지, 범위를 넘는 값을
  trap할지, 경계 타입으로 고정 폭 정수 타입을 따로 둘지 정해야 한다.
- **Host가 잡는 resource.** 파일이나 소켓처럼 host가 소유한 object를 Tribute가
  가리키는 불투명 handle 타입이다. 수명은 host가 관리하고, Tribute 쪽은 handle 값만
  RC로 관리하는 방식을 검토한다.
- **Callback.** Host가 Tribute 함수를 다시 부르는 경로다. 그 함수가 수행할 수 있는
  effect와 경계 안에서의 재진입 규칙을 정해야 한다.
- **비동기 host 호출.** Host가 promise로 답하는 operation이다. Wasm에서는 JS Promise
  Integration처럼 host가 Wasm 실행 전체를 멈추는 장치로 구현할 수 있지만, 그 경우에도
  operation은 Tribute 쪽에서 `fn`으로 보인다. 프로그램 중간의 handler로 제어를
  옮기는 일반 effect를 대신하지는 않는다.
- **`Io`의 위치.** 위 [표준 입출력과의 관계](#표준-입출력과의-관계)를 참고한다.
