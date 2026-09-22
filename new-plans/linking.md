# Linking Strategy

## Compilation Unit

현재 CLI는 하나의 source file과 embedded Prelude를 준비하여 하나의 shared module로
컴파일한다. Source-logical callable/control IR과 CPS legalization은 같은 module 안에서
검증한다. Module system의 source 계약은 [modules.md](modules.md), 실제 지원 범위는
[capabilities.md](capabilities.md)를 따른다.

File-module loading과 별도 Tribute library compilation은 제공하지 않는다. `use`
경로만으로 임의의 파일을 읽거나 dependency graph를 만드는 것으로 가정하지 않는다.
이 제한은 Salsa의 frontend query caching과 별개다.

## WasmGC

Wasm backend는 완성된 shared module을 하나의 `.wasm`으로 emit한다. Module 내부의
GC type index, recursive type group, function/table index는 emitter가 함께 정한다.
WASI import는 [wasm-backend.md](wasm-backend.md)의 target ABI 계약을 따른다.

현재 compilation unit 사이에서 GC object나 closure를 주고받는 별도 linker ABI는
정의하지 않는다. Source nominal identity를 backend type index와 동일시하거나,
서로 다른 module의 같은 정수 index가 같은 타입이라고 가정해서는 안 된다.

## Native

Native backend는 Cranelift object를 만들고 host linker로 executable을 생성한다.
Private runtime helper, allocator와 external C symbol은 target에서 정한 physical
ABI를 사용한다. 이 native link 단계가 별도 Tribute compilation unit 사이의
source-level ABI를 제공하는 것은 아니다.

## Separate Compilation의 요구 조건

별도 compilation unit을 도입하려면 다음 계약을 함께 정의해야 한다.

1. Source graph, module visibility와 canonical declaration identity
2. Generic specialization의 소유권과 중복 제거
3. Exact callable convention, parameter/result ABI와 closure storage
4. WasmGC recursive type group과 nominal layout의 module 간 대응
5. Native ownership, RTTI 및 runtime helper identity의 link 경계
6. Effect operation identity와 evidence/handler dispatch 계약

Wasm import/export나 component boundary는 이 source/ABI 계약을 대신하지 않는다.
지원되지 않는 조합은 명시적으로 진단하고 다른 compilation 경로로 조용히 바꾸지 않는다.
