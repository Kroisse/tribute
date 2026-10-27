# Linking Strategy

## Compilation Unit

**패키지.** 이름 해석과 타입 검사의 단위는 패키지다. 빌드는 패키지 그래프에서
출발한다. 각 패키지의 검사 결과는 자기 source와 의존 패키지의 검사 결과에만
의존하며, 그 패키지에 의존하는 패키지나 같은 빌드의 다른 패키지에는 의존하지
않는다. 따라서 frontend는 검사 결과를 패키지마다 캐시하고, 같은 패키지를 쓰는
빌드는 그 결과를 재사용한다. 의존 패키지의 검사 결과는 선언 메타데이터와 generic
본문을 함께 가진 frontend 결과로 같은 빌드 안에서 전달하며, 이 계약은 그 결과를
직렬화한 package artifact를 요구하지 않는다. 패키지의 이름 범위, 가시성과 선언
식별자는 [modules.md](modules.md)를, 실제 지원 범위는
[capabilities.md](capabilities.md)를 따른다.

**하나의 module.** 특수화, logical lowering, shared·target lowering과 emission은
빌드 전체에 대해 하나의 `core.module`에서 실행한다. 특수화는 빌드에 속한 모든
패키지의 본문을 병합해 실행하며, 병합 순서는
[Implementation — 프론트엔드 단계 경계와 정보 소유권](implementation.md#프론트엔드-단계-경계와-정보-소유권)을
따른다. 의존 패키지 generic의 인스턴스도 그것을 사용하는 패키지와 관계없이 이
module에 둔다. Source-logical callable/control IR과 CPS legalization은 같은 module
안에서 검증한다.

**인스턴스 이름.** Generic 인스턴스의 이름은 그 generic을 정의한 선언의 식별
경로에 타입 인자를 더해 만든다. 인스턴스를 요구한 패키지는 이름에 들어가지
않으므로, 같은 generic을 같은 타입 인자로 요구한 인스턴스는 어느 패키지에서
요구했든 같은 이름을 가지며 module 안에서 하나로 합쳐진다.

패키지 그래프는 빌드 입력이 정한다. `use` 경로만으로 임의의 파일을 읽거나
dependency graph를 만드는 것으로 가정하지 않는다. 패키지마다 별도 module이나
object를 만들어 링크하는 backend separate compilation은 제공하지 않는다
([Separate Compilation의 요구 조건](#separate-compilation의-요구-조건)).

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

### Runtime sysroot

Native executable은 `tribute-runtime` staticlib과 링크한다. 파일 이름은 target의
rustc staticlib 규칙을 따른다(대부분 `libtribute_runtime.a`, MSVC는
`tribute_runtime.lib`).
컴파일러는 이 라이브러리를 sysroot에서 찾는다. Target별 runtime 디렉토리는
`<sysroot>/lib/tribute/<target-triple>/`이다.

Sysroot는 링크 시점에 다음 순서로 정한다.

1. CLI의 `--sysroot <path>`
2. 환경 변수 `TRIBUTE_SYSROOT`
3. 컴파일러 실행 파일 기준 `<exe_dir>/..`
   (`<prefix>/bin/tribute`와 `<prefix>/lib/tribute/<target-triple>/`)

컴파일러는 빌드 머신의 경로를 자신의 바이너리에 기록하지 않는다. Runtime
위치는 컴파일러를 빌드할 때가 아니라 실행할 때 결정하므로, 설치된 컴파일러는
prefix를 옮겨도 동작하고 컴파일러 빌드 산출물은 checkout 경로에 의존하지 않는다.
선택된 sysroot에 runtime 라이브러리가 없으면 runtime 없이 링크하지 않고, 확인한
경로를 포함한 link 오류를 보고한다.

Runtime은 `runtime` profile의 `no_std` staticlib으로 빌드하며, 컴파일러 crate의
build script가 만들지 않는다. 개발 checkout에서는 `cargo xtask runtime`이
`target/sysroot`에 같은 layout을 만들고, workspace의 Cargo 설정이
`TRIBUTE_SYSROOT`를 그 경로로 지정한다.

## Separate Compilation의 요구 조건

Backend separate compilation은 패키지마다 별도 `core.module`, object 또는 Wasm
module을 만들어 링크하는 것을 뜻한다. 이 단계는 제공하지 않으며, 도입하려면 다음
계약을 함께 정의해야 한다.

1. Source graph, module visibility와 canonical declaration identity
2. Generic specialization의 소유권과 중복 제거
3. Exact callable convention, parameter/result ABI와 closure storage
4. WasmGC recursive type group과 nominal layout의 module 간 대응
5. Native ownership, RTTI 및 runtime helper identity의 link 경계
6. Effect operation identity와 evidence/handler dispatch 계약

1번은 [Compilation Unit](#compilation-unit)의 패키지 모델과
[modules.md](modules.md)의 가시성·선언 식별자 규칙이 정한다. 2–6번은 backend
separate compilation에만 필요한 조건으로 남는다. 인스턴스 이름 규칙은 여러 unit이
만든 같은 인스턴스를 같은 이름으로 식별하게 하지만, 어느 unit이 인스턴스를
소유하는지와 중복을 제거하는 link 규칙은 정하지 않는다.

Wasm import/export나 component boundary는 이 source/ABI 계약을 대신하지 않는다.
지원되지 않는 조합은 명시적으로 진단하고 다른 compilation 경로로 조용히 바꾸지 않는다.
