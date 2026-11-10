# Tribute 설계 문서

이 디렉터리의 문서는 Tribute 언어와 compiler 설계의 authoritative contract다.
코드, 테스트, 다른 문서가 이 문서들과 어긋나면 이 문서들을 따르고 구현을 맞춘다.
Operation 하나의 계약은 그 `#[dialect]` 선언의 doc comment가 정하며, 이 문서들은
여러 operation에 걸친 규칙을 적는다. 현재 target 지원과 실행 증거는
[docs/capabilities.md](../docs/capabilities.md)가 기록하고, 구현 작업과 진행
상황은 GitHub Issues에서 추적한다.

## 언어

| 문서 | 범위 |
| ---- | ---- |
| [design.md](design.md) | 언어 설계 원칙과 source-logical compiler 구조, 실행 전략 |
| [syntax.md](syntax.md) | Source syntax |
| [types.md](types.md) | 타입 시스템과 nominal identity |
| [generics.md](generics.md) | 제네릭 타입과 함수, monomorphization |
| [numeric-types.md](numeric-types.md) | 수치 타입과 변환 |
| [string.md](string.md) | `String`과 `Bytes` |
| [abilities.md](abilities.md) | Ability, handler, affine resume |
| [modules.md](modules.md) | Module, 이름 해석, 패키지 매니페스트 |
| [type-inference.md](type-inference.md) | 타입 추론과 ability row, evidence 선택 |
| [io.md](io.md) | Ambient `Io` ability와 기본 입출력 API |
| [host-interface.md](host-interface.md) | Host가 handler를 공급하는 ability 경계 |

## Compiler와 IR

| 문서 | 범위 |
| ---- | ---- |
| [ir.md](ir.md) | TrunkIR core: dialect 계층, legality, 선언적 schema, 타입 모델 |
| [tribute-dialects.md](tribute-dialects.md) | Tribute 고수준 dialect의 operation 간 규칙과 타입 규칙 |
| [cps-effects.md](cps-effects.md) | 직접형 제어의 CPS 변환과 convention class, effect dispatch |
| [frontend.md](frontend.md) | Frontend 단계 경계, 패키지 입력, 증분 분석 |
| [lowering.md](lowering.md) | Lowering pass 구조와 pass 순서, 출력 결정성 |
| [optimizations.md](optimizations.md) | 최적화 경계와 검증 계약 |
| [abi-boundary.md](abi-boundary.md) | 단계 불변 조건과 representation/ABI 경계 |

## Target과 runtime

| 문서 | 범위 |
| ---- | ---- |
| [cranelift-backend.md](cranelift-backend.md) | Native backend |
| [rc.md](rc.md) | Native backend의 reference counting |
| [wasm-backend.md](wasm-backend.md) | WasmGC backend |
| [runtime-types.md](runtime-types.md) | Runtime 타입 descriptor와 layout |
| [linking.md](linking.md) | 컴파일 단위와 linking |

## 도구

| 문서 | 범위 |
| ---- | ---- |
| [lsp.md](lsp.md) | Language server |
