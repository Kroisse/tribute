# Tribute Feature Proposals

언어와 compiler phase의 authoritative contract는 [`new-plans/`](../new-plans/)에
있다. 현재 target 지원과 실행 증거는
[`capabilities.md`](../new-plans/capabilities.md)를 따른다. 구현 작업과 진행 상황은
GitHub Issues에서 추적한다.

## Compiler Design

| 문서 | 범위 |
| ---- | ---- |
| [design.md](../new-plans/design.md) | 언어와 source-logical compiler 구조 |
| [syntax.md](../new-plans/syntax.md) | Source syntax |
| [types.md](../new-plans/types.md) | 타입과 nominal identity |
| [abilities.md](../new-plans/abilities.md) | Ability와 affine resume |
| [modules.md](../new-plans/modules.md) | Module과 이름 해석 |
| [type-inference.md](../new-plans/type-inference.md) | 타입 추론과 effect row |
| [ir.md](../new-plans/ir.md) | IR dialect와 legality |
| [cps-effects.md](../new-plans/cps-effects.md) | Shared CPS와 callable ABI |
| [implementation.md](../new-plans/implementation.md) | Compiler phase 책임 |
| [cranelift-backend.md](../new-plans/cranelift-backend.md) | Native backend와 RC |
| [wasm-backend.md](../new-plans/wasm-backend.md) | WasmGC backend |

## Feature Proposals

아래 문서는 추가 library·tooling 기능의 제안이며 현재 지원이나 compiler 실행 순서를
정의하지 않는다. Source 예제는 `new-plans/`의 syntax와 ability 계약을 따른다.

| 문서 | 범위 |
| ---- | ---- |
| [05-standard-library.md](05-standard-library.md) | 표준 library 확장 방향 |
| [06-package-manager.md](06-package-manager.md) | Package manager와 registry |
| [07-testing-framework.md](07-testing-framework.md) | Source test runner와 effect mocking |
| [08-documentation-system.md](08-documentation-system.md) | API 문서 생성과 example testing |
