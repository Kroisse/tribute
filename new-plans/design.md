# Tribute Language Design

> _This is not the greatest language in the world, no. This is just a tribute._

## Overview

Tribute는 정적 타입과 algebraic effects를 갖춘 함수형 프로그래밍 언어이다.

Current implementation and target support is tracked separately in
[capabilities.md](capabilities.md). Design intent in this document is not a
support claim.

### Design Goals

- **친숙한 문법**: ML의 의미론 + C/Rust 스타일 문법 (중괄호, 세미콜론, fn 키워드 등)
- **기존 언어와 친숙한 외관**: 정적 타입 함수형 언어지만, C/Rust/TypeScript 개발자에게 낯설지 않은 문법
- **정적 타입 시스템**: 타입 추론 + ability 추론
- **Algebraic Effects (Abilities)**: Unison 스타일의 ability 시스템
- **다중 컴파일 타겟**: Cranelift (네이티브) + WasmGC

### Non-Goals

- 매크로 시스템 (현재 범위 외)
- First-class continuation (delimited로 제한)
- Multi-shot continuation (one-shot만 지원)
- Typeclass / Trait (명시적 함수 전달로 대체)

### Syntax Style

ML의 의미론을 갖지만 C 계열 개발자에게 친숙한 문법:

```rust
// 타입 선언
struct User { name: String, age: Int }
enum Option(a) { None, Some(a) }

// 함수 정의
fn add(x: Int, y: Int) -> Int {
    x + y
}

// 패턴 매칭
fn describe(value: Option(Int)) -> String {
    case value {
        Some(n) -> "got: " <> Int::to_string(n)
        None -> "nothing"
    }
}

// Illustrative List algorithms; 별도 public API 절에서 확정되기 전까지 예시다.
// UFCS 체이닝 (인자 없으면 괄호 생략)
fn process(data: List(Int)) -> Int {
    data
        .filter(fn(x) x > 0)
        .map(fn(x) x * 2)
        .fold(0, fn(a, b) a + b)
}

// Record 생성과 업데이트
let user = User { name: "Alice", age: 30 }
let older = User { ..user, age: 31 }

// Abilities
fn fetch_user(id: UserId) ->{Http, Async} User {
    let response = Http::get("/users/" <> id)
    response.await
}
```

---

## Module System

> 상세 내용은 modules.md 참조

### 핵심 결정 사항

| 항목                | 선택                              |
| ------------------- | --------------------------------- |
| 모듈 구분자         | `::`                              |
| 메서드 호출 스타일  | UFCS (`.`)                        |
| 타입 선언           | `struct` (product) / `enum` (sum) |
| Ad-hoc polymorphism | 없음 (명시적 전달)                |
| 이름 해소           | Type-directed (use 범위 내)       |

### 기본 문법

```rust
// Use
use std::collections::{List, Option}

// Enum과 동명 네임스페이스
pub enum Option(a) {
    None
    Some(a)
}

pub mod Option {
    pub fn map(value: Option(a), f: fn(a) -> b) -> Option(b) { ... }
}

// List construction과 UFCS 사용
let xs = [1, 2, 3]
let prefixed = List::prepend(0, xs)
let option = Option::Some(1)
let mapped = option.map(fn(x) x + 1)  // Option::map(option, ...) 로 해석
```

`List(a)`는 compiler-owned canonical identity를 가진 opaque nominal immutable
persistent RRB tree다. List literal과 canonical persistent operation은 원소 표현식을
왼쪽에서 오른쪽으로 정확히 한 번 평가한다. `Empty`/`Cons` 같은 representation
constructor나 RRB node shape은 source API가 아니다. List의 shared IR operation과
source-visible observation은 sequence 의미만 표현하고, native와 Wasm backend는 같은
RRB invariant를 만족하는 target-private physical layout을 사용한다.

---

## Ability System

> 상세 문법은 abilities.md 참조

Tribute는 Unison의 선례를 따라 algebraic effect를 **ability**라고 부른다.

일반 ability와 별도로 compiler-owned builtin ambient ability를 둘 수 있다.
현재 유일한 ambient ability는 `std::io::Io`이며 handler로 제거할 수 없고
entrypoint에 terminal effect로 남을 수 있다. 사용자에게 ambient ability 선언
문법은 제공하지 않는다. 기본 I/O API와 calling convention은
[io.md](io.md)를 따른다.

### Continuation 의미론

Tribute의 ability 시스템은 **delimited, one-shot continuation**을 기반으로 한다.

| 속성       | 선택 | 이유                                    |
| ---------- | ---- | --------------------------------------- |
| Delimited  | ✅   | prompt까지만 캡처, 합성 가능            |
| One-shot   | ✅   | 구현 단순, 대부분의 실용적 ability 지원 |
| Multi-shot | ❌   | nondeterminism 포기, 복잡도 감소        |

### One-shot의 의미

Resumptive `op` handler의 continuation은 **affine capability**다:

- `resume value`로 최대 한 번 재개한다.
- 재개하지 않고 handler가 반환하면 continuation을 암묵적으로 버린다.
- `fn` arm과 `op -> Never` arm에는 resume capability가 없다.

상세한 검증과 capture 규칙은 [abilities.md](abilities.md)를 따른다.

### 지원 가능한 Ability 패턴

```text
✅ Exception / Abort     - continuation 버림
✅ State (Get/Set)       - continuation 1번 사용
✅ Reader / Writer       - continuation 1번 사용
✅ Async / Await         - continuation 1번 사용
✅ Generator / Yield     - continuation 순차 사용
✅ Coroutine             - continuation 순차 사용

❌ Each / Amb            - multi-shot 필요
❌ Backtracking search   - multi-shot 필요
```

---

## Type System

> 타입 선언, record, UFCS 규칙은 types.md 참조

### 함수 타입과 Ability

함수 타입에 ability 정보가 포함된다:

```text
fn(a) ->{E} b
       ~~~~
       이 함수가 수행할 수 있는 abilities
```

**Ability polymorphism이 기본이다:**

```rust
// 이 두 타입은 동일
fn(a) -> b
fn(a) ->{g} b    // 임의의 ability g에 대해 polymorphic
```

**순수 함수는 빈 ability 집합으로 명시:**

```rust
fn(a) ->{} b    // 빈 effect를 명시한 pure 함수
```

### Ability 추론

대부분의 경우 ability는 추론된다:

```rust
fn example() {
    let x = State::get()     // State ability 추론
    let y = Async::await(p)  // Async ability 추론
    x + y
}
// 추론된 타입: fn example() ->{State(Int), Async} Int
```

### Ability Polymorphism 예시

```rust
// f의 ability가 그대로 전파됨
fn map(f: fn(a) -> b, list: List(a)) -> List(b)

// 순수 함수만 받는 경우 명시
fn memoize(f: fn(a) ->{} b) ->{} fn(a) ->{} b
```

---

## Compiler Architecture

> 상세 내용은 ir.md 참조

### TrunkIR

Tribute 컴파일러는 **TrunkIR**이라는 multi-level IR을 사용한다.
MLIR의 dialect 개념을 차용하여 여러 수준의 연산이 한 모듈 내에 공존할 수 있다.

### Dialect 계층

| 수준 | Dialect | 책임 |
| ---- | ------- | ---- |
| Infrastructure | `core` | 모듈, 일반 값 타입과 conversion cast |
| Source-logical control | `tribute_control` | source callable, handle, perform, resume |
| Shared value/control | `func`, `closure`, `ability`, `effect`, `scf`, `cf`, `arith`, `adt`, `list`, `tribute_rt`, `tribute_io` | CPS callable, 명시적 evidence와 dispatch, 일반 값 연산 |
| Target | `wasm`, `wasm_gc`, `clif` | target ABI, 저장소와 명령어 |

### Compilation Pipeline

```text
Source → CST → AST
    → 이름 해석·타입 추론·TDNR
    → Prelude 병합·intrinsic 검증·monomorphization
    → source-logical tribute_control IR
    → pre-CPS 검증·tribute_control_to_cps
    → lambda lifting·intrinsic lowering
    → ability dispatch·evidence resolution·handle delimiter 제거
    → target ABI 검증·물리 CPS signature·root entry bridge
    → closure storage·target evidence lowering
    ├→ WasmGC lowering → Wasm binary
    └→ typed ownership/RTTI 계획 → native lowering → native binary
```

Frontend는 source의 결과 타입과 operation kind를 보존한다. Continuation과 hidden
ABI 인자는 shared CPS legalization이 만들고, target 경계가 callable 결과 목록과
closure 저장소를 물리화한다. 구체적인 pass 순서와 검증 경계는
[implementation.md](implementation.md)와 [cps-effects.md](cps-effects.md)를 따른다.

---

## Target: WasmGC

### 장점

- **네이티브 GC**: 자체 GC 구현 불필요
- **Tail calls**: WASM 3.0에서 표준화
- **크로스 플랫폼**: 브라우저 + WASI 런타임

### Effect 구현 전략

WasmGC는 shared middle-end가 만든 tail-call CPS, 명시적 evidence와 closure IR을
소비한다. Target lowering은 GC layout과 exact callable signature를 정하고,
직접/간접 proper-tail transfer를 Wasm 명령어로 내린다. 실행 지원 범위는
[capabilities.md](capabilities.md), target 계약은 [wasm-backend.md](wasm-backend.md)를
따른다.

### 코드 생성

```text
Shared IR
    → WasmGC 및 Wasm dialect lowering
    → wasm-encoder emission
    → .wasm
```

---

## Target: Cranelift

### 장점

- **네이티브 성능**: 직접 기계어 생성
- **빠른 컴파일**: JIT 설계 기반
- **Rust 친화적**: Rust로 작성됨

### 아키텍처: 2-Layer 패턴

WASM 백엔드와 동일한 구조를 유지한다:

```text
tribute-passes/src/native/       Tribute 전용 native lowering
  evidence runtime adaptation, boxing(RC)
  + func/arith/scf/adt/mem → clif.* dialect 변환

trunk-ir-cranelift-backend/      언어 독립적 Cranelift codegen
  clif.* dialect → Cranelift IR → 네이티브 바이너리
```

- `clif.*` dialect은 Cranelift IR과 1:1 대응 (`wasm.*`과 대칭)
- `trunk-ir-cranelift-backend`은 `trunk-ir`만 의존 (Tribute 독립적)

### Effect 구현 전략: Tail-Call CPS

Shared CPS legalization이 continuation을 typed closure로 명시화한다. Native backend는
검증된 callable ABI를 받아 evidence lookup과 dispatch를 lowering한다.

- `fn` operation: `tribute_control.perform` → `ability.call` → `effect.dispatch_tail`
- `op` operation: `tribute_control.perform` → `ability.perform` → `effect.dispatch_cps`
- `handle` boundary: shared 단계가 prompt tag와 `effect.extend`를 구성하고,
  target 단계가 handler marker와 evidence 저장소를 구체화한다.

### 메모리 관리: Reference Counting

Cranelift 타겟에서는 **Reference Counting**을 채택한다.

- **+1 convention**: 생산자가 소유, 소비자가 retain, 마지막 사용에서 release
- **Object 헤더**: `[-8 bytes] refcount: u32 + type_id: u32 | [0 bytes] first field`
- Cycle 처리는 당면 과제가 아님 (함수형 언어 특성상 cycle이 드묾)
- 타입을 지우기 전에 ownership와 RTTI를 계획하고, 그 계획으로 retain/release를 생성한다.

---

## References

### 언어 설계

- [Koka](https://koka-lang.github.io/) - Algebraic effects 선구자
- [Unison](https://www.unison-lang.org/) - Abilities (effect 시스템)
- [Rust](https://www.rust-lang.org/) - 문법, struct/enum 스타일
- [Gleam](https://gleam.run/) - 문법 참조

### 구현

- [Cranelift](https://cranelift.dev/) - 네이티브 코드 생성

### 논문

- "Liberating Effects with Rows and Handlers" (Koka)
- "Do Be Do Be Do" (Frank)
- "Effekt: Capability-passing style for type- and effect-safe,
  extensible effect handlers"
- "Perceus: Garbage-Free Reference Counting with Reuse" (Koka)
