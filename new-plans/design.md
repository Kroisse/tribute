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
struct User { name: Text, age: Int }
enum Option(a) { None, Some(a) }

// 함수 정의
fn add(x: Int, y: Int) -> Int {
    x + y
}

// 패턴 매칭
fn describe(value: Option(Int)) -> Text {
    case value {
        Some(n) -> "got: " <> Int::to_string(n)
        None -> "nothing"
    }
}

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
pub enum List(a) {
    Empty
    Cons(a, List(a))
}

pub mod List {
    pub fn empty() -> List(a) { Empty }
    pub fn map(xs: List(a), f: fn(a) -> b) -> List(b) { ... }
}

// UFCS 사용 (인자 없으면 괄호 생략)
let xs = List::empty()
let len = xs.len              // List::len(xs)
let ys = xs.map(fn(x) x + 1)  // List::map(xs, ...) 로 해석
```

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

Continuation은 **linear 타입**으로 취급한다:

- 반드시 1번 사용하거나 명시적으로 버려야 함
- 사용: `k(value)` 로 resume
- 버림: `drop(k)` 또는 와일드카드 바인딩

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

### Canonical Prelude List

The embedded prelude defines the language's canonical collection type:

```rust
enum List(a) {
    Empty
    Cons(a, List(a))
}
```

`Empty` is variant 0 and `Cons` is variant 1. List syntax is compiler-owned
surface sugar for this exact prelude declaration, not for whichever type named
`List` is visible in source:

```rust
[]        // Empty
[a, b, c] // Cons(a, Cons(b, Cons(c, Empty)))
```

Element expressions are evaluated exactly once from left to right. Construction
then starts with `Empty` and folds the already evaluated values from right to
left into `Cons`. An empty literal has type `List(a)` for a fresh `a`, which can
be fixed by its expected type. User declarations named `List`, `Empty`, or
`Cons` do not capture list syntax.

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

| 수준           | Dialect                     | 설명                                |
| -------------- | --------------------------- | ----------------------------------- |
| Infrastructure | core, type                  | 모듈 구조, 타입 정의                |
| High-level     | src, ability, adt           | 미해소 호출, ability, ADT           |
| Mid-level      | cont, func, scf, arith, mem | Continuation, 함수, 제어 흐름, 산술 |
| Low-level      | wasm._, clif._              | 타겟별 연산                         |

### Compilation Pipeline

```text
Tribute Source
    │
    ▼ Parse
TrunkIR [src, type, adt, ability, func, scf, arith]
    │
    ▼ Type Inference + Name Resolution
TrunkIR [type, adt, ability, func, scf, arith]
    │
    ▼ Ability Lowering (Evidence Passing)
TrunkIR [type, adt, cont, func, scf, arith]
    │
    ▼ Optimization Passes
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼ Wasm Lowering                       ▼ Cranelift Lowering
TrunkIR [wasm.*]                     TrunkIR [clif.*]
    │                                     │
    ▼                                     ▼
.wasm                                native binary
```

---

## Target: WasmGC

### 장점

- **네이티브 GC**: 자체 GC 구현 불필요
- **Tail calls**: WASM 3.0에서 표준화
- **크로스 플랫폼**: 브라우저 + WASI 런타임

### Effect 구현 전략

WasmGC는 현재 주요 구현 경로가 아니다. 다시 활성화할 경우에도
언어 레벨 effect lowering은 shared middle-end의 tail-call CPS 전략을 따른다.
따라서 WasmGC backend의 과제는 stack switching을 직접 사용해 effect를
구현하는 것이 아니라, 이미 lowered된 closure/evidence/call_indirect IR을
WasmGC 타입과 table 기반 호출로 정확히 emit하는 것이다.

과거의 yield bubbling / `YieldResult` trampoline 설계는 폐기된 대안으로
보존하되, 새 구현의 기준으로 삼지 않는다.

### 코드 생성

Binaryen을 최적화 백엔드로 사용:

```text
Core IR
    │
    ▼ Lower to WasmGC Dialect
WasmGC Ops
    │
    ▼ Emit (wasm-encoder 또는 Binaryen IR)
Unoptimized .wasm
    │
    ▼ wasm-opt -O3
Optimized .wasm
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

현재 native 구현은 libmprompt나 `cont.*` dialect 직접 lowering을 사용하지
않는다. Frontend/shared middle-end에서 continuation을 closure로 명시화하고,
effect operation은 evidence lookup 후 handler dispatch closure로 tail-call한다.

- `fn` ability operation → `ability.call` → `tr_dispatch_fn` 직접 호출
- `op` ability operation → `ability.perform` → `handler_dispatch` closure 호출
- `handle` boundary → evidence에 `tr_dispatch_fn`과 `handler_dispatch`를 가진
  marker 삽입

### 메모리 관리: Reference Counting

Cranelift 타겟에서는 **Reference Counting**을 채택한다.

- **+1 convention**: 생산자가 소유, 소비자가 retain, 마지막 사용에서 release
- **Object 헤더**: `[-8 bytes] refcount: u32 + type_id: u32 | [0 bytes] first field`
- Cycle 처리는 당면 과제가 아님 (함수형 언어 특성상 cycle이 드묾)
- Phase 2에서는 malloc/free 단순 할당으로 시작하고, 이후 RC 삽입

---

## Implementation Phases

### Phase 1: 기본 동작

- [ ] 파서 (UFCS, `::` 모듈 구분자)
- [ ] 단순 타입 체커 (effects 없이)
- [ ] Core IR
- [ ] Cranelift 백엔드 (tail calls만)
- [ ] WasmGC 백엔드 (tail calls만)

### Phase 2: Ability 시스템

- [ ] Ability 타입 추론
- [ ] Effect Dialect
- [ ] Handler 문법 파싱
- [ ] Tail-call CPS effect lowering
  - [ ] Evidence propagation
  - [ ] `fn` operation direct dispatch
  - [ ] `op` operation handler dispatch closure
  - [ ] Backend별 evidence runtime adaptation

### Phase 3: 최적화

- [ ] Effect 특화 최적화 (handler fusion, tail-resumptive 최적화)
- [ ] 공통 최적화 패스
- [ ] 벤치마킹 및 튜닝

---

## References

### 언어 설계

- [Koka](https://koka-lang.github.io/) - Algebraic effects 선구자
- [Unison](https://www.unison-lang.org/) - Abilities (effect 시스템)
- [Rust](https://www.rust-lang.org/) - 문법, struct/enum 스타일
- [Gleam](https://gleam.run/) - 문법 참조

### 구현

- [libmprompt](https://github.com/koka-lang/libmprompt) -
  Delimited continuation 런타임
- [Binaryen](https://github.com/WebAssembly/binaryen) - WasmGC 최적화
- [Cranelift](https://cranelift.dev/) - 네이티브 코드 생성
- [WASM Stack Switching](https://github.com/WebAssembly/stack-switching) -
  Continuation proposal

### 논문

- "Liberating Effects with Rows and Handlers" (Koka)
- "Do Be Do Be Do" (Frank)
- "Effekt: Capability-passing style for type- and effect-safe,
  extensible effect handlers"
- "Perceus: Garbage-Free Reference Counting with Reuse" (Koka)
