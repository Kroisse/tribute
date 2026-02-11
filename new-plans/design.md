# Tribute Language Design

> _This is not the greatest language in the world, no. This is just a tribute._

## Overview

Tribute는 정적 타입과 algebraic effects를 갖춘 함수형 프로그래밍 언어이다.

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
fn(a) ->{} b    // 순수 함수
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
- **Stack Switching** (Phase 3): delimited continuation 네이티브 지원 예정
- **크로스 플랫폼**: 브라우저 + WASI 런타임

### Effect 구현 전략

#### 현재 (Stack Switching 없음)

Asyncify 변환 또는 CPS 변환 필요 (복잡함)

#### 미래 (Stack Switching 도입 후)

```wasm
;; Prompt 설치
(cont.new $handler_type
  (block $prompt
    ;; body 실행
    ;; effect 발생 시 suspend
  )
)

;; Effect 수행 (shift)
(suspend $effect_tag)

;; Handler에서 resume
(resume $cont (local.get $value))
```

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
  cont → libmprompt 호출, evidence, boxing(RC)
  + func/arith/scf/adt/mem → clif.* dialect 변환

trunk-ir-cranelift-backend/      언어 독립적 Cranelift codegen
  clif.* dialect → Cranelift IR → 네이티브 바이너리
```

- `clif.*` dialect은 Cranelift IR과 1:1 대응 (`wasm.*`과 대칭)
- `trunk-ir-cranelift-backend`은 `trunk-ir`만 의존 (Tribute 독립적)

### Effect 구현 전략: libmprompt

[libmprompt](https://github.com/koka-lang/libmprompt) (Koka 언어의 effect 런타임)을
사용하여 `cont.*` dialect을 직접 변환한다. WASM의 trampoline과 달리
라이브 변수 캡처/state struct가 불필요 — libmprompt가 스택 자체를 관리.

```c
// libmprompt C FFI
void* mp_prompt(mp_prompt_t* p, mp_start_fun_t* fun, void* arg);
void* mp_yield(mp_prompt_t* p, mp_yield_fun_t* fun, void* arg);
void* mp_resume(mp_resume_t* r, void* result);
void  mp_resume_drop(mp_resume_t* r);
```

핵심 변환:

- `cont.push_prompt` → `mp_prompt` 호출
- `cont.shift` → `mp_yield` 호출
- `cont.resume` → `mp_resume` 호출
- `cont.drop` → `mp_resume_drop` 호출

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
- [ ] Delimited continuation lowering
  - [ ] Cranelift: setjmp + 스택 복사
  - [ ] WasmGC: CPS 또는 Stack Switching

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
