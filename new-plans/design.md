# Tribute Language Design

> *This is not the greatest language in the world, no. This is just a tribute.*

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

| 항목 | 선택 |
|------|------|
| 모듈 구분자 | `::` |
| 메서드 호출 스타일 | UFCS (`.`) |
| 타입 선언 | `struct` (product) / `enum` (sum) |
| Ad-hoc polymorphism | 없음 (명시적 전달) |
| 이름 해소 | Type-directed (use 범위 내) |

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

| 속성 | 선택 | 이유 |
|------|------|------|
| Delimited | ✅ | prompt까지만 캡처, 합성 가능 |
| One-shot | ✅ | 구현 단순, 대부분의 실용적 ability 지원 |
| Multi-shot | ❌ | nondeterminism 포기, 복잡도 감소 |

### One-shot의 의미

Continuation은 **linear 타입**으로 취급한다:
- 반드시 1번 사용하거나 명시적으로 버려야 함
- 사용: `k(value)` 로 resume
- 버림: `drop(k)` 또는 와일드카드 바인딩

### 지원 가능한 Ability 패턴

```
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

```
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

### Multi-Level IR

MLIR의 dialect 개념을 차용하여 여러 수준의 IR이 한 모듈 내에 공존할 수 있다.
각 lowering pass는 특정 dialect만 변환하고 나머지는 보존한다.

```
┌─────────────────────────────────────────────────────┐
│                    Module                           │
│  ┌───────────────────────────────────────────────┐  │
│  │ Block                                         │  │
│  │  ┌─────────────┐ ┌─────────────┐             │  │
│  │  │ Gleam.Case  │ │ Arith.Add   │  ...        │  │
│  │  │ (dialect A) │ │ (dialect B) │             │  │
│  │  └─────────────┘ └─────────────┘             │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Dialect 목록

| Dialect | 수준 | 설명 |
|---------|------|------|
| Surface | 고 | 파싱 직후, 문법 그대로 |
| Typed | 고 | 타입/effect 정보 부착 |
| Core | 중 | desugaring 후, 패턴 매칭 분해 |
| Effect | 중 | effect handler 명시적 표현 |
| Arith | 공유 | 산술 연산 (모든 수준에서 사용) |
| WasmGC | 저 | WasmGC 타겟 전용 |
| Cranelift | 저 | Cranelift 타겟 전용 |

### Compilation Pipeline

```
Tribute Source
    │
    ▼ Parse
Surface AST
    │
    ▼ Type Inference + Effect Inference
Typed HIR
    │
    ▼ Desugar (UFCS, use, etc.)
    │
    ▼ Pattern Match Compilation
    │
Core IR + Effect Dialect
    │
    ▼ Effect Lowering
    │   ├─ Handle → Prompt/Shift
    │   └─ Continuation 명시화
    │
Core IR (effects resolved)
    │
    ▼ Optimization Passes
    │   ├─ Inlining
    │   ├─ Dead Code Elimination
    │   ├─ Constant Folding
    │   └─ Tail Call Optimization
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
WasmGC Dialect      Cranelift Dialect     (Future: BEAM?)
    │                     │
    ▼                     ▼
Binaryen              Cranelift
    │                     │
    ▼                     ▼
.wasm                 native binary
```

---

## IR Design

### Common Infrastructure

```rust
/// SSA Value - 모든 dialect에서 공유
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(u32);

/// Block - 여러 dialect의 op들이 공존
pub struct Block {
    pub id: u32,
    pub params: Vec<(Value, Type)>,
    pub ops: Vec<Op>,
    pub terminator: Terminator,
}

/// Region - 중첩 구조 (handler body 등)
pub struct Region {
    pub blocks: Vec<Block>,
}
```

### Operation Enum

```rust
pub enum Op {
    // -------- Effect Dialect --------
    /// Effect 수행
    Perform {
        effect: EffectRef,
        operation: String,
        args: Vec<Value>,
        result: Value,
    },
    
    /// Handler 설치
    Handle {
        body: Region,
        clauses: Vec<HandlerClause>,
        result: Value,
    },
    
    // -------- Core Dialect --------
    /// Delimited continuation prompt
    PushPrompt {
        tag: PromptTag,
        body: Region,
        result: Value,
    },
    
    /// Continuation 캡처 + handler로 점프
    Shift {
        tag: PromptTag,
        continuation: Value,
    },
    
    /// Continuation resume
    Resume {
        continuation: Value,
        value: Value,
        result: Value,
    },
    
    /// Continuation abort
    Abort {
        continuation: Value,
    },
    
    // -------- Arith Dialect --------
    Add { lhs: Value, rhs: Value, result: Value },
    Sub { lhs: Value, rhs: Value, result: Value },
    // ...
    
    // -------- WasmGC Dialect --------
    StructNew { type_idx: u32, fields: Vec<Value>, result: Value },
    StructGet { struct_ref: Value, field_idx: u32, result: Value },
    // ...
    
    // -------- Cranelift Dialect --------
    Load { ptr: Value, offset: i32, result: Value },
    Store { ptr: Value, offset: i32, value: Value },
    // ...
}
```

### Handler Clause

```rust
pub enum HandlerClause {
    /// Abort: continuation을 받지 않음
    Abort {
        effect: EffectRef,
        operation: String,
        params: Vec<Value>,
        body: Region,
    },
    
    /// WithContinuation: continuation을 받아서 명시적 처리
    Resume {
        effect: EffectRef,
        operation: String,
        params: Vec<Value>,
        continuation: Value,  // Linear 타입
        body: Region,
    },
}
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

```
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

### Effect 구현 전략

**setjmp/longjmp + 스택 복사** 방식 사용:

```rust
#[repr(C)]
pub struct Continuation {
    registers: JmpBuf,
    stack_segment: Box<[u8]>,
    stack_base: *mut u8,
    stack_size: usize,
}

impl Continuation {
    /// 현재 continuation 캡처 (shift)
    pub fn capture(prompt: &Prompt) -> Self { ... }
    
    /// Continuation 실행 (resume)
    pub fn resume(self, value: Value) -> ! { ... }
    
    /// Continuation 버림 (abort)
    pub fn abort(self) { ... }
}
```

참고: [libmprompt](https://github.com/koka-lang/libmprompt) (Koka 언어의 effect 런타임)

### 메모리 관리

Cranelift 타겟에서는 GC가 필요하다. 선택지:

1. **Reference Counting**: 단순하지만 cycle 처리 필요
2. **Boehm GC**: 보수적 GC, 쉬운 통합
3. **Custom GC**: 정밀한 제어, 높은 구현 비용

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

- [libmprompt](https://github.com/koka-lang/libmprompt) - Delimited continuation 런타임
- [Binaryen](https://github.com/WebAssembly/binaryen) - WasmGC 최적화
- [Cranelift](https://cranelift.dev/) - 네이티브 코드 생성
- [WASM Stack Switching](https://github.com/WebAssembly/stack-switching) - Continuation proposal

### 논문

- "Liberating Effects with Rows and Handlers" (Koka)
- "Do Be Do Be Do" (Frank)
- "Effekt: Capability-passing style for type- and effect-safe, extensible effect handlers"
- "Perceus: Garbage-Free Reference Counting with Reuse" (Koka)
