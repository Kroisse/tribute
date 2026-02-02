# WasmGC Yield Bubbling 타입 시스템

> 이 문서는 WasmGC 백엔드의 yield bubbling 메커니즘에서 발생하는 타입 시스템 문제와 해결책을 정의한다.

## Overview

WasmGC (Stack Switching 미지원) 환경에서 ability operation은 yield bubbling
방식으로 구현된다. 이 과정에서 effectful 함수는 정상 반환값 또는
continuation을 반환해야 하므로 통일된 반환 타입이 필요하다.

**핵심 결정**: Koka 스타일의 `YieldResult` 구조체 사용

---

## 배경

### WasmGC 타입 계층 제약

WasmGC에서 `anyref`와 `funcref`는 별개의 타입 계층에 속한다:

```text
any
├── eq
│   ├── i31
│   ├── struct
│   └── array
└── (separate hierarchy)

func (funcref)
└── (no common supertype with anyref)
```

따라서 "결과값 또는 continuation" 같은 union 타입을 직접 표현할 수 없다.

### Yield Bubbling 동작

`implementation.md`의 "WasmGC (Stack Switching 없음)" 섹션에서 정의된 대로:

```rust
fn effectful_call(ev: *const Evidence) -> Result<T, Yield> {
    let result = do_operation(ev)
    if is_yielding() {
        return Yield(build_continuation())
    }
    Ok(result)
}
```

모든 effectful 호출 후 yield 상태를 체크하고, yield 발생 시 continuation을 반환해야 한다.

---

## 문제 정의

### 발생 오류

```text
error: func 4 failed to validate
Caused by: type mismatch: expected funcref, found (ref null $type)
```

### Root Cause

1. `counter()` 함수가 `State::get!` 수행
2. shift 확장으로 `wasm.return ref.null any` 삽입
3. 함수 반환 타입이 anyref로 변경됨
4. caller인 `main::__lambda_2`는 여전히 i32 기대
5. 타입 불일치 발생

### 탐색 결과

| 문제점 | 현재 상태 |
| ------ | --------- |
| Shift 변환이 로컬 전용 | caller 업데이트 없음 |
| Call graph 추적 없음 | wasm backend에 인프라 부재 |
| Effect row 미활용 | typeck 정보가 wasm까지 전달 안됨 |

---

## 대안 분석

### Koka 컴파일러 연구

Koka는 유사한 문제를 다음과 같이 해결한다:

- **Full effect monomorphization 사용하지 않음**
- Type-selective transformation: total 코드는 continuation 기계 건너뜀
- Evidence는 항상 전달 (8바이트 포인터)
- **Tail-resumptive 최적화로 ~90% 핸들러 최적화**

**핵심 통찰**: Koka는 `Result<T, Yield>` 패턴 사용

### 고려한 접근법들

| 접근법 | 장점 | 단점 | 채택 |
| ------ | ---- | ---- | ---- |
| **A: Call graph + anyref 전파** | IR 기반, Direct yielder만 boxing | Cascading 전파 필요, 복잡한 unboxing | ❌ |
| **B: YieldResult 구조** | 통일된 반환 타입, 전파 불필요 | 모든 effectful 호출 boxing | ✅ |
| **Effect row 활용** | semantic 정확성 | lowering 중 정보 손실 | ❌ |
| **Full effect monomorphization** | 최적 성능 | 코드 폭발, 복잡성 | ❌ |

---

## 설계 결정

### YieldResult 구조체

```wasm
(type $YieldResult (struct
  (field $tag i32)       ;; 0 = Done, 1 = Yielded
  (field $value anyref)  ;; 결과값 또는 continuation
))
```

### 변환 규칙

#### Effectful 함수 반환

```rust
// 원본
fn counter() ->{State(Int)} Int { ... }

// 변환 후
wasm.func @counter() -> (ref $YieldResult) {
    // ... ability operation ...
    wasm.if %yield_state -> (ref $YieldResult) {
        // Yield path
        %result = wasm.struct_new $YieldResult (i32.const 1, %cont)
        wasm.return %result
    } else {
        // Done path
        %boxed = wasm.i31_new %n
        %result = wasm.struct_new $YieldResult (i32.const 0, %boxed)
        wasm.return %result
    }
}
```

#### Call Site 변환

```text
// 원본
let result = call @effectful_func()
use(result)

// 변환 후
let yield_result = call @effectful_func() : ref $YieldResult
if yield_result.tag != 0 {
    return yield_result  // yield 전파
}
let result = unbox(yield_result.value)
use(result)
```

### 선택 이유

1. **Generic 함수와 자연스러운 상호작용**: 효과 다형성 처리 용이
2. **Cascading 전파 불필요**: 모든 effectful 함수가 동일한 타입 반환
3. **Koka에서 검증된 패턴**: 실전에서 효과 입증
4. **최적화 가능**: Tail-resumptive, inlining으로 오버헤드 제거 가능

---

## Generic 함수와의 상호작용

### 문제: Effect Polymorphism

```rust
fn map(a, b)(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b) {
    // f를 call_indirect로 호출
    // f가 yield할지 컴파일 타임에 모름
}
```

`generics.md`에 따르면:

- 타입 다형성: Monomorphization
- 효과 다형성: Evidence passing (monomorphization 아님)

### 해결책: 보수적 접근

효과 변수를 가진 함수 파라미터를 받는 함수는 잠재적 yielder로 간주:

```rust
fn is_potentially_yielding(func: &FuncDef) -> bool {
    // 1. Direct yielder: shift 포함
    if function_body_can_yield(func) { return true; }

    // 2. 효과 변수 포함 함수 파라미터 있음
    if has_effectful_function_param(func) { return true; }

    // 3. Transitive: yielding 함수 호출
    if calls_yielding_function(func) { return true; }

    false
}
```

### 향후 최적화

- 호출 사이트에서 `f`가 순수임이 알려지면 최적화된 경로 사용
- Tail-resumptive handler: YieldResult 생성 건너뜀
- Inlining: boxing/unboxing 제거

---

## Handler Lambda 예외 처리

Handler arm lambda와 computation lambda는 다르게 처리:

| Lambda 종류 | 반환 타입 | 설명 |
| ----------- | --------- | ---- |
| Computation lambda | `ref $YieldResult` | 실제 결과값 또는 yield |
| Handler arm lambda | `funcref` | Continuation 호출 결과 |

기존 `should_adjust_handler_return_to_i32()` 로직을 확장하여 이 구분을 유지한다.

---

## 코드 예시

### 원본 (ability_core.trb)

```rust
fn counter() ->{State(Int)} Int {
    let n = State::get!;
    State::set!(n + 1);
    n
}

fn main() {
    handle {
        let a = counter();
        let b = counter();
        a + b
    } with State(init: 0) {
        get!() -> k => k(state),
        set!(new) -> k => k((), state: new),
    }
}
```

### 변환된 IR

```mlir
// counter: YieldResult 반환
wasm.func @counter() -> (ref $YieldResult) {
    wasm.if %yield_state -> (ref $YieldResult) {
        %result = wasm.struct_new $YieldResult (i32.const 1, %cont)
        wasm.return %result
    } else {
        %boxed = wasm.i31_new %n
        %result = wasm.struct_new $YieldResult (i32.const 0, %boxed)
        wasm.return %result
    }
}

// main::__lambda_2: 동일한 YieldResult 반환
wasm.func @main::__lambda_2() -> (ref $YieldResult) {
    %res_a = wasm.call @counter() : (ref $YieldResult)
    %tag_a = wasm.struct_get $YieldResult 0 %res_a
    wasm.if (i32.ne %tag_a (i32.const 0)) -> (ref $YieldResult) {
        wasm.return %res_a  // yield 전파
    } else {
        %a_any = wasm.struct_get $YieldResult 1 %res_a
        %a = wasm.i31_get_s (wasm.ref_cast i31ref %a_any)
        // ... 계속 ...
    }
}
```

---

## 향후 고려사항

1. **Stack Switching 도입 시**: 이 전체 메커니즘 제거 가능
2. **Tail-resumptive 최적화**: ~90% 핸들러에서 YieldResult 생성 제거
3. **효과 기반 specialization**: 성능 critical path에서 선택적 적용

---

## References

- [Generalized Evidence Passing for Effect Handlers][evidence-passing] (Koka)
- [Effect Handlers, Evidently][handlers-evidently] (Scoped Resumption)
- [libmprompt] (Delimited Continuation Runtime)
- `new-plans/implementation.md` - Tribute ability implementation strategy
- `new-plans/generics.md` - Tribute generics and effect polymorphism

<!-- markdownlint-disable MD013 -->
[evidence-passing]: https://www.microsoft.com/en-us/research/publication/generalized-evidence-passing-for-effect-handlers/
<!-- markdownlint-enable MD013 -->
[handlers-evidently]: https://dl.acm.org/doi/10.1145/3408981
[libmprompt]: https://github.com/koka-lang/libmprompt
