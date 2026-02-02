# TrunkIR Design

> 이 문서는 Tribute 컴파일러의 중간 표현(IR)인 TrunkIR의 설계를 정의한다.

## Overview

TrunkIR은 Tribute 소스 코드에서 Wasm/네이티브 바이너리로 내려가는 중심 intermediate representation이다.

### 설계 원칙

- **SSA 기반**: Block arguments를 사용 (φ 노드 대신)
- **Dialect 네임스페이싱**: 모든 연산은 `<dialect>.<operation>` 형태
- **Multi-level**: 여러 수준의 dialect이 한 모듈 내에 공존
- **Structured Control Flow**: 임의 CFG가 아닌 structured control flow 사용

### Dialect 계층 구조

```
┌─────────────────────────────────────────────────────────┐
│ Infrastructure                                          │
│   core             module, unrealized_conversion_cast   │
│   type             struct, enum, ability 정의           │
├─────────────────────────────────────────────────────────┤
│ High-level (언어 의미론)                                 │
│   src              미해소 호출, 추론 전 타입              │
│   ability          handle, perform, resume, abort       │
│   closure          new, func, env                       │
│   adt              struct, variant, array, ref          │
├─────────────────────────────────────────────────────────┤
│ Mid-level (lowered, 타겟 독립)                          │
│   cont             push_prompt, shift, resume, drop     │
│   func             func, call, call_indirect, constant  │
│   scf              case, yield, (loop, continue, break) │
│   arith            산술, 비교, 비트 연산                  │
│   mem              data, load, store                    │
├─────────────────────────────────────────────────────────┤
│ Low-level (타겟 종속)                                   │
│   wasm.*           Wasm 3.0 + WasmGC                    │
│   clif.*           Cranelift                            │
└─────────────────────────────────────────────────────────┘
```

---

## Infrastructure Dialects

### core Dialect

모듈 구조와 dialect 간 변환을 위한 기반 연산.

#### 연산

```
core.module : (name: String, body: Region) -> Module
    모듈 정의

core.unrealized_conversion_cast : (value: T) -> U
    Dialect 변환 중 임시 캐스트
    lowering 완료 후 모두 제거되어야 함
```

#### 타입

```
// 정수 (비트폭 명시)
i1, i8, i16, i32, i64, i128, ...

// 부동소수점
f32, f64

// 연속 메모리
Bytes           // raw 바이트
Array<T>        // 연속 배열

// GC 참조
ref<T>          // non-nullable
ref<T>?         // nullable
```

#### 식별자 타입

```rust
/// Interned 문자열 (단순 이름 또는 qualified path)
/// lasso::Spur로 구현되어 4바이트, O(1) 비교
///
/// Qualified path는 `::`로 구분된 문자열로 저장됨 (e.g., "std::List::map")
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Symbol(lasso::Spur);

impl Symbol {
    pub fn new(text: &'static str) -> Self;
    pub fn from_dynamic(text: &str) -> Self;
    pub fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R;
}
```

Qualified path 조작은 `tribute-ir`의 `ModulePathExt` trait을 통해 제공:

```rust
// tribute-ir/src/lib.rs
pub trait ModulePathExt {
    /// 마지막 segment 반환 (e.g., "std::List::map" → "map")
    fn last_segment(&self) -> Symbol;

    /// 부모 경로 반환 (e.g., "std::List::map" → Some("std::List"))
    fn parent_path(&self) -> Option<Symbol>;

    /// 두 경로 연결 (e.g., "std::List" + "map" → "std::List::map")
    fn join_path(&self, other: Symbol) -> Symbol;

    /// 단일 segment인지 확인 (`::`가 없으면 true)
    fn is_simple(&self) -> bool;
}

impl ModulePathExt for Symbol { ... }
```

- `Symbol`: 단순 식별자 또는 qualified path, lasso로 interning되어 4바이트, 비교 O(1)
- Qualified path는 `Symbol` 내부에 `::` 구분자로 저장
- Path 조작이 필요한 곳에서는 `ModulePathExt` trait 사용 (`tribute-ir`에서 제공)

---

## High-level Dialects

### tribute Dialect

파싱 직후의 미해소 상태와 Tribute 언어 레벨 연산/타입을 표현한다.
타입 추론과 이름 해소 후 대부분 제거된다.

#### 연산

```
tribute.call : (name: Symbol, args...) -> T
    Unqualified 호출 (foo(x, y) 형태), 미해소
    name은 단순 이름 또는 qualified path (e.g., "foo" 또는 "math::double")

tribute.var : (name: Symbol) -> T
    변수 참조, 미해소

tribute.lambda : (params: [(String, Type?)], body: Region) -> T
    람다 (캡처 분석 전)

tribute.case : (scrutinee) -> T { body }
    패턴 매칭
```

#### 타입

```
tribute.type     // 미해소 타입 참조
tribute.int      // 정수 타입
tribute.nat      // 자연수 타입
```

#### Invariant

| Pass 완료 후 | 조건                            |
| ------------ | ------------------------------- |
| Resolution   | tribute.var/call 대부분 해소    |
| Type Check   | 타입 추론 변수 모두 해소 (front-end에서 처리) |

### ability Dialect

Ability (algebraic effect) 실행 연산.

```
ability.perform : (ability: AbilityRef, op: String, args...) -> T
    Ability operation 수행

ability.resume : (continuation: Continuation<T>, value: T) -> U
    Continuation resume

ability.abort : (continuation: Continuation<T>) -> !
    Continuation 버림 (linear type 만족)
```

### tribute.handle

Handler 구문 `handle expr { ... }`는 `tribute.handle`로 표현된다:
```
tribute.handle(expr) {
    { value } -> ...
    { Op(args) -> k } -> ...
}
```

`tribute.case`와 유사한 구조지만, handler pattern (`{ ... }`)만 허용된다.

Handler lowering pass에서 `cont.push_prompt` + handler dispatch로 변환된다.

### closure Dialect

클로저 생성 및 분해 연산. 클로저는 함수 참조와 캡처된 환경의 조합이다.
타겟별로 다르게 lowering된다 (wasm: funcref + struct, native: 함수 포인터 + 힙).

```
closure.new : @func_ref(captures...) -> Closure<T>
    클로저 생성 (캡처된 변수들 명시)

closure.func : (closure: Closure<T>) -> FuncRef
    클로저에서 funcref 추출

closure.env : (closure: Closure<T>) -> Env
    클로저에서 environment 추출
```

#### 클로저 호출 패턴

클로저는 분해 후 `func.call_indirect`로 호출:

```
%closure = closure.new @lambda_0, [%captured]
%fn = closure.func %closure
%env = closure.env %closure
func.call_indirect %fn(%env, %args...)  // env가 첫 번째 인자
```

### adt Dialect

Algebraic Data Type 연산. 타겟 독립적.

#### Struct (Product Type)

```
adt.struct_new : (type: StructType, fields...) -> ref<T>
    struct 인스턴스 생성

adt.struct_get : (ref: ref<T>, field: u32) -> FieldType
    필드 읽기

adt.struct_set : (ref: ref<T>, field: u32, value: FieldType) -> ()
    필드 쓰기
```

#### Variant (Sum Type)

```
adt.variant_new : (type: EnumType, tag: u32, fields...) -> ref<T>
    variant 인스턴스 생성

adt.variant_tag : (ref: ref<T>) -> u32
    variant의 tag 읽기

adt.variant_get : (ref: ref<T>, field: u32) -> FieldType
    variant의 필드 읽기
```

#### Array

```
adt.array_new : (type: ArrayType, size: i32, init?) -> ref<Array<T>>
    배열 생성

adt.array_get : (ref: ref<Array<T>>, index: i32) -> T
    요소 읽기

adt.array_set : (ref: ref<Array<T>>, index: i32, value: T) -> ()
    요소 쓰기

adt.array_len : (ref: ref<Array<T>>) -> i32
    배열 길이
```

#### Reference

```
adt.ref_null : (type: RefType) -> ref<T>?
    null 참조 생성

adt.ref_is_null : (ref: ref<T>?) -> i1
    null 검사

adt.ref_cast : (ref: ref<T>) -> ref<U>
    참조 타입 캐스트 (런타임 검사)
```

#### 리터럴

```
adt.text_const : (literal: Text) -> Text
    텍스트 상수

adt.bytes_const : (bytes: [u8]) -> Bytes
    바이트열 상수
```

---

## Mid-level Dialects

### cont Dialect

Delimited continuation 연산. ability가 이 수준으로 lowering된다.

```
cont.push_prompt : (tag: PromptTag, body: Region) -> T
    Prompt 설치, body 실행

cont.shift : (tag: PromptTag, handler: Region) -> !
    Continuation 캡처 후 handler로 점프

cont.resume : (continuation: Continuation<T>, value: T) -> U
    Continuation 실행

cont.drop : (continuation: Continuation<T>) -> ()
    Continuation 해제 (linear type 만족)
```

### func Dialect

함수 정의 및 호출. MLIR 스타일을 따름.

```
func.func : (name: Symbol, type: Type, body: Region) -> FuncDef
    함수 정의

func.call : @callee(args...) -> T
    Direct call (callee는 symbol attribute)

func.call_indirect : (callee: Value, args...) -> T
    Indirect call (callee는 SSA value, plain funcref만)

func.constant : @func_ref -> FuncValue
    함수 심볼에서 일급 함수 값 생성 (indirect call용)

func.tail_call : @callee(args...) -> !
    Tail call (반환하지 않음)

func.return : (value: T) -> !
    함수에서 반환

func.unreachable : () -> !
    도달 불가 지점 (trap)
```

#### Direct vs Indirect Call

```
// Direct call: callee가 컴파일 타임에 알려진 경우
func.call @add(%x, %y) : (i32, i32) -> i32

// Indirect call: callee가 런타임 값인 경우 (plain funcref)
%f = func.constant @add : fn(i32, i32) -> i32
func.call_indirect %f(%x, %y) : (i32, i32) -> i32
```

#### 람다 Lowering

람다는 세 단계로 lowering된다:

```
// 1. 파싱 직후 (캡처 분석 전)
%f = tribute.lambda (%x) {
    arith.add %x, %y        // %y는 외부 변수 (캡처 대상인지 아직 모름)
} -> tribute.type_var

// 2. 캡처 분석 후 (tribute → closure + func)
//    별도 함수로 추출되고, 캡처 변수가 명시됨
func.func @lambda_0(%env: ref<Env>, %x: i32) -> i32 {
    %y = adt.struct_get %env, 0 : i32
    %result = arith.add %x, %y : i32
    func.return %result
}
...
%f = closure.new @lambda_0, [%y] -> Closure<fn(i32) -> i32>

// 클로저 호출 시 (closure dialect 사용)
%fn = closure.func %f
%env = closure.env %f
func.call_indirect %fn(%env, %arg)

// 3. 타겟별 lowering (closure → wasm/clif)
//    Wasm: funcref + struct
//    Cranelift: 함수 포인터 + 힙 환경
```

### scf Dialect

Structured Control Flow. Tribute는 loop 구문이 없고 재귀만 사용한다.

#### 기본 연산

```
scf.case : (scrutinee: T, branches: [(Pattern, Region)]) -> U
    패턴 매칭, 모든 Region이 같은 타입 U를 yield

scf.yield : (values...) -> !
    Region에서 값 반환
```

#### Tail Call Inlining 결과물 (최적화 패스 산출)

```
scf.loop : (init: (T1, T2, ...), body: Region) -> U
    루프 (tail recursion 최적화 결과)

scf.continue : (values...) -> !
    루프 처음으로, 새 인자 전달

scf.break : (value: T) -> !
    루프 탈출, 결과 반환
```

### arith Dialect

산술, 비교, 비트 연산. 모든 정수/부동소수점 타입 지원.

#### 산술

```
arith.const : (value: Immediate) -> T
    상수

arith.add : (lhs: T, rhs: T) -> T
arith.sub : (lhs: T, rhs: T) -> T
arith.mul : (lhs: T, rhs: T) -> T
arith.div : (lhs: T, rhs: T) -> T
arith.rem : (lhs: T, rhs: T) -> T
arith.neg : (value: T) -> T
```

#### 비교

```
arith.cmp_eq  : (lhs: T, rhs: T) -> i1
arith.cmp_ne  : (lhs: T, rhs: T) -> i1
arith.cmp_lt  : (lhs: T, rhs: T) -> i1
arith.cmp_le  : (lhs: T, rhs: T) -> i1
arith.cmp_gt  : (lhs: T, rhs: T) -> i1
arith.cmp_ge  : (lhs: T, rhs: T) -> i1
```

#### 비트 연산

```
arith.and : (lhs: T, rhs: T) -> T
arith.or  : (lhs: T, rhs: T) -> T
arith.xor : (lhs: T, rhs: T) -> T
arith.shl : (value: T, amount: T) -> T
arith.shr : (value: T, amount: T) -> T    // arithmetic
arith.shru : (value: T, amount: T) -> T   // logical
```

#### 타입 변환

```
arith.cast    : (value: T) -> U    // 부호 확장/축소
arith.trunc   : (value: T) -> U    // 절삭
arith.extend  : (value: T) -> U    // 확장
arith.convert : (value: T) -> U    // int ↔ float
```

### mem Dialect

저수준 메모리 연산. FFI 및 런타임 지원용.

```
mem.data : (bytes: [u8]) -> ptr
    Data section에 바이트 배치, 포인터 반환

mem.load : (ptr: ptr, offset: i32) -> T
    메모리 읽기

mem.store : (ptr: ptr, offset: i32, value: T) -> ()
    메모리 쓰기
```

---

## Low-level Dialects

### wasm Dialect

Wasm 3.0 + WasmGC 타겟.

#### Control

```
wasm.block : (body: Region) -> T
wasm.loop : (body: Region) -> ()
wasm.if : (cond: i32, then: Region, else: Region?) -> T
wasm.br : (target: BlockRef) -> !
wasm.br_if : (cond: i32, target: BlockRef) -> ()
wasm.return : (value: T) -> !
wasm.return_call : (callee: FuncRef, args...) -> !
```

#### Arithmetic (예시)

```
wasm.i32_add : (lhs: i32, rhs: i32) -> i32
wasm.i32_sub : (lhs: i32, rhs: i32) -> i32
wasm.i32_eq  : (lhs: i32, rhs: i32) -> i32
...
```

#### WasmGC

```
wasm.struct_new : (type: TypeIdx, fields...) -> ref
wasm.struct_get : (ref: ref, field: u32) -> T
wasm.struct_set : (ref: ref, field: u32, value: T) -> ()

wasm.array_new : (type: TypeIdx, size: i32, init?) -> ref
wasm.array_get : (ref: ref, index: i32) -> T
wasm.array_set : (ref: ref, index: i32, value: T) -> ()
wasm.array_len : (ref: ref) -> i32

wasm.ref_null : (type: HeapType) -> ref?
wasm.ref_is_null : (ref: ref?) -> i32
wasm.ref_cast : (ref: ref) -> ref
```

### clif Dialect

Cranelift 타겟. (상세 정의 추후)

```
clif.* : Cranelift IR에 대응하는 연산들
```

---

## Block Arguments와 SSA

TrunkIR은 SSA 기반이며, φ 노드 대신 block arguments를 사용한다.

```
func.func @sum(%xs: ref<List>, %acc: i32) -> i32 {
^entry:
    %tag = adt.variant_tag %xs
    scf.case %tag {
        0 -> {  // Empty
            scf.yield %acc
        }
        1 -> {  // Cons
            %h = adt.variant_get %xs, 0 : i32
            %t = adt.variant_get %xs, 1 : ref<List>
            %new_acc = arith.add %acc, %h : i32
            func.tail_call @sum(%t, %new_acc)
        }
    }
}
```

Tail call inlining 후:

```
func.func @sum(%xs: ref<List>, %acc: i32) -> i32 {
    scf.loop (%xs, %acc) -> i32 {
    ^body(%xs_cur: ref<List>, %acc_cur: i32):
        %tag = adt.variant_tag %xs_cur
        scf.case %tag {
            0 -> {
                scf.break %acc_cur
            }
            1 -> {
                %h = adt.variant_get %xs_cur, 0 : i32
                %t = adt.variant_get %xs_cur, 1 : ref<List>
                %new_acc = arith.add %acc_cur, %h : i32
                scf.continue %t, %new_acc
            }
        }
    }
}
```

---

## Compilation Pipeline

```
Tribute Source
    │
    ▼ Parse (CST)
    │
    ▼ Lower (CST → TrunkIR)
    │
TrunkIR [tribute, adt, ability, func, scf, arith, cont]
    │   (tribute.var, tribute.path, tribute.call, tribute.type 포함)
    │   (tribute.lambda 포함)
    │
    ▼ Name Resolution + Type Inference (interleaved)
    │   ┌─────────────────────────────────────────────┐
    │   │ 1. Basic Name Resolution                    │
    │   │    - qualified paths (List::empty)          │
    │   │    - constructors (Some, None)              │
    │   │    - unambiguous function names             │
    │   │    → tribute.path, 일부 tribute.var 해소    │
    │   │                                             │
    │   │ 2. First-pass Type Inference                │
    │   │    - constraint 수집                        │
    │   │    - tribute.type_var 해소 시작             │
    │   │                                             │
    │   │ 3. Type-directed Name Resolution (UFCS)     │
    │   │    - xs.map(f) → List::map(xs, f)          │
    │   │    - 첫 번째 인자 타입으로 함수 선택        │
    │   │    → 나머지 tribute.var, tribute.call 해소  │
    │   │                                             │
    │   │ 4. Complete Type Inference                  │
    │   │    - UFCS 해소 후 추가 constraint 수집      │
    │   │    - 모든 tribute.type_var 해소             │
    │   │    - effect row 통합                        │
    │   └─────────────────────────────────────────────┘
    │
    ▼ Capture Analysis
    │   tribute.lambda → closure.new (캡처 변수 명시)
    │
TrunkIR [tribute, adt, ability, closure, func, scf, arith]
    │
    ▼ Ability Lowering (Evidence Passing)
    │   ability.* → cont.* + func.call
    │
TrunkIR [tribute, adt, cont, func, scf, arith]
    │
    ▼ Optimization Passes
    │   - Tail Call Inlining (func.tail_call → scf.loop)
    │   - Inlining
    │   - Dead Code Elimination
    │   - Constant Folding
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼ Wasm Lowering                       ▼ Cranelift Lowering
    │                                     │
TrunkIR [wasm.*]                     TrunkIR [clif.*]
    │                                     │
    ▼ Emit                                ▼ Emit
    │                                     │
.wasm                                native binary
```

### Name Resolution + Type Inference 상세

Name resolution과 type inference가 interleaved되는 이유:

1. **UFCS 해소에 타입 필요**: `xs.map(f)`에서 `map`이 `List::map`인지 `Option::map`인지는 `xs`의 타입에 따라 결정됨
2. **타입 추론에 해소된 이름 필요**: `List::map`의 타입 시그니처를 알아야 결과 타입 추론 가능

따라서 두 pass가 순차적으로 완전히 분리될 수 없고, 상호작용하면서 진행된다.

```
┌────────────────┐     ┌────────────────┐
│ Name Resolution│ ←─→ │ Type Inference │
└────────────────┘     └────────────────┘
        ↓                      ↓
tribute.var/call 해소   tribute.type_var 해소
```

**기본 해소 (타입 불필요)**:
- `List::empty` → qualified path, 바로 해소
- `Some(x)` → constructor, 바로 해소
- `foo(x)` → 스코프에 `foo`가 하나만 있으면 바로 해소

**타입 기반 해소 (타입 필요)**:
- `xs.map(f)` → `xs`의 타입이 `List(a)`이면 `List::map` 선택
- `x <> y` → `x`, `y`의 타입이 `Text`이면 `Text::<>`, `List`이면 `List::<>`

---

## Pass Invariants

각 pass가 완료된 후 만족해야 하는 조건:

| Pass                     | Invariant                                        |
| ------------------------ | ------------------------------------------------ |
| Parse                    | 유효한 CST 구조                                  |
| Lower                    | 유효한 TrunkIR 구조                              |
| Basic Name Resolution    | tribute.path 없음, 일부 tribute.var 해소         |
| Type Inference (1차)     | 대부분의 tribute.type_var에 constraint 존재      |
| Type-directed Resolution | tribute.var, tribute.call 없음 (UFCS 포함)       |
| Type Inference (완료)    | tribute.type_var 없음, 모든 타입 구체화          |
| Capture Analysis         | tribute.lambda 없음, closure.new로 대체          |
| Ability Lowering         | ability.\* 없음                                  |
| Wasm Lowering            | wasm.\* 만 존재 (타겟이 Wasm일 때)               |
| Cranelift Lowering       | clif.\* 만 존재 (타겟이 native일 때)             |

---

## Data Types

### Primitive vs Library

| Primitive (core 타입)    | Library (adt로 정의) |
| ------------------------ | -------------------- |
| i1 ~ i128                | List (finger tree)   |
| f32, f64                 | Text (rope)          |
| Bytes                    | Option               |
| Array\<T\> (연속 메모리) | Result               |
| ref\<T\>, ref\<T\>?      | ...                  |

### 변환

```rust
// Array ↔ List
arr.to_list()
list.to_array()

// Bytes ↔ Text
bytes.to_text_utf8()
text.to_bytes()
```

---

## Open Questions

1. **클로저 표현**: `func.closure_new`의 구체적인 환경 캡처 방식
2. **Effect row 표현**: IR에서 effect polymorphism 표현 방식
3. **디버그 정보**: Source map 생성 방식
4. **Wasm Stack Switching**: cont dialect의 wasm lowering 전략

## Future Considerations

### Persistent Data Structure for Block Operations

현재 `Block::operations`는 `SmallVec`을 사용하지만, `im::Vector` 같은 persistent data structure를 고려할 수 있다.

**배경:**
- TrunkIR은 Salsa tracked struct로 immutable
- 현재 rewrite는 블록 전체를 재구축 (O(n) 복사)
- 대부분의 rewrite pass에서 변경되는 op은 소수

**im::Vector 사용 시 이점:**
- 구조적 공유로 변경된 부분만 새 노드 생성
- `update(index, new_op)` O(log n)
- 1000개 op 중 10개 변경 시: SmallVec은 1000개 복사, im::Vector는 ~100개 노드

**고려사항:**
- 작은 블록에서는 SmallVec이 캐시 지역성 면에서 유리
- Rewriter 설계를 surgical update 방식으로 변경해야 최대 이점
- 추가 의존성 (im crate)

**결정 시점:** 프로파일링에서 블록 복사가 병목으로 확인되면 재검토

### AST 제거하고 Tree-sitter CST에서 직접 TrunkIR로 lowering

현재 파이프라인: `Tree-sitter CST → AST (tribute-ast) → TrunkIR`

고려 사항:
- AST가 상당히 thin함 (대부분 concrete syntax 제거 정도)
- TrunkIR이 이미 `Location`으로 소스 위치 보존
- 중간 표현 하나 제거 시 코드/메모리 절약 가능

Trade-off:
- Tree-sitter CST는 더 verbose하고 cursor 관리 필요
- AST가 다른 도구들 (formatter, linter, IDE)에 유용할 수 있음
- 에러 리포팅이 AST 수준에서 더 쉬울 수 있음

결정 시점: AST의 다른 용도가 명확해지면 재검토
