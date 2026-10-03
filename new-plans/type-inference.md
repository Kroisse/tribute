# Tribute Type Inference

> 이 문서는 Tribute의 타입 추론 시스템, 특히 row polymorphic effect typing과
> bidirectional typing의 통합을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목 | 선택 | 대안 (채택하지 않음) |
| ---- | ---- | -------------------- |
| 타입 추론 방식 | Bidirectional | 순수 HM, 전면 양방향 |
| Effect polymorphism | Row variables | Subtyping constraints |
| Effect 흐름 | Hybrid (inward + outward) | Frank (순수 inward), Koka (순수 outward) |
| 중복 label | 금지 | 허용 (scoped label과 `mask`) |
| Tail effect의 dispatch | Row 위치로 정적 선택 (tunneling) | 가장 가까운 handler |
| 암묵적 polymorphism | `fn(a) -> b` = `fn(a) ->{e} b` | 항상 명시 |

### Nominal Type Equality

Named types unify only when their resolved declaration identities match and
their type arguments unify pairwise. The source spelling is diagnostic
presentation, not semantic identity. Substitution and generalization preserve
the declaration identity of every named type.

### 레코드 생성 검사

레코드 필드 검증은 generic 타입 인자 추론과 독립적으로, 이름 해석으로 확정한 struct
선언 identity와 선언 순서의 필드 목록을 사용한다. 알 수 없는 필드, 중복 필드,
누락 필드는 `TypeChecking` 진단을 생성한다. 오류가 있으면 공개 컴파일 경로는
IR을 생성하지 않는다.

추론과 검사 과정에서 같은 레코드 노드를 다시 방문하더라도 필드 구성은 함수 검사
컨텍스트마다 한 번만 검증한다. 검증 여부는 해당 컨텍스트에만 저장하며, 선언 필드
목록을 조회한 뒤에만 기록한다. 소스에 실제로 반복해서 나타난 필드는 메시지와 span이
같아도 각각 진단한다. 소스나 선언을 편집한 뒤 새로 검사하면 필드 구성도 다시 검증한다.

재방문 시 생략하는 것은 필드 구성 검증뿐이다. 모든 자식 표현식은 계속 검사한다.
선언에 있는 필드에는 instantiate한 기대 타입을 전달하고
`TypeCoerce(actual, expected)`를 적용한다. 알 수 없는 필드도 RHS를 추론하며,
spread는 생성할 struct 타입을 기대 타입으로 삼아 방향성 있게 검사한다.
필드 구성 오류가 독립적인 자식 표현식 오류, spread 오류, 누락 필드 오류를 억제하지
않는다. 문맥에 따른 람다 검사와 최상위 `Never` 제거에도 다른 표현식 검사 위치와
동일한 규칙을 적용한다.

Generic struct의 필드 구성은 선언 identity로 결정하지만, 필드의 기대 타입은 현재
constructor 인스턴스의 타입 인자를 선언의 `BoundVar`에 치환하여 얻는다. 정상적인
constructor instantiate는 타입 매개변수마다 fresh `UniVar`를 생성한다. 아직 타입이
결정되지 않았다는 것은 인자 목록이 비어 있다는 뜻이 아니며, 이를 이유로 치환을
생략하지 않는다. 같은 레코드 NodeId의 추론·재검사·typed AST 변환은 함수 검사
컨텍스트 안에서 하나의 constructor 인스턴스를 공유한다. 결과 struct 타입뿐 아니라
constructor 타입에 포함된 callable 타입과 effect 변수도 재방문 때 새로 만들지 않는다.
서로 다른 레코드 노드는 독립적으로 instantiate하며, 함수나 소스를 다시 검사할 때
인스턴스를 새로 만든다.
개별 필드 조회·치환 결과를 캐시하지는 않는다. 모든 RHS와 spread는 재방문 때도
검사하며, 필드의 기대 타입은 공유한 constructor 인스턴스에서 매번 구한다.

Spread는 동일한 선언 identity와 호환되는 타입 인자를 요구한다. 명시적 필드로 모든
값을 덮어써도 spread의 타입 검사를 생략하지 않는다. 필드 타입에 나타나지 않는
phantom 인자도 nominal 타입의 일부이므로 기대 타입이나 spread를 통해 연결된다.
기대 타입이 없는 생성은 기존 순수 let 및 함수 일반화 규칙을 따른다. 레코드 생성
인스턴스를 공유하는 것은 일반화한 binding의 각 사용 위치를 하나의 인스턴스로
합치는 것이 아니다. 일반화한 값의 사용 위치는 기존대로 독립적으로 instantiate한다.

Enum variant 생성자도 값 참조의 `NodeId`별로 하나의 전체 constructor 인스턴스를
유지한다. `Box(+42)`의 인자 검사에서 선택한 `fn(Int) -> Boxed(Int)`를 typed AST의
생성자 참조까지 전달한다. 생성 결과 타입만 보존하고 변환 단계에서 생성자 자체를
새로 instantiate하면, 생성과 패턴 추출이 서로 다른 generic/specialized 레이아웃을
선택할 수 있다. 서로 다른 생성 위치와 일반화한 생성자 값의 사용 위치는 독립적이다.

Variant patterns preserve the constructor type constrained against the matched
value. This applies to let destructuring as well as case arms: conversion must
not instantiate a fresh constructor after the pattern has been solved.

### Equality, 표현식 검사, 공통 결과 타입

제약 solver는 다음 세 관계를 구별한다.

```text
TypeEq(A, B)                  -- 정확한 타입 equality
TypeCoerce(actual, expected)   -- 방향성 있는 표현식 검사
TypeJoin(sources, result)      -- 값을 반환하는 경로들의 공통 결과
```

`TypeEq`는 `Never`에 별도의 호환 규칙을 적용하지 않는다. 복합 타입과 ability
인자에도 재귀적으로 정확한 equality를 요구한다. `TypeCoerce`는 최상위 actual이
`Never`이면 미해결 expected 변수를 묶지 않고 허용하며, 그 외에는 equality를
요구한다. 복합 타입 내부에 재귀적인 variance를 도입하지 않는다.

`TypeJoin`은 `Never`를 제외하고 공통 타입을 선택하며, 서로 다른 정상 결과 타입은
거부한다. 모든 source가 `Never`이면 결과도 `Never`다. 결과 변수는 각 producer의
실제 타입과 분리하므로, 바깥 기대 타입으로 검사하더라도 producer의 `Never` 타입을
덮어쓸 수 없다. Case arm, 리스트 리터럴 원소, handle의 answer 경로에 이 관계를
사용한다.

관계는 substitution과 지연된 메서드 해석을 거치는 동안 source origin과 미해결
의존성을 보존한다. 지연된 호출의 실제 결과는 선택된 선언의 결과와 정확히 같아야
하며, 그 결과를 표현식으로 사용하는 위치는 별도로 검사한다. 같은 source node를
다시 추론하거나 검사할 때는 하나의 결과 관계를 공유한다. Handler의 arm을 수집하는
동안에는 answer 관계를 확정하지 않는다.

지연된 메서드에 전달하는 람다 리터럴은 반환 슬롯을 fresh 변수로 두고 실제 본문을
그 슬롯으로 검사한다. 메서드의 매개변수 타입이 정해지면 callable signature를
equality로 연결한다. 따라서 실제 본문이 `Never`여도 문맥의 반환 타입을 선택할 수
있다. 이 문맥 전달은 리터럴 정의에만 적용하며 기존 함수 값의 내부 타입은 바꾸지
않는다.

Solver는 equality, 공통 결과 관계, 표현식 검사와 지연된 producer 해석을 더 이상
진전이 없을 때까지 반복한다. 그 뒤에만 남은 순수 추론 변수의 공통 타입을 일반화하거나
equality로 확정할 수 있다. 미해결 producer를 `Never`의 근거로 삼지 않는다.
Answer의 재귀 참조는 의존성이며, 독립적인 정상 반환 경로가 아니다. 닫힌 answer
참조 순환의 결과는 순환 밖의 독립적인 source로 결정한다. 독립적인 source가 없거나
모두 `Never`이면 최소 결과는 `Never`다. 그 외에는 정상 source들이 공통 타입을
결정한다. 순환 밖에 미해결 producer가 있으면 이 결정을 미룬다.

지역 `let`의 임시 solver에도 같은 규칙을 적용한다. Effect 인자와 row를 통한
의존성을 포함하여 미해결 관계에 연결된 변수는 일반화에서 제외한다. 순수 binding의
관련 없는 변수는 계속 일반화할 수 있으며, 미해결 관계 하나 때문에 binding 전체를
monomorphic하게 바꾸지 않는다. 추론 변수를 지역 quantifier로 변환하면서 미해결
제약을 지워서는 안 된다.
임시 solver에서 생성한 row 변수의 identity도 이후 함수 추론과 지연된 메서드
해석의 fresh 변수 할당과 충돌하지 않아야 한다.

---

## Effect Row Syntax

### 기본 문법

```rust
// 구체적 ability만
fn foo() ->{State(Int), Logger} Nil

// Row 변수 (나머지를 나타냄)
fn bar(f: fn() ->{e} a) ->{e} a

// Row 변수 + 구체적 ability
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a

// 여러 row 변수의 합집합
fn compose(f: fn(a) ->{e1} b, g: fn(b) ->{e2} c) -> fn(a) ->{e1, e2} c

// 순수 함수 (빈 row)
fn pure(x: a) ->{} a

// 암묵적 polymorphic (생략 시): fresh row 변수
fn length(xs: List(a)) -> Nat
```

`std::io::Io`도 effect row의 concrete label로 전파된다. 다만 compiler-owned
ambient ability이므로 handler로 제거할 수 없고 `main`에 terminal effect로 남을
수 있다. Calling convention과 표준 API는 [io.md](io.md)를 따른다.

`List(a)`도 compiler-owned nominal identity를 갖는다. List literal과 list pattern은
resolver scope의 short name을 다시 검색하지 않고 이 identity로 constraint를 만든다.
Named type unification은 declaration identity와 type argument를 모두 비교하므로, 같은
`List` spelling을 가진 source declaration이 canonical list syntax를 capture할 수 없다.

### 암묵적 Effect Polymorphism

Effect annotation이 생략되면 그 위치마다 fresh row 변수를 쓴 것과 같다:

```rust
// 이 두 선언은 동일
fn length(xs: List(a)) -> Nat
fn length(xs: List(a)) ->{e} Nat
```

생략된 row는 서로 다른 변수이다. 고차 함수가 전달받은 함수의 effect를
전파하려면 같은 row 변수를 명시한다([Effect Annotation 규칙](#effect-annotation-규칙)).

### Row 구성 요소

Effect row는 다음으로 구성된다:

```text
Row ::= {}                    -- 빈 row (순수)
      | {A₁, A₂, ..., Aₙ}     -- 구체적 ability들
      | {e}                   -- row 변수
      | {A₁, ..., Aₙ, e}      -- 구체적 + row 변수
      | {e₁, e₂}              -- row 변수 합집합
      | {A₁, ..., e₁, e₂}     -- 혼합
```

### 중복 Label 금지

같은 ability가 row에 두 번 나타나면 컴파일 에러:

```rust
// OK: 타입 파라미터가 다르면 다른 ability로 취급
fn foo() ->{State(Int), State(String)} Nil

// Error: 동일한 ability 중복
fn bar() ->{State(Int), State(Int)} Nil
```

**이유**: Tribute는 한 ability instance의 여러 occurrence나 handler 인스턴스를
구별하는 의미론을 정의하지 않는다. Row는 instance의 집합이며, operation은 자기
instance만 가리킨다. 같은 instance의 handler가 중첩되면 `State::get()`은 자기
callable row의 명시 label에 묶인 handler로 가고, 같은 위치에서 바깥 handler를 따로
가리킬 방법은 없다. 중복을 허용하려면
occurrence의 순서나 이름으로 handler를 고르는 규칙이 함께 필요하다.

**향후 확장**: 동일 ability의 여러 인스턴스가 필요한 경우, effect row에서 이름을 붙일 수 있다:

```rust
// 향후 가능한 확장 (현재 미지원)
fn nested() ->{State(Int) as counter, State(Int) as total} Nil {
    let n = counter::get()
    let t = total::get()
    counter::set(n + 1)
    total::set(t + n)
}
```

이름이 네임스페이스 역할을 하여 `counter::get()`과 `total::get()`을 구분한다.

---

## Effect accumulation and retained relations

Effect accumulation is set union, not row equality. `RowUnion(sources, result)`
means that `result` contains exactly the union of its sources. Distinct open
sources remain independent; neither source is replaced by the other or equated
to the complete result. Closed-empty rows are identities and repeated union is
idempotent. Source annotation duplicate diagnostics remain distinct from
idempotent accumulation of effects performed more than once.

Closed function effect contracts are the expected side of boundary diagnostics;
the accumulated body requirements are the actual side. Ability argument counts
come from the exact ability declaration, regardless of row merge order.

Unsolved union relations survive substitution and deferred method resolution.
Generalization of a local scheme retains the relations in the type scheme and
quantifies their variables together with the body. Instantiation freshens the
body and relations with one shared mapping. Variables connected to the
surrounding environment or an unresolved producer are not independently
generalized. Handler subtraction must not reintroduce its consumed instance into
the outward row.

Handler의 차집합도 지연 가능한 semantic 제약이다. `RowRemoval(source, removed,
result)`는 `result = source − removed`를 뜻한다. 제거 대상은 닫힌 exact ability
instance 집합이며, source의 열린 tail에서 나중에 드러나는 label에도 적용한다.
타입 인자가 아직 미정이라 제거 여부가 모호하면 관계를 남긴다. 지역 스킴의
일반화와 인스턴스화는 이 관계를 합집합 관계와 함께 보존한다. 결과가 비어 있다는
이유로 source 자체를 빈 row로 닫아서는 안 된다.

### 모듈 수준 함수의 관계

모듈 수준 함수의 스킴은 선언 시그니처이며, annotation이 만든 합집합 관계만
담는다. 본문 검사가 남긴 관계는 스킴에 옮기지 않는다. 본문을 검사한 뒤
시그니처의 row 변수를 rigid로 두고 남은 관계를 해소한다:

- 원천 row가 명시한 label만 제거하는 차집합의 결과는 원천의 tail을 그대로
  남긴다. 한 row는 같은 label을 한 번만 담으므로 tail에는 제거한 label이 없다.
- 선언된 합집합이 덮는 tail들의 합집합은 그 합집합의 결과이다. 덮는다는 것은
  모든 tail이 선언된 결과 또는 원천이고, 결과가 포함되거나 원천이 모두
  포함된다는 뜻이다.

이렇게 해소된 관계는 본문 지역 row만 정한다. 해소되지 않은 채 시그니처 row를
제약하는 관계는 모든 호출자를 제약하므로 오류이다. 시그니처가 그 제약을 직접
적어야 한다:

```rust
// Error: 본문이 Ping만 처리하므로 `e`가 Ping 외의 effect를 담을 수 없게 된다
fn handled(comp: fn() ->{e} Nil) ->{} Nil {
    handle comp() { do v { v } op Ping::ping() { resume Nil } }
}

// OK: 처리하는 effect를 row에 적는다
fn handled(comp: fn() ->{e, Ping} Nil) ->{e} Nil {
    handle comp() { do v { v } op Ping::ping() { resume Nil } }
}
```

"tail에는 제거한 label이 없다"는 본문 검사 안에서의 사실이다. 호출자는 그
tail을 같은 instance를 담은 row로 채울 수 있다. 대입한 row는
[중복 처리](#기본-규칙)에 따라 그 instance를 한 번만 담는다.

시그니처가 약속하는 것은 tail의 effect가 호출자에게 속한다는 점이다. Handler는
handle 지점 row가 명시한 label만 처리하고, tail로 들어온 operation은 함수 안의
handler를 지나쳐 그 tail에 묶인 호출자의 handler로 간다. 이 선택은 정적이다.
한 callable 안에서 label은 명시 label이거나 tail의 일부이며 둘 다일 수 없으므로,
타입 검사는 각 호출이 callee에게 넘길 handler를 정한다:

```rust
fn twice_counted(f: fn() ->{e} Nil) ->{e} Nat {
    run_state(fn() {
        f()
        State::set(State::get() + 1)
        f()
        State::set(State::get() + 1)
        State::get()
    }, 0)
}

fn main() -> Nil {
    run_state(fn() {
        let calls = twice_counted(fn() { State::set(State::get() + 10) })
        // calls == 2, 바깥 state == 20
    }, 0)
    Nil
}
```

`main`의 lambda가 `e`에 `State(Nat)`를 채워도, `twice_counted` 안의 `f()`는
`run_state`가 설치한 handler가 아니라 `twice_counted`가 받은 evidence의 tail을
받는다. `mask`나 `lacks` 제약은 필요 없다. "`e`에 처리한 label이 없다"는 호출자에
대한 제약이 아니라 dispatch의 성질이다.

#### 호출의 evidence 선택

호출 지점에서 caller row, callee의 선언 row, 그리고 callee row 변수의
인스턴스화가 정적으로 정해진다. Caller row가 명시한 각 ability instance `L`에 대해,
callee가 `L`의 handler를 caller의 명시 `L`에서 받는 자리의 수 `k`를 센다. 자리는
callee 선언 row의 명시 `L`과, callee의 tail 인스턴스가 명시한 `L`이다.

| `k` | 의미 | 예 |
| ---: | --- | --- |
| 0 | caller의 명시 `L`은 callee에게 보이지 않는다. Callee의 `L`은 caller의 tail에서 온다 | `{State \| e}` 안에서 `{e}` 콜백 호출 |
| 1 | 그대로 전달한다 | 대부분의 호출 |
| 2 | Callee의 명시 `L`과 tail의 `L`이 모두 caller의 명시 `L`이다 | `g: fn(fn() ->{e} Nil) ->{State \| e} Nil`을 `e := {State \| e'}`로 호출 |

`k = 2`는 [중복 처리](#기본-규칙)로 대입한 row에서는 보이지 않으므로, 이 선택은
병합한 row가 아니라 인스턴스화에서 계산한다. `k = 0`이 가리기가 되는 것은 callee의
tail 인스턴스가 caller의 tail로 이어질 때뿐이다. Tail이 닫혔거나 풀이 뒤에도 제약이
없는 row 변수로 남으면(제약 없는 변수는 빈 row로 인스턴스화한 것과 같다) callee가
caller tail의 handler에 닿는 자리가 없으므로 그대로 전달한다. Caller row가 명시하지 않은 label과
handler marker가 없는 ambient `Io`는 그대로 전달한다. 이 결과는 typechecking이
각 호출에 확정하는 metadata이며 이후 단계는 signature나 본문 형상에서 다시
계산하지 않는다. 표현과 lowering은
[cps-effects.md](cps-effects.md#row-directed-evidence)를 따른다.

이 규칙은 callee row의 tail이 row 변수 하나일 때를 정의한다. 선언 row가 여러 row
변수의 합집합인 callee(`fn both(f: fn() ->{e1} a, g: fn() ->{e2} b) ->{e1, e2}`)에서는
callee가 각 tail에 넘길 handler를 ability별 marker 순서 하나로 구별할 수 없다. 이런
callee의 tail 인스턴스에 caller의 명시 label이 들어가면 `k`는 그 label이 들어간 tail
수와 무관하게 1로 두고 그대로 전달한다. 각 tail이 같은 instance의 서로 다른
handler를 받아야 하는 경우, 즉 한 tail에는 caller의 명시 label이, 다른 tail에는
caller tail의 같은 instance가 들어가는 경우의 dispatch는 보장하지 않는다.

Handle body는 처리하는 label을 새 handler에 묶고 나머지 label은 바깥 그대로
본다. 바깥 row가 처리하는 label을 명시하면 handle body는 그 바깥 handler를 보지
않는다. Body row는 그 label을 한 번만 명시하므로 body 안에서 바깥 handler에
닿는 자리가 없고, 바깥 handler가 남아 있으면 새 handler와 바깥 tail의 handler
사이에 끼어 tail 자리를 밀어낸다. 그래서 handle 설치도 `k = 0`인 호출처럼 그
label을 먼저 가린다. 바깥 row가 그 label을 tail로만 가지면 가리지 않는다.
Handler arm과 `do` arm은 처리한 label이 빠진 row로 검사되므로, arm의
operation과 호출은 handle 바깥의 handler를 본다. `resume`의 handler 선택은
[abilities.md](abilities.md#resume과-handler-선택)를 따른다.

Row 단일화는 시그니처 row 변수를 별칭의 대표로 유지한다. 열린 두 row를 맞출 때
한쪽이 시그니처 row이고 다른 쪽이 label을 더하지 않으면, 새 변수를 만들지 않고
다른 쪽을 시그니처 row에 묶는다.

Effect 집합 equality는 양방향 후보 검사를 끝낸 뒤에 확정된 타입 치환을 적용한다.
한쪽 순회에서 먼저 찾은 대응의 치환으로 다른 쪽의 모호성을 없애서는 안 되며,
입력 row나 label의 순서를 바꾸어도 같은 제약을 보존해야 한다.

Named row variables have declaration-scoped identity: repeated names share an
identity and distinct names do not. Multiple row names denote their union.

지역 타입 주석도 선언의 row 이름 환경을 사용한다. 같은 함수 선언의 서명과 본문에서
같은 이름은 같은 row를 가리키고, 다른 이름은 독립적으로 유지한다. `{e1, e2}`는
두 row를 하나로 대체하지 않고 합집합 관계로 보존한다. 같은 annotation 노드를
재방문할 때는 처음 변환한 타입을 재사용하며, 생략된 row나 `_`는 해당 주석 위치의
새 변수로 유지한다.

스킴 인스턴스화는 스킴 소유 row를 먼저 freshening한 뒤 호출자의 타입 인자를
대입한다. 대입된 함수 타입 내부의 row는 호출자가 소유하므로, 숫자 식별자가 스킴의
quantifier와 같더라도 다시 freshening하지 않는다. 스킴 본문과 보존한 제약에는
동일한 대응표와 변환 순서를 적용한다.

스킴은 일반 데이터인 builder에서 본문·binder·semantic 제약을 조립한 뒤 한 번에
intern하여 게시한다. 조립 중간 상태는 intern하지 않는다. 게시된 스킴의 타입을
재작성할 때는 본문뿐 아니라 합집합·차집합 제약 안의 타입에도 같은 변환을 적용한다.
이 생성 경계는 binder 정규화나 의미적 동치 판정을 수행하지 않는다.

## Row Unification

### 기본 규칙

두 row를 unify할 때, 공통 label을 맞추고 나머지를 row 변수로 표현한다:

```text
unify({A, B | e₁}, {A, C | e₂})

1. 공통 label A 확인
2. e₁ = {C | e₃} 로 인스턴스화
3. e₂ = {B | e₃} 로 인스턴스화
4. 결과: {A, B, C | e₃}
```

**중복 처리**: 추론 중 동일한 ability instance의 반복은 집합에서 하나로 합친다.
소스 annotation에 같은 instance를 중복 표기한 경우는 별도로 진단한다:

```text
unify({State(Int) | e₁}, {State(Int) | e₂})
// e₁ = e₂ 로 unify됨, 결과: {State(Int) | e₁}  -- OK (중복 아님)

unify({State(Int)}, {State(Int), State(Int)})
// 추론 내부 row는 같은 집합: {State(Int)}

source annotation: {State(Int), State(Int)}
// Error: 동일한 ability State(Int) 중복 표기
```

### 예시

```rust
fn example(f: fn() ->{State(Int)} a, g: fn() ->{Logger} b) {
    f()  // effect: {State(Int)}
    g()  // effect: {Logger}
}
// 추론된 타입: fn(...) ->{State(Int), Logger} b
```

Unification 과정:

```text
{State(Int) | e₁} ∪ {Logger | e₂}
= {State(Int), Logger | e₃}
  where RowUnion([e₁, e₂], e₃)
// 입력 tail 사이에는 equality를 추가하지 않는다.
```

### Occurs Check

Row 변수에 대해서도 occurs check 필요:

```text
unify(e, {State(Int) | e})  -- 에러: e가 자기 자신을 포함
```

---

## Handler와 Effect 소비

### Effect 제거

Handler는 특정 ability를 row에서 "소비"한다:

```rust
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a
```

타입 규칙:

```text
Γ ⊢ comp : fn() ->{e, State(s)} a
Γ ⊢ init : s
────────────────────────────────────────
Γ ⊢ run_state(comp, init) ->{e} a
```

`comp`의 effect `{e, State(s)}`에서:

- `State(s)`는 handler가 처리
- `e`(나머지)만 외부로 전파

### Row에서 Ability 제거

```text
remove(State(s), {State(s), Logger | e}) = {Logger | e}
remove(State(s), {Logger | e}) = 에러: State(s)가 없음
remove(State(s), {e}) = e' where e = {State(s) | e'}
```

마지막 경우는 row 변수 `e`가 `State(s)`를 포함한다고 가정하고, 새 변수 `e'`를 도입한다.

---

## Bidirectional Typing

### Judgment 형태

```text
Γ ⊢ e ⇒ A ; E    -- Infer: 표현식 e의 타입 A와 effect E를 추론
Γ ⊢ e ⇐ A ; E    -- Check: 표현식 e가 타입 A, effect E를 가지는지 검사
```

### 핵심 규칙

#### 변수 (Infer)

```text
x : A ∈ Γ
─────────────────
Γ ⊢ x ⇒ A ; {}
```

#### 람다 (Check)

```text
Γ, x : A ⊢ body ⇐ B ; E
───────────────────────────────
Γ ⊢ fn(x) body ⇐ fn(A) ->{E} B ; {}
```

#### 람다 (Infer)

```text
fresh α, β, e
Γ, x : α ⊢ body ⇐ β ; e
─────────────────────────────────
Γ ⊢ fn(x) body ⇒ fn(α) ->{e} β ; {}
```

람다 본문을 검사할 때의 누적기는 항상 닫힌 빈 row `{}`에서 시작한다. 본문이
실제로 요구한 잔여 row가 이미 열려 있으면 그 row를 그대로 callable type에
기록한다. 본문이 `resume`을 쓰고 잔여 row가 닫혀 있으면, 람다의 row는 그
label들과 continuation row의 합집합이다. Continuation을 재개하면 그 row의
effect를 수행하기 때문이다. 그 밖에 잔여 row가 닫혀 있으면 본문에서 확정한
concrete effect만 보존하고, 람다가 검사된 문맥의 callable signature가 제공한
open tail만 다시 붙인다. 따라서 문맥이 없는 local lambda가
새로운 open tail을 본문 효과의 무조건적인 기본값으로 만들지 않는다. 반환되거나
escaping 값에 저장되거나 open-effect consumer에 전달되어 open callable contract를
받은 람다와, 본문에서 effect를 수행한 람다의 convention은 이 결과에서 그대로
계산한다.

#### 함수 적용 (Infer)

```text
Γ ⊢ f ⇒ fn(A) ->{E} B ; E₁
Γ ⊢ x ⇐ A ; E₂
──────────────────────────────
Γ ⊢ f(x) ⇒ B ; E ∪ E₁ ∪ E₂
```

#### Ability Operation (Infer)

```text
op : fn(A₁, ..., Aₙ) ->{Eff} B ∈ Ability
Γ ⊢ eᵢ ⇐ Aᵢ ; Eᵢ
─────────────────────────────────────
Γ ⊢ Eff::op(e₁, ..., eₙ) ⇒ B ; {Eff} ∪ E₁ ∪ ... ∪ Eₙ
```

#### Handle (Infer)

```text
Γ ⊢ comp ⇒ fn() ->{E, Eff} A ; E₁
Γ ⊢ clauses handle Eff with continuation type
────────────────────────────────────────────
Γ ⊢ handle comp() { clauses } ⇒ B ; E ∪ E₁
```

### Subsumption

Effect row 간의 subsumption:

```text
E₁ ⊆ E₂
Γ ⊢ e ⇐ A ; E₁
───────────────────
Γ ⊢ e ⇐ A ; E₂
```

`{State(Int)} ⊆ {State(Int), Logger}` 이므로, 더 적은 effect를 가진
표현식은 더 많은 effect가 허용되는 컨텍스트에서 사용 가능하다.

---

## Type Inference Algorithm

### 개요

1. **Parse** → Surface AST
2. **Rename** → 고유한 이름 부여
3. **Constraint Generation** → bidirectional traversal로 제약 수집
4. **Constraint Solving** → unification (타입 + row)
5. **Generalization** → let-polymorphism

### Constraint 종류

```text
C ::= τ₁ = τ₂           -- 타입 동치
    | ρ₁ = ρ₂           -- row 동치
    | ρ₁ ⊆ ρ₂           -- row 포함 (subsumption)
    | A ∈ ρ             -- ability 멤버십
    | C₁ ∧ C₂           -- conjunction
```

### Row Unification Algorithm

```text
unify_row(ρ₁, ρ₂):
  match (ρ₁, ρ₂):
    ({}, {}) → success
    
    ({A | ρ₁'}, {A | ρ₂'}) → 
      unify_row(ρ₁', ρ₂')
    
    ({A | ρ₁'}, {B | ρ₂'}) where A ≠ B →
      fresh e
      unify_row(ρ₁', {B | e})
      unify_row(ρ₂', {A | e})
    
    (e, ρ) where e is variable →
      if e ∈ FV(ρ) then error "occurs check"
      else substitute e := ρ
    
    (ρ, e) where e is variable →
      unify_row(e, ρ)
```

### Generalization

Let generalization distinguishes the effect of **evaluating the right-hand
side now** from effects latent in a function value. After solving the
constraints generated up to the binding, `let p = e` generalizes variables in
the type of `e` exactly when evaluating `e` has the closed-empty effect `{}`.
Variables free in the surrounding environment are never generalized.

앞의 lambda inference 규칙으로 얻은 callable effect row는 lambda literal을
평가하는 효과와 구별한다. 따라서 local callable의 generalization은 실제로 추론된
type과 row의 변수만 대상으로 하며, generalization을 위해 새 open tail을 만들지
않는다.

This is an effect-based value restriction, not a restriction to a syntactic
class of values. A lambda evaluates purely, so its type can be generalized even
when calling the resulting function later performs effects:

다음은 toplevel function 본문 안의 local `let` binding이다.

```rust
fn example() ->{State(Int)} Nil {
    let id = fn(x) x
    // id : forall a. fn(a) ->{} a

    let read = fn() State::get()
    // evaluating the lambda is pure; State is a latent call effect
    // read : forall s. fn() ->{State(s)} s

    let current = State::get()
    // evaluating the RHS performs State, so current remains monomorphic
    Nil
}
```

For a destructuring pattern, each introduced name receives the corresponding
component scheme under the same decision. Function parameters, handler
continuations, and pattern bindings not introduced by `let` are monomorphic.

A type scheme quantifies both type variables and effect-row variables:

```text
sigma ::= forall a1 ... an, e1 ... em. tau
```

Only variables not free in the surrounding environment are quantified. Every
lookup instantiates every quantified type variable and row variable freshly;
repeated occurrences of one quantified variable remain shared within that
single instantiation. Variables in a monomorphic scheme are not freshened.

### 지역 quantifier의 소유권

한 `let`이 일반화한 type 변수는 그 `let`의 패턴 전체가 소유한다. 검사된 본문
type과 callable metadata에서 이 변수는 `LocalBoundVar { scope, index }`로
나타난다. `scope`는 `let` 패턴 루트의 NodeId이고, `index`는 그 `let` 안에서
변수마다 한 번 부여하는 번호다.

- destructuring이나 `as` 패턴이 여러 이름에 같은 변수를 나누어 주더라도
  (`let f as g = fn(v) v`) 이름마다 scheme은 따로 가지지만 그 변수의
  `LocalBoundVar`는 하나다. RHS는 한 번만 검사되므로 그 본문 type이 가리키는
  소유자도 하나여야 한다.
- 이름별 scheme의 quantifier 번호(`BoundVar`)는 그 scheme 안에서만 의미가 있고
  `LocalBoundVar`의 번호와 대응하지 않는다.
- 이 소유권은 지역 callable 인스턴스가 사용하는 이름별 binding identity와 별개다.
- 환경에 자유로운 변수는 일반화하지 않으므로, 서로 다른 `let`이 같은 변수를
  소유하는 기록은 컴파일러 내부 오류다.

### 지역 callable 인스턴스의 전달

함수 본문의 타입 주석에서 부모 함수의 타입 매개변수를 참조하면, 선언 signature의
같은 매개변수 인스턴스를 사용한다. 이 변수는 지역 binding의 일반화 대상이 아니며,
부모 함수가 특수화될 때 지역 callable의 매개변수·결과·capture 타입에도 같은
치환을 적용한다. 부모 매개변수를 별도의 nominal 타입 이름으로 해석하지 않는다.

지역 binding의 scheme과 참조별 instantiation은 서로 다른 정보다. 타입 검사는
binding의 NodeId와 LocalId, scheme, 참조별 타입·row 인자와 선택된 callable 타입을
함께 전달한다. 동일 참조의 재검사는 같은 인스턴스를 사용하며, 특수화는 enclosing
함수의 복제와 함께 binding identity 및 이 메타데이터를 갱신한다.

데이터 매개변수·결과·capture 타입이 고정된 source lambda는 검사된 지역 row
인스턴스로 생성할 수 있다. 여기서 고정 타입에는 부모 함수가 소유한 타입
매개변수도 포함된다. 특수화 전 generic 본문에서도 이 변수는 같은 부모 binder를
참조하며, 지역 사용처마다 새로 인스턴스화하지 않는다. 이 과정은 binding이
소유한 row만 치환하고 환경의
자유 변수나 다른 지역 scheme을 닫지 않는다. Lowering은 lambda 본문 형상이나
비어 있는 explicit effect 목록으로 purity를 다시 추론하지 않는다.

지역 callable 생성은 원래 binding 위치의 값을 사용한다. 같은 인스턴스와 호출
계약은 생성물을 공유할 수 있지만, 첫 사용의 타입을 다른 사용 전체에 적용해서는
안 된다. 임의의 callable-producing RHS를 사용처마다 다시 평가하지 않는다.

---

## 예시

### 기본 함수

```rust
fn add(x: Int, y: Int) -> Int {
    x + y
}
// 추론: fn(Int, Int) ->{} Int
```

### Effect 전파

```rust
fn fetch_and_print(url: String) ->{Http, Io} Nil {
    let response = Http::get(url)
    print_line(response.body)
}
// Http::get : fn(String) ->{Http} Response
// print_line : fn(String) ->{Io} Nil
// 합집합: {Http, Io}
```

### Handler

```rust
fn with_state(comp: fn() ->{e, State(Int)} a) ->{e} a {
    run_state(comp, 0)
}

fn example() ->{Io} Int {
    with_state(fn() {
        let n = State::get()
        State::set(n + 1)
        print_line("incremented")
        State::get()
    })
}
// comp의 effect: {Io, State(Int)}
// State(Int) 소비 후: {Io}
```

### 고차 함수

```rust
fn twice(f: fn(a) ->{e} a, x: a) ->{e} a {
    f(f(x))
}

fn use_twice() ->{State(Int)} Int {
    twice(fn(n) { State::set(n); n + 1 }, 0)
}
// f의 effect {State(Int)}가 twice의 결과로 전파
```

### 합성

```rust
fn compose(f: fn(a) ->{e1} b, g: fn(b) ->{e2} c) -> fn(a) ->{e1, e2} c {
    fn(x) g(f(x))
}

let h = compose(
    fn(x) { print_line(x); x },             // ->{Io}
    fn(x) { Http::get(x) }                  // ->{Http}
)
// h : fn(String) ->{Io, Http} Response
```

---

## Design Rules

### Function Type Annotation 규칙

**모듈 수준 함수는 반드시 타입을 명시**해야 한다. 소스 파일 최상위와 inline
`mod` 안의 함수 선언이 모두 해당하며, `main`도 예외가 아니다:

```rust
// OK: 파라미터와 반환 타입 명시
fn add(x: Int, y: Int) -> Int { x + y }

// OK: effect도 명시 가능 (생략 시 암묵적 polymorphic)
fn fetch(url: String) ->{Http} Response { ... }

// Error: 파라미터 타입 누락
fn add(x, y) -> Int { x + y }

// Error: 반환 타입 누락
fn add(x: Int, y: Int) { x + y }

// OK: main도 반환 타입을 명시
fn main() -> Nil { ... }
```

선언 시그니처가 그 함수의 최종 타입 스킴이다. 본문을 검사해서 시그니처를 더
구체적으로 만들거나 효과를 덧붙이지 않는다. 시그니처의 타입 변수와 row 변수는
본문 안에서 rigid하다. 본문이 타입 변수를 구체 타입으로 정하거나, 서로 다른 두
변수를 같게 만들거나, 선언된 row에 없는 concrete effect를 수행하면 오류이다:

```rust
// Error: 본문이 `a`를 Nat으로 정함
fn first(x: a) -> a { 1 }

// Error: 선언된 row에 없는 Ask를 수행
fn ask_twice() ->{State(Int), e} Nat { Ask::ask() + Ask::ask() }
```

**중첩 함수와 람다는 추론 가능**:

```rust
fn example() -> Int {
    let double = fn(x) x * 2  // OK: 람다 타입 추론됨
    let result = [1, 2, 3].map(fn(x) x + 1)  // OK
    double(21)
}
```

**이유**:

1. **문서화**: 모듈의 공개 API는 명시적 타입이 필수
2. **에러 지역화**: 타입 에러가 함수 경계를 넘어 전파되지 않음
3. **증분 컴파일**: 함수 단위로 Salsa 캐싱 가능
4. **별도 컴파일**: 모듈 간 의존성 분석에 시그니처만 필요

**추론 범위**: 타입 추론은 **함수 본문 내부에서만** 동작한다. 각 함수는
독립적으로 타입 체크되며, 함수 간에 타입 변수가 공유되지 않는다. 호출하는 쪽은
피호출 함수의 선언 시그니처만 사용하므로, 검사 순서와 자기 재귀는 결과에 영향을
주지 않는다. 함수 검사는 선언 수집이 만든 모듈 환경을 읽기만 하고, 그 함수의
본문 metadata만 결과로 낸다.

### Effect Annotation 규칙

Effect annotation을 생략하면 fresh한 ability 변수가 생성된다. 본문의 effect로
시그니처를 추론하지 않으므로, 생략한 함수의 본문은 concrete ability를 수행할 수
없다. Ability를 수행하는 함수는 그 ability를 row에 명시한다:

```rust
// 단순 함수: 생략 가능
fn fetch(url: String) -> Response
// 위는 아래와 동일:
fn fetch(url: String) ->{e} Response

// 순수 함수는 명시적으로 {} 표기
fn add(x: Int, y: Int) ->{} Int { x + y }

// 특정 effect
fn fetch_data(url: String) ->{Http} Response { ... }
```

**고차 함수에서 effect 전파**: 내부 함수의 effect를 외부로 전파하려면 **같은 변수를 명시**해야 한다:

```rust
// 생략하면 다른 변수가 됨 (전파 안 됨)
fn map(xs: List(a), f: fn(a) -> b) -> List(b)
// 위는 아래와 동일:
fn map(xs: List(a), f: fn(a) ->{e1} b) ->{e2} List(b)
// e1 ≠ e2 → f의 effect가 map으로 전파되지 않음!

// 올바른 선언: 같은 변수 사용
fn map(xs: List(a), f: fn(a) ->{e} b) ->{e} List(b)

// 여러 함수의 effect 합치기
fn compose(f: fn(a) ->{e1} b, g: fn(b) ->{e2} c) -> fn(a) ->{e1, e2} c
```

**참고**: 순수 함수(`->{} T`)와 polymorphic 함수(`-> T`, 즉 `->{e} T`)는 다르다:

- `->{} Int`: 어떤 effect도 수행하지 않음
- `-> Int`: 암묵적 effect 변수. 본문은 concrete ability를 수행할 수 없지만, 같은
  row 변수를 쓰는 effect-polymorphic 함수를 호출할 수 있다.

---

## 일반화할 제약과 람다 검사 재사용

일반화할 스킴에는 본문 타입의 타입 변수 또는 row 변수와 연결되는 합집합 관계만
옮긴다. 연결은 effect 인자 내부의 타입 변수도 포함하여 전이적으로 계산한다.
무관한 관계는 원래 함수의 solver에서 계속 검사하지만 다른 `let` 스킴에 복사하지
않는다. 예를 들어 앞선 Io 호출의 관계를 뒤에서 바인딩한 `String` 값에 붙이면
그 값을 참조할 때마다 무관한 제약까지 재인스턴스화하게 된다.

함수 본문을 한 번 검사하는 동안, 동일한 람다 `NodeId`를 같은 기대 타입으로
재방문하면 완료된 본문 검사와 typed AST를 재사용한다. 이미 검사한 람다 타입을
그대로 요구하는 변환 방문도 이에 포함된다. 본문을 다시 검사하여 지역 `let`의
일반화나 fresh row 제약을 중첩 깊이에 따라 반복 생성하지 않는다.

기대 타입이 달라진 방문은 새 기대 타입을 검사해야 한다. 이 재사용은 함수 검사
컨텍스트에 한정되며, 서로 다른 람다 노드나 일반화한 함수 값의 서로 다른 사용
위치를 합치지 않는다. 람다 생성 자체는 본문의 latent effect를 수행하지 않으므로
재사용 때 본문 effect를 바깥 누적 row에 추가하지 않는다.

## Open Questions

1. **에러 메시지**: Row unification 실패 시 사용자 친화적인 메시지 생성

2. **IDE 지원**: 추론된 effect를 어떻게 표시할지

3. **Effect aliases** (향후 고려):

   ```rust
   type Network = {Http, Async}
   fn fetch_all() ->{Network} List(Response)
   ```

---

## References

- [Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism](https://www.cl.cam.ac.uk/~nk480/bidir.pdf)
- [Koka: Programming with Row-polymorphic Effect Types](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/koka-effects-2013.pdf)
- [Do Be Do Be Do](https://arxiv.org/abs/1611.09259) (Frank)
- [Extensible Records with Scoped Labels](https://www.microsoft.com/en-us/research/publication/extensible-records-with-scoped-labels/)
