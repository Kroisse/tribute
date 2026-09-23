# Standard Library Extensions

표준 library는 source의 ability와 immutable value 계약을 따른다. 현재 public API와
추가 기능 제안을 구분하며, 실제 지원 범위는
[`capabilities.md`](../new-plans/capabilities.md)를 따른다.

## Canonical Surface

- 수치 타입과 변환은 [`numeric-types.md`](../new-plans/numeric-types.md)를 따른다.
- `String`과 `Bytes`의 표현과 public API는
  [`string.md`](../new-plans/string.md)를 따른다.
- Compiler-owned opaque `List(a)`는 literal, sequence-view pattern과
  `std::collections::List::prepend`를 제공한다. RRB node constructor는 source에
  노출하지 않는다. Public namespace 계약은
  [`modules.md`](../new-plans/modules.md)를 따른다.
- Prelude는 `Option(a)`와 `Result(a, e)`, non-resumptive
  `abilities::Abort`와 `abilities::Throw(e)`를 제공한다.

## I/O와 오류 처리

`std::io::Io`는 compiler-owned ambient ability다. 사용자 handler로 구현하거나
제거하지 않는다. Public `print_line`과 `read_line`은 일반 source 함수이며,
`read_line` 실패는 `Throw(std::io::Error)`로 처리한다. 전체 API, effect와 target
계약은 [`io.md`](../new-plans/io.md)를 따른다.

```rust
use abilities::Throw
use std::io::{Io, print_line, read_line}

fn greet() ->{Io} Nil {
    print_line("What is your name?")
    handle read_line() {
        do name { print_line("Hello, " <> name) }
        op Throw::throw(error) { print_line("Input failed") }
    }
}
```

`Abort`와 `Throw` operation의 결과는 `Never`다. Handler는 대체 결과를 반환하며
resume capability를 받지 않는다. 일반 resumptive operation은 `op`으로 선언하고
handler에서 `resume value`를 최대 한 번 사용한다.

## 추가 Collection API

`List::map`, `filter`, `fold`, concat, slice, index와 추가 collection 타입은 별도
public API 계약과 검증을 갖추어 도입한다. Source API는 node layout 대신 sequence
의미를 표현하고, native와 Wasm은 같은 관찰 가능한 결과를 보장해야 한다.

Higher-order operation은 callback의 effect row를 전파한다. Source에 raw pointer나
mutable backend storage를 노출하여 effect 또는 ownership 계약을 우회하지 않는다.

## Stream과 Async

Stream과 Async는 추가 library 설계 대상이다. Generator는 명시적인 affine
`resume`과 immutable accumulator를 사용하며 multi-shot continuation을 요구하지
않아야 한다. Promise, scheduling, cancellation의 ownership와 effect row는 public
API를 확정할 때 함께 정의한다.

Console 외의 file·network API도 concrete operation별 오류 타입, `Io` effect와
native/Wasm 지원 경계를 명시해야 한다. 별도의 resumptive user ability를 통한
의존성 주입은 가능하지만 ambient `Io` 자체를 mock handler로 제거하지 않는다.

## 검증 조건

- Public API는 source syntax, type/effect checking과 backend 실행 검증을 갖춘다.
- Collection은 immutable value 의미와 persistent sharing의 ownership을 보존한다.
- 오류·abort·resume 경로를 포함한 handler 조합을 검증한다.
- API 예제는 공개 namespace만 사용하며 compile-only와 runtime 증거를 구별한다.
