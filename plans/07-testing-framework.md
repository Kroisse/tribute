# Testing Framework Proposal

Source-level test runner는 추가 tooling 설계 대상이다. Repository 자체의 Rust 테스트
실행법은 [`README.md`](../README.md#development)와 `tribute-testing` skill을 따른다.

## Test Failure

Test assertion 실패는 typed `Throw`로 표현할 수 있다.

```rust
use abilities::Throw

fn assert(condition: Bool) ->{Throw(String)} Nil {
    if condition { Nil } else { Throw::throw("assertion failed") }
}
```

Runner는 test body를 handle하여 완료와 실패를 수집한다. Test declaration은 annotation,
명시적인 등록 또는 file/module convention 중에서 별도로 정해야 한다. Annotation이
source transformation을 수행하는지는 test runner와 독립적인 언어 설계 결정이다.

## Effect Mocking

테스트할 외부 동작을 일반 user ability로 선언하면 handler로 대체할 수 있다.
Compiler-owned ambient `Io`는 handler로 제거할 수 없으므로 mocking 경계로 사용하지
않는다.

```rust
ability Fetch {
    fn get(url: String) -> String
}

fn fetch_user() ->{Fetch} String {
    Fetch::get("/users/123")
}

fn mocked_user() -> String {
    handle fetch_user() {
        do result { result }
        fn Fetch::get(url) { "mock response" }
    }
}
```

명시적인 중단과 재개를 검사할 때는 `op`과 affine `resume`을 사용한다. Test runner는
생성한 continuation을 여러 번 실행해 multi-shot 의미를 만들지 않는다.

## Property Testing

입력 탐색은 독립적인 test invocation과 명시적인 random generator state로 구현한다.
Shrinking도 새 test invocation을 실행하며 이미 소비한 continuation을 재개하지 않는다.

## 미결정 사항

- Test declaration과 runner CLI
- Package/build system 통합과 test discovery
- Random generation·shrinking API
- Coverage instrumentation과 보고 형식

## 검증 조건

- 성공, typed failure와 abort를 명확하게 구별한다.
- Native/Wasm 실행 지원을 개별적으로 검증한다.
- Handler mocking이 source의 operation kind와 effect row를 보존한다.
- 실패 위치와 원인을 structured diagnostic으로 보고한다.
