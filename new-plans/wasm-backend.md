# Wasm Backend Architecture

> 이 문서는 Tribute → WebAssembly 컴파일 백엔드의 아키텍처를 정의한다.

## Overview

Tribute는 WasmGC (Wasm 3.0)를 주요 타겟으로 한다. 백엔드는 다음 원칙을 따른다:

1. **타겟 독립적 IR 유지**: trunk-ir는 특정 타겟에 종속되지 않음
2. **명시적 타입 정의**: 타입 정보는 attribute가 아닌 연산으로 표현
3. **관심사 분리**: lowering (tribute-passes)과 emission (trunk-ir-wasm-backend) 분리

---

## 크레이트 구조

```
trunk-ir/
├── dialect/
│   ├── wasm.rs           # wasm ops (struct_new, array_new, call, ...)
│   ├── gc_type.rs        # GC 타입 정의 연산
│   └── ...

trunk-ir-wasm-backend/    # trunk-ir만 의존
├── emit.rs               # wasm.* + gc_type.* → binary
├── type_section.rs       # gc_type.* → wasm type section
├── func_to_wasm.rs       # func.* → wasm.*
├── arith_to_wasm.rs      # arith.* → wasm.*
├── scf_to_wasm.rs        # scf.* → wasm.*
└── ...

tribute-passes/           # tribute-ir 의존
├── adt_to_wasmgc.rs      # adt.* → gc_type.* + wasm.*
├── closure_to_wasmgc.rs  # closure.* → gc_type.* + wasm.*
└── ...

tribute/                  # main crate - 파이프라인 조율
└── pipeline.rs
```

---

## gc_type Dialect

WasmGC 타입 정의를 위한 dialect. trunk-ir에 위치한다.

### 연산 정의

```rust
dialect! {
    mod gc_type {
        /// Struct 타입 정의
        #[attr(name: Symbol, fields: Vec<(Symbol, Type, bool)>)]
        fn struct_def() -> type_ref;

        /// Array 타입 정의
        #[attr(name: Symbol, element: Type, mutable: bool)]
        fn array_def() -> type_ref;

        /// 재귀 타입 그룹 (WasmGC rec group)
        fn rec_group() { #[region(types)] {} };

        /// Subtype 관계 선언
        #[attr(sub: TypeRef, super: TypeRef)]
        fn subtype();
    }
}
```

### IR 예시

```
// 타입 정의 (모듈 최상단)
gc_type.struct_def @Point { x: f64, y: f64 }
gc_type.struct_def @Node { value: i32, next: ref<@Node>? }

gc_type.rec_group {
    gc_type.struct_def @Tree { left: ref<@Tree>?, right: ref<@Tree>? }
}

// 사용
func.func @make_point(%x: f64, %y: f64) -> ref<@Point> {
    %p = wasm.struct_new @Point (%x, %y)
    func.return %p
}
```

### WasmGC 대응

| gc_type 연산 | WasmGC |
|-------------|--------|
| `struct_def` | type section의 struct type |
| `array_def` | type section의 array type |
| `rec_group` | recursive type group |
| `subtype` | subtype declaration |

---

## Lowering 경로

### WasmGC 타겟 (주요 경로)

```
tribute-ir (High-level)
├── adt.struct_new
├── adt.variant_new
├── closure.new
│
▼ tribute-passes/adt_to_wasmgc.rs
│
trunk-ir (Mid-level)
├── gc_type.struct_def    # 타입 정의 생성
├── wasm.struct_new       # 인스턴스 생성
├── wasm.struct_get/set
│
▼ trunk-ir-wasm-backend
│
WebAssembly Binary
```

### Linear Memory 타겟 (미래)

```
tribute-ir (High-level)
├── adt.struct_new
│
▼ tribute-passes/adt_to_linear.rs
│
trunk-ir (Mid-level)
├── mem.alloc
├── mem.store (field offset 계산)
│
▼ trunk-ir-wasm-backend (또는 trunk-ir-clif-backend)
│
WebAssembly Binary (linear memory) 또는 Native Binary
```

---

## 설계 결정 배경

### gc dialect를 추가하지 않는 이유

Cranelift 팀의 교훈 참고 ([Stack Maps 문서](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)):

> IR 코어에 GC 참조 타입을 넣으면 복잡해진다. Frontend가 처리하는 게 낫다.

Cranelift는 초기에 GC 참조를 IR 전체에서 추적했으나, 다음 문제 발생:
- 전용 참조 타입이 최적화 방해
- Mid-end에서 safepoint spill/reload가 보이지 않아 버그 발생
- 복잡성 증가

해결책: "User Stack Maps" - frontend가 GC 관련 처리를 담당

Tribute에서의 적용:
- trunk-ir에 범용 `gc` dialect 추가하지 않음
- WasmGC-specific ops는 `wasm` dialect에 유지
- 타입 정의만 `gc_type` dialect로 분리 (이는 GC 추적이 아닌 타입 선언)

### 타입 정의를 연산으로 표현하는 이유

Attribute 방식의 문제:
- 타입 정보가 분산됨
- 중복 정의 가능
- Emit 시 수집 로직 필요

연산 방식의 장점:
- 타입이 IR에 명시적으로 존재
- WasmGC type section과 직접 대응
- Subtyping, 재귀 타입 자연스럽게 표현
- 모듈이 self-contained

### tribute-wasm-backend 제거

역할 분담 후 불필요:
- Lowering → tribute-passes
- Emission → trunk-ir-wasm-backend
- 조율 → tribute main crate

---

## 구현 단계 (제안)

### Phase 1: gc_type dialect 추가
1. `trunk-ir/src/dialect/gc_type.rs` 생성
2. struct_def, array_def, rec_group, subtype 연산 정의
3. 기본 테스트

### Phase 2: trunk-ir-wasm-backend 분리
1. 새 크레이트 생성
2. func/arith/scf → wasm 변환 이동
3. gc_type → type section 변환 구현
4. emit.rs 이동 및 리팩토링

### Phase 3: adt/closure lowering 이동
1. tribute-passes에 adt_to_wasmgc.rs 추가
2. adt.* → gc_type.* + wasm.* 변환 구현
3. closure_to_wasmgc.rs 추가
4. 기존 tribute-wasm-backend에서 해당 코드 제거

### Phase 4: tribute-wasm-backend 제거
1. 남은 기능을 tribute main으로 이동
2. 크레이트 삭제
3. 문서 업데이트

---

## References

- [Wasm 3.0 Release](https://webassembly.org/news/2025-09-17-wasm-3.0/)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md)
- [Cranelift Stack Maps](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
