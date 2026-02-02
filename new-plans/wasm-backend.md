# Wasm Backend Architecture

> 이 문서는 Tribute → WebAssembly 컴파일 백엔드의 아키텍처를 정의한다.

## Overview

Tribute는 WasmGC (Wasm 3.0)를 주요 타겟으로 한다. 백엔드는 다음 원칙을 따른다:

1. **타겟 독립적 IR 유지**: trunk-ir는 특정 타겟에 종속되지 않음
2. **Backend-specific 타입 처리**: WasmGC 타입 정의는 백엔드에서 처리
3. **관심사 분리**: lowering (tribute-passes)과 emission (trunk-ir-wasm-backend) 분리

---

## 크레이트 구조

```text
trunk-ir/
├── dialect/
│   ├── wasm.rs           # wasm ops (struct_new, array_new, call, ...)
│   └── ...               # target-independent dialects only

trunk-ir-wasm-backend/    # trunk-ir만 의존
├── emit.rs               # IR → WebAssembly binary
├── type_section.rs       # WasmGC type section 생성 (rec group, subtype 처리)
├── type_collector.rs     # wasm.* ops에서 타입 정보 수집
├── func_to_wasm.rs       # func.* → wasm instructions
├── arith_to_wasm.rs      # arith.* → wasm instructions
├── scf_to_wasm.rs        # scf.* → wasm instructions
└── ...

tribute-passes/           # tribute-ir 의존
├── adt_to_wasmgc.rs      # adt.* → wasm.* (struct_new, array_new 등)
├── closure_to_wasmgc.rs  # closure.* → wasm.*
└── ...

tribute/                  # main crate - 파이프라인 조율
└── pipeline.rs
```

---

## Lowering 경로

### WasmGC 타겟 (주요 경로)

```text
tribute-ir (High-level)
├── adt.struct_new
├── adt.variant_new
├── closure.new
│
▼ tribute-passes/adt_to_wasmgc.rs
│
trunk-ir (Mid-level)
├── wasm.struct_new       # 인스턴스 생성
├── wasm.struct_get/set   # 필드 접근
├── wasm.array_new        # 배열 생성
│
▼ trunk-ir-wasm-backend
│   (type_collector: wasm.* ops에서 타입 수집)
│   (type_section: rec group 분석, type section 생성)
│
WebAssembly Binary
```

### Linear Memory 타겟 (미래)

```text
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

## WasmGC 타입 처리

### Backend에서 타입 수집

trunk-ir-wasm-backend는 `wasm.*` 연산들에서 타입 정보를 수집한다:

```rust
// wasm.struct_new 연산에서 타입 정보 추출
// @Point 타입과 필드 타입들을 수집
%p = wasm.struct_new @Point (%x: f64, %y: f64) : ref<@Point>

// wasm.array_new에서 배열 타입 정보 추출
%arr = wasm.array_new @IntArray (%len) : ref<@IntArray>
```

### Type Section 생성

수집된 타입 정보로 WasmGC type section 생성:

1. **타입 의존성 분석**: 타입 간 참조 관계 파악
2. **SCC 분석**: 상호 재귀 타입 탐지 → rec group 생성
3. **Type section emit**: struct/array type 정의 출력

```wasm
;; 생성된 type section 예시
(rec
  (type $Node (struct (field i32) (field (ref null $Node)))))
(type $Point (struct (field f64) (field f64)))
```

---

## 설계 결정 배경

### GC 관련 타입을 trunk-ir에 추가하지 않는 이유

Cranelift 팀의 교훈 참고 ([Stack Maps 문서](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)):

> IR 코어에 GC 참조 타입을 넣으면 복잡해진다. Frontend가 처리하는 게 낫다.

Cranelift는 초기에 GC 참조를 IR 전체에서 추적했으나, 다음 문제 발생:

- 전용 참조 타입이 최적화 방해
- Mid-end에서 safepoint spill/reload가 보이지 않아 버그 발생
- 복잡성 증가

해결책: "User Stack Maps" - frontend가 GC 관련 처리를 담당

**Tribute에서의 적용:**

- trunk-ir에 GC 관련 dialect 추가하지 않음 (gc, gc_type 등)
- WasmGC-specific 개념 (rec_group, subtype)은 백엔드에서 처리
- trunk-ir는 target-independent하게 유지

### wasm dialect의 역할

wasm dialect는 WasmGC 인스턴스 연산만 포함:

- `wasm.struct_new`, `wasm.struct_get`, `wasm.struct_set`
- `wasm.array_new`, `wasm.array_get`, `wasm.array_set`
- 기타 Wasm 명령어들

타입 정의 (type section)는 백엔드가 이 연산들에서 추론하여 생성한다.

### tribute-wasm-backend 제거

역할 분담 후 불필요:

- Lowering → tribute-passes
- Emission → trunk-ir-wasm-backend
- 조율 → tribute main crate

---

## 구현 단계 (제안)

### Phase 1: trunk-ir-wasm-backend 생성

1. 새 크레이트 생성 (trunk-ir만 의존)
2. type_collector.rs: wasm.\* 연산에서 타입 정보 수집
3. type_section.rs: rec group 분석 및 type section 생성
4. emit.rs: WebAssembly binary 출력

### Phase 2: Lowering passes 이동

1. tribute-passes에 adt_to_wasmgc.rs 추가
2. adt.\* → wasm.\* 변환 구현
3. closure_to_wasmgc.rs 추가
4. 기존 tribute-wasm-backend에서 해당 코드 제거

### Phase 3: tribute-wasm-backend 제거

1. 남은 기능을 tribute main으로 이동
2. 크레이트 삭제
3. 문서 업데이트

---

## References

- [Wasm 3.0 Release](https://webassembly.org/news/2025-09-17-wasm-3.0/)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md)
- [Cranelift Stack Maps](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
