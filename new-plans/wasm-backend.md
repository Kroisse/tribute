# Wasm Backend Architecture

> 이 문서는 Tribute → WebAssembly 컴파일 백엔드의 아키텍처를 정의한다.

## Overview

Tribute의 Wasm backend는 WasmGC (Wasm 3.0) 표현을 emit한다. 현재 주요 구현
경로는 native이지만, Wasm backend도 같은 shared middle-end와 effect ABI를
입력으로 받는다. 백엔드는 다음 원칙을 따른다:

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
├── translate.rs          # IR → WebAssembly binary entrypoint
├── emit.rs               # instruction/function/module emission
├── gc_types.rs           # builtin WasmGC type index layout
├── emit/gc_types_collection.rs
│                         # wasm.* ops에서 타입 정보 수집
├── passes/func_to_wasm.rs
├── passes/arith_to_wasm.rs
├── passes/scf_to_wasm.rs
├── passes/adt_to_wasm.rs
└── ...

tribute-passes/           # tribute-ir 의존
├── wasm/lower.rs         # Wasm lowering pipeline orchestration
├── wasm/evidence_to_wasm.rs
│                         # effect.* → evidence helpers + call_indirect
├── wasm/tribute_rt_to_wasm.rs
├── wasm/const_to_wasm.rs
├── wasm/intrinsic_to_wasm.rs
├── wasm/normalize_primitive_types.rs
└── ...

tribute/                  # main crate - 파이프라인 조율
└── pipeline.rs
```

Native (Cranelift) 백엔드는
[cranelift-backend.md](cranelift-backend.md)를 참조.

---

## Lowering 경로

### WasmGC 타겟

```text
tribute-ir (High-level)
├── adt.struct_new
├── adt.variant_new
├── closure.new
├── effect.extend
├── effect.dispatch_tail / effect.dispatch_cps
│
▼ tribute-passes/wasm/lower.rs
│
trunk-ir (Mid-level)
├── wasm.struct_new       # 인스턴스 생성
├── wasm.struct_get/set   # 필드 접근
├── wasm.array_new        # 배열 생성
├── wasm.call_indirect    # dispatch closure 호출
├── wasm.func             # evidence lookup/extend helpers
│
▼ trunk-ir-wasm-backend
│   (gc_types_collection: wasm.* ops에서 타입 수집)
│   (builtin GC type layout + user type collection)
│
WebAssembly Binary
```

Effect lowering is target-specific. Shared ability lowering produces
`effect.*` operations and must not inspect Marker field numbers or closure
layout. `wasm/evidence_to_wasm` is the Wasm boundary that removes those
operations by generating evidence lookup/extend helpers, unpacking closure
structs `(table_idx, env)`, and emitting `wasm.call_indirect`.

---

## WasmGC 타입 처리

### Backend에서 타입 수집

`trunk-ir-wasm-backend`는 builtin GC type layout을 먼저 예약하고, `wasm.*`
연산들에서 user type 정보를 수집한다:

```rust
// wasm.struct_new 연산에서 타입 정보 추출
// @Point 타입과 필드 타입들을 수집
%p = wasm.struct_new @Point (%x: f64, %y: f64) : ref<@Point>

// wasm.array_new에서 배열 타입 정보 추출
%arr = wasm.array_new @IntArray (%len) : ref<@IntArray>
```

### Type Section 생성

수집된 타입 정보로 WasmGC type section을 생성한다. Builtin type index layout은
현재 고정되어 있고, user-defined type은 그 뒤에 배치된다:

| Index | Type |
| ---: | --- |
| 0 | `BoxedF64` |
| 1 | `BytesArray` |
| 2 | `BytesStruct` |
| 3 | `Step` legacy trampoline struct |
| 4 | `_closure { table_idx: i32, env: anyref }` |
| 5 | `_Marker { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref }` |
| 6 | `Evidence` array |
| 7 | `Continuation` legacy trampoline struct |
| 8 | `ResumeWrapper` legacy trampoline struct |
| 9+ | user-defined structs, arrays, variants, closures |

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
- WasmGC-specific 개념 (type indices, builtin type layout, ref/nullability)은
  백엔드에서 처리
- trunk-ir는 target-independent하게 유지

### wasm dialect의 역할

wasm dialect는 WasmGC 인스턴스 연산만 포함:

- `wasm.struct_new`, `wasm.struct_get`, `wasm.struct_set`
- `wasm.array_new`, `wasm.array_get`, `wasm.array_set`
- 기타 Wasm 명령어들

타입 정의 (type section)는 백엔드가 이 연산들에서 추론하여 생성한다.

### tribute-wasm-backend 제거

별도 `tribute-wasm-backend` 크레이트는 현재 사용하지 않는다. 역할 분담은 다음과
같다:

- Lowering → tribute-passes
- Emission → trunk-ir-wasm-backend
- 조율 → tribute main crate

---

## Current Work Items

- Remove or repurpose the legacy `Step`, `Continuation`, and `ResumeWrapper`
  builtin types once the old trampoline path is fully retired.
- Add a backend-ready conversion target for Wasm that rejects residual
  `effect.*` and non-wasm dialect operations at the emission boundary.
- Keep pass-level textual IR fixtures for `effect.extend`,
  `effect.dispatch_tail`, and `effect.dispatch_cps` so the Wasm and native
  lowering paths cannot drift silently.

---

## References

- [Wasm 3.0 Release](https://webassembly.org/news/2025-09-17-wasm-3.0/)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md)
- [Cranelift Stack Maps](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
