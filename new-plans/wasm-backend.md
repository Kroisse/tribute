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
│                         # effect.* → evidence helpers + call/return_call
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
├── func.tail_call / func.tail_call_indirect
│
▼ tribute-passes/wasm/lower.rs
│
trunk-ir (Mid-level)
├── wasm.struct_new       # 인스턴스 생성
├── wasm.struct_get/set   # 필드 접근
├── wasm.array_new        # 배열 생성
├── wasm.call_indirect    # ordinary indirect 호출
├── wasm.return_call      # direct proper tail transfer
├── wasm.return_call_indirect
│                         # indirect proper tail transfer
├── wasm.func             # evidence lookup/extend helpers
│
▼ trunk-ir-wasm-backend
│   (gc_types_collection: wasm.* ops에서 타입 수집)
│   (builtin GC type layout + user type collection)
│
WebAssembly Binary
```

Effect lowering은 target별로 수행한다. Shared ability lowering은 `effect.*`를
만들며 Marker field 번호나 closure layout을 검사하지 않는다.
`wasm/evidence_to_wasm`은 evidence lookup/extend helper를 만들고 closure struct
`(table_idx, env)`를 풀어 semantic role에 맞는 일반 호출 또는 proper-tail call을
emit하여 해당 operation을 제거하는 Wasm 경계다.

### Entrypoint contract

The frontend accepts `main` only when its declared result is `Nil`. A frontend
error is terminal, so the Wasm backend never receives a valid program whose
`main` returns an `Int`, `Nat`, or another user value. The generated `_start`
function therefore calls `main` for its side effects. A pure `main` is called
directly; a `main ->{Io} Nil` receives target-provided initial evidence through
the `EvidenceDirect` ABI. Other residual effects are rejected by the frontend.
Printing program results belongs in explicit standard I/O calls such as
`std::io::print_line`, which shared lowering maps to the target-independent I/O
boundary described in [io.md](io.md), not in backend entrypoint lowering.

CPS root가 필요한 export는
[직접형 wrapper와 completion cell](cps-effects.md#root-main-delimiter)을 사용한다.
Shared IR은 completion cell과 `core.never` root `done_k`의 추상 조합 계약만
보존한다. #825가 CPS signature를 Wasm empty-result signature로 내린 뒤 현재
nil/void machinery에 맞는 wrapper와 ordinary call을 합성한다. Wrapper는 root
`done_k`가 typed cell을 쓴 뒤 call이 돌아오면 이를 읽는다. Shared `func.call`에
zero-result 형상을 추가하지 않으며, 이 bridge는 trampoline이나 `anyref` control
carrier가 아니다.

### 올바른 꼬리 호출 계약

[WebAssembly 3.0 validation](https://webassembly.github.io/spec/core/valid/instructions.html#valid-return-call-indirect)은
`return_call`, `return_call_indirect`, `return_call_ref`의 tail-call 형식과 caller
result matching을 정의한다. Tribute는 다음 경로를 구현한다:

| Shared IR | Wasm IR | Encoder |
| ---- | ---- | ---- |
| `func.tail_call` | `wasm.return_call` | `wasm_encoder::Instruction::ReturnCall` |
| `func.tail_call_indirect` | `wasm.return_call_indirect` | `wasm_encoder::Instruction::ReturnCallIndirect { type_index, table_index }` |

`func.tail_call_indirect` lowering은 callee table index와 argument를 기존
`call_indirect`와 같은 순서로 평가하고, callee `core.func`에서 `type_index`를
결정한다. CPS caller/callee의 result vector는 모두 비어 있어야 하며
`wasm.return_call_indirect`는 result local을 만들지 않는다. 일반 source-data
indirect call만 `wasm.call_indirect`를 유지한다.

### Dynamic basic output

WASI preview1 `fd_write` cannot read a WasmGC `Bytes` array directly because its
iovec points into linear memory. The initial `tribute_io.write` lowering copies
the dynamic `Bytes` slice into an instance-local linear scratch buffer, appends
the optional newline there, and invokes `fd_write` with compiler-owned iovec and
`nwritten` cells. The lowering grows memory when required and retries partial or
interrupted writes. See [io.md](io.md#wasm-runtime-boundary) for lifetime and
failure rules.

Wasm output uses only `tribute_io.write`. The former `__print_line` literal
analysis and its `i32` pointer plus `literal_len` representation are not part of
the backend boundary; string literals remain canonical `String` values until
the standard-library I/O wrapper explicitly converts them to `Bytes`.

### Private List layout

WasmGC must lower the same representation-independent `list.*` sequence
operations to a target-private GC layout and eliminate them before the
backend-ready boundary. It is not required to share native's linked-node/null
layout. The M1 native implementation and compile-only shared frontend evidence
do not by themselves establish Wasm compilation or execution support; capability
claims require focused Wasm evidence.

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

수집된 타입 정보로 WasmGC type section을 생성한다. 다음 표는 migration 전 현재
고정 layout이며 user-defined type은 그 뒤에 배치된다:

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

Issue #826 완료 시 `Step`, `Continuation`, `ResumeWrapper`를 삭제하고 builtin/user index를
다시 계산한다. 최종 layout은 이 세 index를 예약하거나 호환 placeholder를
남겨서는 안 된다. `_closure` environment와 Marker의 dispatch closure field는
일반 reference erasure이므로 계속 `anyref`를 사용할 수 있다.

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

## 선택한 마이그레이션 작업

- #823은 generic `func.tail_call_indirect`와 empty-result verifier contract를
  제공한다.
- #825는 `wasm.return_call_indirect` operation, func-to-Wasm rewrite와
  `Instruction::ReturnCallIndirect` emission을 구현한다.
- #826은 `Step`, `Continuation`, `ResumeWrapper` 및 compatibility control carrier
  관련 type/emission path를 삭제한다.
- 최종 Wasm emission boundary는 residual `tribute_control.*`, `ability.*`,
  `effect.*`, compatibility carrier와 result-producing CPS transfer를 거부한다.
- `effect.extend`, `effect.dispatch_tail`, `effect.dispatch_cps`와 direct/indirect
  tail-call fixture를 함께 유지해 native/Wasm 경로의 drift를 막는다.

---

## References

- [Wasm 3.0 Release](https://webassembly.org/news/2025-09-17-wasm-3.0/)
- [WebAssembly 3.0 tail-call validation](https://webassembly.github.io/spec/core/valid/instructions.html#valid-return-call-indirect)
- [WasmGC Proposal](https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md)
- [Cranelift Stack Maps](https://bytecodealliance.org/articles/new-stack-maps-for-wasmtime)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
