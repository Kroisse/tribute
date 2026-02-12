# Reference Counting (Native Backend)

> This document defines the RC memory management strategy for the Cranelift
> native backend. The WASM backend uses WasmGC and is not affected.
>
> See also: [cranelift-backend.md](cranelift-backend.md),
> [implementation.md](implementation.md)

## Overview

The native backend uses **reference counting** for heap-allocated objects
(structs, enums, arrays, boxed primitives). Key principles:

- **No runtime library** — all RC logic is compiler-generated code
- **libc only** — depends solely on `malloc`/`free` (via allocator indirection)
- **Dialect-based** — RC operations are `tribute_rt.retain`/`tribute_rt.release`
  dialect ops, lowered to inline code
- **Phased rollout** — Phase 3a (shallow), 3b (deep release), 4 (continuations)

## Allocator Interface

All heap allocation goes through two indirection symbols:

```text
__tribute_alloc(size: i64) -> ptr
__tribute_dealloc(ptr: ptr, size: i64)
```

### Default Implementation

The compiler generates default implementations as simple `malloc`/`free`
wrappers. These are declared with `Import` linkage so they can be overridden
at link time (e.g., with a custom allocator via weak symbols).

### Alloc Sequence (compiler-generated inline)

```text
raw_ptr = call @__tribute_alloc(size + 8)    // include header
store refcount=1       at raw_ptr
store rtti_idx=<type>  at raw_ptr + 4
obj_ptr = raw_ptr + 8                        // caller sees offset 0
```

### Free Sequence (compiler-generated inline)

```text
raw_ptr = obj_ptr - 8
call @__tribute_dealloc(raw_ptr, size + 8)
```

### Symbol Convention

Internal/runtime symbols use the `__tribute_` prefix to avoid collisions
with user code and to clearly mark compiler-generated functions.

---

*Sections on Memory Layout, RC Operations, Boxing/Unboxing, RC Pipeline,
and Phasing will be added in subsequent PRs.*
