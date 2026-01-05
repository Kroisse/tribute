# Linking Strategy

## Overview

Tribute targets WasmGC, which significantly constrains linking options compared to traditional linear memory WebAssembly.

## Options Considered

### 1. Single Module Compilation (Current)

```
user.trb + std/*.trb → single .wasm
```

**Pros:**
- Simplest implementation
- No linker needed
- GC type sharing is trivial (same module)

**Cons:**
- No incremental compilation
- Full recompile on any change
- No separate library distribution

### 2. Import/Export Based

```wasm
;; std.wasm
(func (export "Int::add") ...)

;; user.wasm
(import "std" "Int::add" (func ...))
```

**Pros:**
- Standard Wasm mechanism
- Separate compilation possible

**Cons:**
- GC type sharing across modules is problematic
- How does runtime know `BigNat` in std.wasm == `BigNat` in user.wasm?
- Type canonicalization at instantiation time is complex

### 3. Static Linking (wasm-ld style)

```
foo.trb → foo.o (relocatable)
bar.trb → bar.o
        ↓ wasm-ld
    output.wasm
```

**Pros:**
- Incremental compilation (only recompile changed files)
- Link-time optimization (dead code elimination, cross-module inlining)
- Standard tooling (LLVM ecosystem)
- Static library distribution (.a files)

**Cons:**
- **WasmGC is not supported** - wasm-ld designed for linear memory model
- No standard for WasmGC object file format
- Type definition merging undefined (recursive type groups, type hierarchies)
- Additional toolchain dependency (LLVM/wasm-ld)

### 4. Component Model

**Pros:**
- Modern approach with WIT interface definitions
- Clean component composition
- wasmtime support available

**Cons:**
- Browser support not yet available
- WasmGC + Component Model combination is unexplored territory
- Tooling still maturing

## The WasmGC Problem

wasm-ld handles relocations for linear memory Wasm:
- Data section relocations
- Function index relocations
- Table index relocations

WasmGC introduces new challenges:
- **Type definitions** - struct/array types need merging
- **Type indices** - GC ref types reference type indices
- **Recursive type groups** - `(rec ...)` must be handled atomically
- **Type hierarchies** - subtyping relationships must be preserved

There is no standard for WasmGC relocatable object files.

## Current State of WasmGC Languages

| Language | Linking Approach |
|----------|-----------------|
| Rust → WasmGC | Single module only |
| OCaml → WasmGC | Single module only |
| Kotlin/Wasm | Single module only |
| Dart → WasmGC | Single module only |

All major WasmGC implementations use single-module compilation.

## Decision

**Short term:** Source-level merging → single module compilation
- Compiler finds `use std::Int` and compiles referenced sources together
- Similar to Go or Zig compilation model
- No separate linking phase

**Long term:** Revisit when Component Model matures with WasmGC support

## Implementation Notes

For source-level merging, the compiler needs:
1. Module resolution - finding source files from `use` paths
2. Dependency graph - determining compilation order
3. Symbol visibility - `pub` vs private declarations
4. Cycle detection - handling mutual dependencies

These are independent of the linking strategy and needed regardless.
