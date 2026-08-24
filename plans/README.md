# Tribute Development Plans

This directory contains implementation plans for major features.

> **Important**: The source of truth for language and compiler design is **[../new-plans/](../new-plans/)**.
> This directory is for implementation plans and work tracking.

## Language Design (Source of Truth)

See **[../new-plans/](../new-plans/)** directory:

| Document | Description |
| -------- | ----------- |
| [design.md](../new-plans/design.md) | Language design overview |
| [syntax.md](../new-plans/syntax.md) | Syntax definition |
| [types.md](../new-plans/types.md) | Type system (struct/enum, UFCS) |
| [abilities.md](../new-plans/abilities.md) | Ability (algebraic effects) system |
| [modules.md](../new-plans/modules.md) | Module system and name resolution |
| [type-inference.md](../new-plans/type-inference.md) | Type inference and effect rows |
| [ir.md](../new-plans/ir.md) | TrunkIR multi-level dialect IR |
| [implementation.md](../new-plans/implementation.md) | Ability implementation strategy |
| [cranelift-backend.md](../new-plans/cranelift-backend.md) | Cranelift native backend architecture |

---

## Implementation Plans

### Active

| Plan | Description | Priority |
| ---- | ----------- | -------- |
| Wasm backend | Shared tail-call CPS to WasmGC (see [wasm-backend.md](../new-plans/wasm-backend.md)) | High |
| Cranelift backend | Native tail-call CPS backend with RC (see [cranelift-backend.md](../new-plans/cranelift-backend.md)) | High |

### Future

| Plan | Description | Priority |
| ---- | ----------- | -------- |
| [05-standard-library.md](05-standard-library.md) | Standard library (ability-based) | Medium |
| [06-package-manager.md](06-package-manager.md) | Package manager | Medium |
| [07-testing-framework.md](07-testing-framework.md) | Testing framework | Medium |
| [08-documentation-system.md](08-documentation-system.md) | Documentation system | Low |

### Research

| Document | Description |
| -------- | ----------- |
| [02.03-wasm-runtime-research.md](02.03-wasm-runtime-research.md) | WebAssembly runtime research (WasmGC + WASI) |
| [02.04-wasm-translation.md](02.04-wasm-translation.md) | Superseded Wasm implementation plan |

---

## Implementation Roadmap

### Current Focus

1. **TrunkIR Pipeline** - Compiler implementation based on `new-plans/ir.md`
2. **Type Inference** - Bidirectional typing based on `new-plans/type-inference.md`

### Future Phases

1. **Ability System**: Algebraic effects via evidence passing and tail-call CPS
2. **Cranelift Backend**: Native lowering with reference counting
3. **Wasm Backend**: Emit the shared tail-call CPS representation
4. **Developer Tools**: LSP (see #31-37), package manager, documentation
