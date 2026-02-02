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

---

## Implementation Plans

### Active

| Plan | Description | Priority |
| ---- | ----------- | -------- |
| [02.04-wasm-translation.md](02.04-wasm-translation.md) | Wasm backend (see #38-41 for remaining work) | High |

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

---

## Implementation Roadmap

### Current Focus

1. **TrunkIR Pipeline** - Compiler implementation based on `new-plans/ir.md`
2. **Type Inference** - Bidirectional typing based on `new-plans/type-inference.md`

### Future Phases

1. **Ability System**: Algebraic effects via evidence passing (see #23-26)
2. **Cranelift Backend**: libmprompt + Boehm GC integration
3. **WasmGC Backend**: Stack Switching support
4. **Developer Tools**: LSP (see #31-37), package manager, documentation
