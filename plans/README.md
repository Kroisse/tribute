# Tribute Development Plans

This directory contains implementation plans for major features.

> **Note**: A new comprehensive language design has been created in `new-plans/` directory.
> The new design introduces algebraic effects (abilities), row polymorphic types, and a dual-target
> compiler architecture (Cranelift + WasmGC). Some plans below have been deprecated in favor of the new design.

## New Design Documents

See **[../new-plans/](../new-plans/)** for the comprehensive language redesign:

| Document | Description |
|----------|-------------|
| [tribute-design.md](../new-plans/tribute-design.md) | Overall language design overview |
| [tribute-types.md](../new-plans/tribute-types.md) | Type system (struct/enum, UFCS) |
| [tribute-abilities.md](../new-plans/tribute-abilities.md) | Ability (algebraic effects) system |
| [tribute-modules.md](../new-plans/tribute-modules.md) | Module system and name resolution |
| [tribute-type-inference.md](../new-plans/tribute-type-inference.md) | Type inference with effect rows |
| [tribute-implementation.md](../new-plans/tribute-implementation.md) | Implementation strategy |

---

## Legacy Plans

### Completed

| Plan | Status |
|------|--------|
| [01-syntax-modernization.md](01-syntax-modernization.md) | ✅ **COMPLETED** |
| [01.01-string-interpolation.md](01.01-string-interpolation.md) | ✅ **COMPLETED** |

### Active (Transitioning)

| Plan | Status | Notes |
|------|--------|-------|
| [02-compiler-implementation.md](02-compiler-implementation.md) | 🔄 **TRANSITIONING** | Current Cranelift work continues; new design expands scope |
| [02.02-cranelift-completion.md](02.02-cranelift-completion.md) | 🚧 **IN PROGRESS** | Active Cranelift implementation |

### Deprecated

| Plan | Status | Superseded By |
|------|--------|---------------|
| [02.01-hir-to-mlir.md](02.01-hir-to-mlir.md) | ❌ **DEPRECATED** | `new-plans/tribute-implementation.md` |
| [02.02-mlir-interpreter.md](02.02-mlir-interpreter.md) | ❌ **DEPRECATED** | Cranelift approach |
| [04-static-type-system.md](04-static-type-system.md) | ❌ **DEPRECATED** | `new-plans/tribute-type-inference.md` |

### Planned (May Need Updates)

| Plan | Status | Notes |
|------|--------|-------|
| [03-lsp-implementation.md](03-lsp-implementation.md) | 📋 **PLANNED** | May need updates for new type/ability system |
| [05-standard-library.md](05-standard-library.md) | 📋 **PLANNED** | |
| [06-package-manager.md](06-package-manager.md) | 📋 **PLANNED** | |
| [07-testing-framework.md](07-testing-framework.md) | 📋 **PLANNED** | |
| [08-documentation-system.md](08-documentation-system.md) | 📋 **PLANNED** | |

---

## Implementation Roadmap

### Current Focus
1. Complete Cranelift compiler (Plan 02.02) for basic language features
2. Integrate new language design (new-plans/) for ability system and advanced types

### Future Phases
1. **Ability System**: Implement algebraic effects with evidence passing
2. **Static Types**: Row polymorphic effect types with bidirectional inference
3. **WasmGC Target**: Add WebAssembly compilation with Stack Switching
4. **Developer Tools**: LSP, package manager, documentation
