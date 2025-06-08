# Tribute Development Plans

This directory contains detailed plans for major features and improvements to the Tribute programming language.

## Plans Overview

### 1. [Syntax Modernization](01-syntax-modernization.md) - **Priority: High**
Transform from Lisp-style S-expressions to a more modern, familiar syntax. This is the foundation for all other improvements and directly impacts user experience.

### 2. [MLIR Compiler Implementation](02-compiler-implementation.md) - **Priority: Medium**
Build a complete compilation pipeline using MLIR/melior to generate native binaries. Essential for performance and real-world deployment.

### 3. [LSP Implementation](03-lsp-implementation.md) - **Priority: Medium**
Create Language Server Protocol support for IDE integration. Critical for developer productivity and adoption.

### 4. [Static Type System](04-static-type-system.md) - **Priority: Low**
Add gradual static typing with type inference. Important for language maturity but requires other foundations first.

## Recommended Implementation Order

1. **Phase 1: Syntax Modernization**
   - Start immediately as it affects all other work
   - Can be done incrementally with backward compatibility

2. **Phase 2: Parallel Development**
   - **Compiler**: Begin after syntax is stable
   - **LSP**: Can start in parallel, initially supporting both syntaxes

3. **Phase 3: Type System**
   - Implement after compiler and LSP are functional
   - Benefits from existing infrastructure

## Cross-Cutting Concerns

- **Testing**: Each feature needs comprehensive test coverage
- **Documentation**: Update docs as features are implemented
- **Performance**: Monitor compilation and runtime performance
- **Compatibility**: Maintain backward compatibility where possible
- **Error Messages**: Focus on helpful, actionable error reporting

## Resource Requirements

- **Syntax**: 2-3 months with Tree-sitter expertise
- **Compiler**: 4-6 months with MLIR/LLVM knowledge
- **LSP**: 3-4 months with IDE integration experience
- **Types**: 3-5 months with type theory background

## Next Steps

1. Review and refine each plan
2. Set up tracking for implementation progress
3. Begin with syntax modernization prototype
4. Recruit contributors with relevant expertise