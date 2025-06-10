# Tribute Development Plans

This directory contains detailed plans for major features and improvements to the Tribute programming language.

## Plans Overview

### 1. [Syntax Modernization](01-syntax-modernization.md) - **Priority: High** ✅ **COMPLETED**
Transform from Lisp-style S-expressions to a more modern, familiar syntax. This is the foundation for all other improvements and directly impacts user experience.

### 2. [MLIR Compiler Implementation](02-compiler-implementation.md) - **Priority: Medium** 📋 **PLANNED**
Build a complete compilation pipeline using MLIR/melior to generate native binaries with a custom Tribute dialect. Essential for performance and real-world deployment.

### 3. [LSP Implementation](03-lsp-implementation.md) - **Priority: Medium**
Create Language Server Protocol support for IDE integration. Critical for developer productivity and adoption.

### 4. [Static Type System](04-static-type-system.md) - **Priority: Medium-High**
Add gradual static typing with type inference. Strongly recommended before compiler implementation for better performance and simpler codegen.

## Recommended Implementation Order

1. **Phase 1: Syntax Modernization** ✅ **COMPLETED**
   - Start immediately as it affects all other work
   - Can be done incrementally with backward compatibility

2. **Phase 2: Type System Foundation**
   - **Option A**: Implement basic static typing before compiler
   - **Option B**: Proceed with dynamic compiler (slower, more complex)
   - Strong recommendation for Option A

3. **Phase 3: Compiler & LSP**
   - **Compiler**: Much simpler with type system foundation
   - **LSP**: Can provide better IntelliSense with type information

4. **Phase 4: Advanced Features**
   - Expand type system with generics, effects
   - Advanced compiler optimizations

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