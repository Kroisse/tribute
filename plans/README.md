# Tribute Development Plans

This directory contains detailed plans for major features and improvements to the Tribute programming language.

## Plans Overview

### 1. [Syntax Modernization](01-syntax-modernization.md) - **Priority: High** ✅ **COMPLETED**
Transform from Lisp-style S-expressions to a more modern, familiar syntax. This is the foundation for all other improvements and directly impacts user experience.

### 2. [MLIR Compiler Implementation](02-compiler-implementation.md) - **Priority: Medium** 📋 **PLANNED**
Build a complete compilation pipeline using MLIR/melior to generate native binaries with a custom Tribute dialect. Essential for performance and real-world deployment.

#### 2.01. [HIR to MLIR Lowering](02.01-hir-to-mlir.md) - **Priority: High** 📋 **PLANNED**
Foundation step: implement translation from HIR to custom MLIR dialect, establishing the IR used by both interpreter and compiler.

#### 2.02. [MLIR Interpreter](02.02-mlir-interpreter.md) - **Priority: Medium-High** 📋 **PLANNED**
Validation step: implement MLIR interpreter to test dialect and optimization passes before native compilation.

### 3. [LSP Implementation](03-lsp-implementation.md) - **Priority: Medium**
Create Language Server Protocol support for IDE integration. Critical for developer productivity and adoption.

### 4. [Static Type System](04-static-type-system.md) - **Priority: Medium-High**
Add gradual static typing with type inference. Strongly recommended before compiler implementation for better performance and simpler codegen.

## Recommended Implementation Order

1. **Phase 1: Syntax Modernization** ✅ **COMPLETED**
   - Start immediately as it affects all other work
   - Can be done incrementally with backward compatibility

2. **Phase 2: MLIR Foundation**
   - **HIR to MLIR (02.01)**: Establish dialect and translation layer
   - **MLIR Interpreter (02.02)**: Validate dialect with real execution
   - **Native Compilation (02)**: Build on proven MLIR infrastructure
   - Three-step incremental approach minimizes risk

3. **Phase 3: Developer Experience**
   - **LSP (03)**: Leverage MLIR for better language analysis
   - **Type System (04)**: Add as optimization layer over MLIR

4. **Phase 4: Advanced Features**
   - Standard library, package manager
   - Advanced compiler optimizations
   - Effect systems and advanced types

## Cross-Cutting Concerns

- **Testing**: Each feature needs comprehensive test coverage
- **Documentation**: Update docs as features are implemented
- **Performance**: Monitor compilation and runtime performance
- **Compatibility**: Maintain backward compatibility where possible
- **Error Messages**: Focus on helpful, actionable error reporting

## Resource Requirements

- **Syntax**: 2-3 months with Tree-sitter expertise ✅ **COMPLETED**
- **HIR to MLIR**: 2-3 weeks with MLIR basics
- **MLIR Interpreter**: 2-3 weeks building on established dialect
- **Native Compiler**: 3-4 months with MLIR/LLVM knowledge  
- **LSP**: 3-4 months with IDE integration experience
- **Types**: 3-5 months with type theory background

## Next Steps

1. Review and refine each plan
2. Set up tracking for implementation progress
3. Begin with syntax modernization prototype
4. Recruit contributors with relevant expertise