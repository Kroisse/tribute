# MLIR Compiler Implementation Plan

## Overview
Implement a full compilation pipeline from Tribute source code to native binaries using MLIR (via melior crate).

## Priority: Medium (2/4)
Important for performance and deployment, but requires stable syntax first.

## Architecture
```
Source (.trb) → AST → HIR → MIR → MLIR → LLVM IR → Native Binary
```

## Implementation Steps

### Phase 1: MLIR Integration
1. Add `melior` dependency to Cargo.toml
2. Create `tribute-mir` crate for mid-level IR
3. Design MIR representation optimized for MLIR lowering
4. Set up basic MLIR context and module creation

### Phase 2: HIR to MIR Lowering
1. Create lowering pass from HIR to MIR
2. Implement type inference/checking
3. Handle:
   - Function definitions → MLIR functions
   - Variable bindings → SSA form
   - Control flow → MLIR blocks
   - Built-in operations → MLIR operations

### Phase 3: MIR to MLIR Translation
1. Map MIR types to MLIR types
2. Generate MLIR operations:
   - Arithmetic → arith dialect
   - Function calls → func dialect
   - Memory operations → memref dialect
   - Control flow → cf dialect
3. Implement optimizations at MLIR level

### Phase 4: Code Generation
1. MLIR → LLVM IR lowering
2. Link with runtime library
3. Implement built-in functions in C/Rust
4. Generate executable binaries

### Phase 5: Compiler Driver
1. Complete `src/bin/trbc.rs` implementation
2. Add compilation flags:
   - `-O` optimization levels
   - `--emit-mlir` for debugging
   - `--target` for cross-compilation
3. Integrate with build systems

### Phase 6: Incremental Compilation
1. Leverage Salsa for dependency tracking:
   - Track HIR → MIR transformations
   - Cache MLIR modules per function
   - Invalidate only affected compilation units
2. Design compilation unit boundaries:
   - Module-level incremental compilation
   - Function-level for hot reload scenarios
   - Cross-module dependency graph
3. Persistent caching strategy:
   - Serialize Salsa database state
   - Store intermediate representations
   - Fast rebuild from cached artifacts
4. Integration with build systems:
   - Watch mode for continuous compilation
   - Parallel compilation of independent units
   - Build artifact management

### Phase 7: Benchmark Suite
1. Create comprehensive benchmark set:
   - Micro-benchmarks for language features
   - Real-world algorithm implementations
   - Stress tests for compiler scalability
2. Performance tracking infrastructure:
   - Automated benchmark runs
   - Regression detection
   - Comparison with interpreter baseline
3. Optimization validation:
   - Measure impact of MLIR optimizations
   - Profile compilation time vs runtime trade-offs
   - Memory usage analysis

## Technical Challenges
- Memory management strategy (GC vs manual)
- FFI for built-in functions
- Error handling and debugging info
- Cross-platform support

## Dependencies
- `melior` crate for MLIR bindings
- LLVM toolchain
- Runtime library implementation
- Type system design

## Success Criteria
- Compile all language examples to native code
- Performance improvement over interpreter
- Support for major platforms (Linux, macOS, Windows)
- Debugging support (DWARF generation)