# LSP Implementation Plan

## Overview
Implement a Language Server Protocol (LSP) server for Tribute to provide IDE support in VSCode and other editors.

## Priority: Medium (3/4)
Essential for developer experience, can be developed in parallel with compiler.

## Features to Implement
1. **Basic Features**
   - Syntax highlighting (via Tree-sitter)
   - Document synchronization
   - Diagnostics (errors/warnings)
   
2. **Navigation**
   - Go to definition
   - Find references
   - Document symbols
   - Workspace symbols

3. **Code Intelligence**
   - Hover information
   - Auto-completion
   - Signature help
   - Code actions (quick fixes)

4. **Refactoring**
   - Rename symbol
   - Extract function
   - Inline variable

## Implementation Steps

### Phase 1: LSP Server Setup
1. Create `tribute-lsp` crate
2. Add dependencies:
   - `tower-lsp` for LSP protocol
   - `tokio` for async runtime
3. Implement basic server lifecycle
4. Set up document store with Salsa integration

### Phase 2: Diagnostics
1. Hook into existing error reporting
2. Convert `Diagnostic` to LSP format
3. Implement incremental parsing
4. Real-time error reporting

### Phase 3: Code Navigation
1. Build symbol index using HIR
2. Implement definition provider
3. Add reference finding
4. Create document outline

### Phase 4: Code Completion
1. Context-aware completion
2. Function signature completion
3. Variable name suggestions
4. Built-in function documentation

### Phase 5: VSCode Extension
1. Create `tribute-vscode` extension
2. Package Tree-sitter grammar for syntax highlighting
3. Configure language client
4. Add language configuration (brackets, comments)

## Architecture Design
```
VSCode ←→ LSP Client ←→ Tribute LSP Server
                            ↓
                     Salsa Database
                     ↙     ↓      ↘
                Parser   HIR    Type Info
```

## Technical Considerations
- Leverage Salsa for incremental computation
- Efficient document synchronization
- Cancellation support for long operations
- Memory usage optimization

## Testing Strategy
- Unit tests for each LSP feature
- Integration tests with mock client
- Manual testing in VSCode
- Performance benchmarks

## Success Criteria
- Sub-100ms response for most operations
- Accurate diagnostics and navigation
- Stable operation with large files
- Published VSCode extension