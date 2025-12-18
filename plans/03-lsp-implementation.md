# LSP Implementation Plan

## Overview

Implementation of a Language Server Protocol (LSP) server for Tribute to provide IDE support in VSCode and other editors.

## Priority: Medium (3/4)

Essential for developer experience, can be developed in parallel with the compiler.

## Features to Implement

### 1. Basic Features
- Syntax highlighting (Tree-sitter based)
- Document synchronization
- Diagnostics (errors/warnings)

### 2. Navigation
- Go to definition
- Find references
- Document symbols
- Workspace symbols

### 3. Code Intelligence
- Hover information (type + effect info)
- Auto-completion
- Signature help
- Code actions (quick fixes)

### 4. Refactoring
- Rename symbol
- Extract function
- Inline variable

## Implementation Steps

### Phase 1: LSP Server Setup

1. Create `tribute-lsp` crate
2. Dependencies:
   - `tower-lsp` - LSP protocol
   - `tokio` - async runtime
3. Implement basic server lifecycle
4. Set up document store with Salsa DB integration

### Phase 2: Diagnostics

1. Hook into existing error reporting system
2. Convert `Diagnostic` to LSP format
3. Implement incremental parsing
4. Real-time error reporting

### Phase 3: Code Navigation

1. Build symbol index based on TrunkIR
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
                Parser  TrunkIR  Type Info
```

### TrunkIR Integration

LSP utilizes information from various TrunkIR dialects:

| LSP Feature | TrunkIR Source |
|-------------|----------------|
| Go to definition | `src.*` → resolved symbols |
| Hover (type) | `type.*` definitions |
| Hover (effect) | `ability.*` annotations |
| Completion | `func.*` declarations |
| Diagnostics | Pass invariant violations |

### Ability/Effect Information Display

```rust
// Hover example
fn fetch(url: String) ->{Http} Response
//                     ^^^^^^
// Effect: Http
// - Http::get(url: String) -> Response
// - Http::post(url: String, body: String) -> Response
```

Display type information including effect rows on hover:
- Function effect signatures
- List of operations for the ability
- Whether a handler is required

### Type Inference Result Display

```rust
let x = some_function()
//  ^
// Inferred type: Option(Int)
// Inferred effects: {State(Int), Console}
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

## Future Enhancements

### Ability-aware Features

1. **Handler suggestion**: Suggest handlers when effects are unhandled
2. **Effect propagation view**: Visualize effect propagation through call chains
3. **Resume/abort tracking**: Track continuation usage

### Advanced Diagnostics

1. **Unused continuation warning**: Warn when one-shot continuation is unused
2. **Effect mismatch**: Detect expected vs actual effect mismatches
3. **Scoped resumption violation**: Detect continuation scope escape attempts
