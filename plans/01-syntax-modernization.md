# Syntax Modernization Plan

## Overview
Transform Tribute from Lisp-style S-expressions to a more modern, familiar syntax while maintaining the language's core semantics.

## Priority: High (1/4)
This should be the first major change as it affects all other development areas and user experience.

## Current State
```tribute
(fn add (a b)
  (+ a b))

(let result (add 5 3))
(print_line result)
```

## Target Syntax
```tribute
fn add(a, b) {
  a + b
}

let result = add(5, 3)
print_line(result)
```

## Implementation Steps

### Phase 1: Grammar Definition
1. Design new syntax specification
   - Function definitions: `fn name(params) { body }`
   - Variable bindings: `let name = expr`
   - Function calls: `name(args)`
   - Infix operators: `a + b`, `a * b`, etc.
   - Control flow: `if expr { } else { }`, `match expr { patterns }`
2. Update Tree-sitter grammar incrementally
3. Support both old and new syntax during transition

### Phase 2: Parser Updates
1. Modify `tree-sitter-tribute/grammar.js`
2. Update precedence rules for operators
3. Add support for:
   - Block expressions with `{}`
   - Statement separators (newlines or semicolons)
   - Comments (`//` and `/* */`)
   - String interpolation

### Phase 3: AST/HIR Adjustments
1. Minimal changes to AST structure (expressions remain similar)
2. Update HIR lowering to handle new syntax nodes
3. Ensure backward compatibility during transition

### Phase 4: Tooling Updates
1. Update syntax highlighting rules
2. Modify example files in `lang-examples/`
3. Update documentation

### Phase 5: Error Recovery
1. Leverage Tree-sitter's built-in error recovery features:
   - ERROR nodes for invalid syntax while maintaining parse tree structure
   - Missing nodes for expected but absent tokens
   - Incremental parsing to isolate errors
2. Design grammar with error recovery in mind:
   - Use `optional()` for commonly omitted elements
   - Add recovery rules at statement boundaries
   - Define synchronization points (e.g., newlines, semicolons, closing braces)
3. Implement intelligent error messages:
   - Track ERROR node locations and expected tokens
   - Provide context-aware suggestions
   - Show multiple related errors when appropriate
4. Testing error recovery:
   - Create corpus of common syntax errors
   - Ensure partial programs are still analyzable
   - Verify LSP features work with incomplete code

## Technical Considerations
- Maintain expression-oriented design
- Keep parser performance optimal
- Ensure clear error messages for syntax errors
- Support gradual migration of existing code

## Dependencies
- Tree-sitter grammar expertise
- Careful precedence and associativity design
- Test suite expansion for new syntax

## Success Criteria
- All existing features work with new syntax
- Clear migration path for existing code
- Improved readability and familiarity
- No performance regression