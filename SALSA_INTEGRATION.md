# Salsa 0.22 Integration for Tribute Language

This document describes the Salsa database integration for the Tribute programming language project, providing on-demand, incremental computation for parsing and analysis.

## Overview

The integration uses Salsa 0.22 to provide:
- **Incremental parsing**: Only recompute when source files change
- **On-demand computation**: Parse files only when needed
- **Dependency tracking**: Automatically detect what needs recomputation
- **Diagnostic collection**: Accumulate errors and warnings during parsing

## Key Components

### Database Implementation

```rust
#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
}
```

Concrete implementation of the database that manages all incremental state.

### Input Types

#### SourceFile
```rust
#[salsa::input]
pub struct SourceFile {
    pub path: String,
    pub text: String,
}
```

Represents input source files that can be modified. When the text changes, all dependent computations are automatically invalidated.

### Tracked Types

#### Program
```rust
#[salsa::tracked]
pub struct Program<'db> {
    #[return_ref]
    pub expressions: Vec<TrackedExpression<'db>>,
}
```

Represents a parsed program containing multiple expressions.

#### TrackedExpression
```rust
#[salsa::tracked]
pub struct TrackedExpression<'db> {
    pub expr: Expr,
    pub span: SimpleSpan,
}
```

A single expression with its source location information.

### Accumulator Types

#### Diagnostic
```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct Diagnostic {
    pub message: String,
    pub span: SimpleSpan,
    pub severity: DiagnosticSeverity,
}
```

Collects diagnostics (errors, warnings) during parsing.

## Query Functions

### parse_source_file

```rust
#[salsa::tracked]
pub fn parse_source_file<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Program<'db>
```

Main parsing query that converts a source file into a parsed program. Automatically handles error reporting via the diagnostic accumulator.

### diagnostics

```rust
#[salsa::tracked]
pub fn diagnostics(db: &dyn salsa::Database, source: SourceFile) -> Diagnostics
```

Collects all diagnostics generated during parsing of a source file.

## Usage Patterns

### Basic Usage

```rust
use tribute::{TributeDatabaseImpl, SourceFile, parse_source_file, diagnostics};

// Create database
let db = TributeDatabaseImpl::default();

// Create source file
let source = SourceFile::new(&db, "example.trb".to_string(), "(+ 1 2)".to_string());

// Parse
let program = parse_source_file(&db, source);
let diags = diagnostics(&db, source);

// Access results
for expr in program.expressions(&db) {
    println!("Expression: {}", expr.expr(&db));
}
```

### Test Usage Pattern

For tests, use the `attach` method to ensure proper isolation between test cases:

```rust
use salsa::Database as _;
use tribute::{TributeDatabaseImpl, SourceFile, parse_source_file, diagnostics};

#[test]
fn test_parsing() {
    let result = TributeDatabaseImpl::default().attach(|db| {
        let source = SourceFile::new(db, "test.trb".into(), "(+ 1 2)".to_string());
        let program = parse_source_file(db, source);
        let diags = diagnostics(db, source);

        // Return result that doesn't depend on database lifetime
        (
            program.expressions(db).len(),
            diags.len()
        )
    });

    assert_eq!(result.0, 1); // One expression
    assert_eq!(result.1, 0); // No diagnostics
}
```

**Important**: The `attach` pattern ensures that:
- Each test gets a fresh database instance
- Test outputs don't interfere with each other
- Database lifetime is properly managed
- Results are extracted before the database scope ends

### Incremental Updates

```rust
use salsa::Setter as _;

let mut db = TributeDatabaseImpl::default();
let source = SourceFile::new(&db, "test.trb".to_string(), "(+ 1 2)".to_string());

// Initial parse
let program1 = parse_source_file(&db, source);

// Modify source
source.set_text(&mut db).to("(* 1 2)".to_string());

// Reparse - only recomputes what changed
let program2 = parse_source_file(&db, source);
```

### Error Handling

```rust
let source = SourceFile::new(&db, "invalid.trb".to_string(), "invalid syntax".to_string());
let program = parse_source_file(&db, source);
let diagnostics = diagnostics(&db, source);

for diagnostic in &diagnostics {
    eprintln!("[{}] {} at {}..{}",
        diagnostic.severity,
        diagnostic.message,
        diagnostic.span.start,
        diagnostic.span.end
    );
}
```

## Benefits

1. **Performance**: Only recomputes when inputs change
2. **Memory efficiency**: Cached results are automatically managed
3. **Correctness**: Dependency tracking ensures consistency
4. **Scalability**: Works efficiently with large codebases
5. **Diagnostics**: Centralized error collection and reporting

## Migration from Direct Parsing

The old direct parsing approach:
```rust
let expressions = parse(source_code);
```

Can be replaced with the Salsa-based approach:
```rust
let (program, diagnostics) = parse_with_database(&db, path, source_code);
let expressions: Vec<_> = program.expressions(&db)
    .iter()
    .map(|tracked| (tracked.expr(&db).clone(), tracked.span(&db).clone()))
    .collect();
```

## Integration with Language Server

This Salsa integration provides the foundation for implementing a language server with features like:
- Incremental parsing and validation
- Go-to-definition and find references
- Real-time diagnostics
- Code completion
- Refactoring support

The database can be extended with additional queries for semantic analysis, type checking, and other language services.
