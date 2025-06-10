# Salsa Integration Guide for Tribute

This document explains how Salsa (an incremental computation framework) is integrated and used in the Tribute project.

## Table of Contents

1. [What is Salsa?](#what-is-salsa)
2. [Salsa Architecture in Tribute](#salsa-architecture-in-tribute)
3. [Core Components](#core-components)
4. [Implementation Guide](#implementation-guide)
5. [Practical Examples](#practical-examples)
6. [Writing Tests](#writing-tests)
7. [Future Extensions](#future-extensions)

## What is Salsa?

Salsa is an incremental computation framework written in Rust. Key features:

- **Automatic dependency tracking**: Automatically tracks dependencies between queries
- **Incremental recomputation**: Only recomputes affected parts when inputs change
- **Memoization**: Automatically caches computation results
- **Parallel execution**: Can execute independent queries in parallel

## Salsa Architecture in Tribute

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SourceFile │ ──> │    Parse    │ ──> │   Program   │
│   (Input)   │     │   (Query)   │     │  (Tracked)  │
└─────────────┘     └─────────────┘     └─────────────┘
                            │
                            ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Lower to   │ ──> │ HirProgram  │
                    │     HIR     │     │  (Tracked)  │
                    └─────────────┘     └─────────────┘
```

## Core Components

### 1. Database Definition

In modern Salsa, the database is defined directly without a trait:

```rust
// tribute-ast/src/database.rs
#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
}
```

### 2. Input Types

Inputs represent data provided from the outside:

```rust
#[salsa::input]
pub struct SourceFile {
    #[return_ref]
    pub path: PathBuf,
    #[return_ref]
    pub text: String,
}
```

### 3. Tracked Types

Tracked types store query results and Salsa manages their lifecycle:

```rust
#[salsa::tracked]
pub struct Program<'db> {
    pub source_file: SourceFile,
    #[return_ref]
    pub expressions: Vec<TrackedExpression<'db>>,
}

#[salsa::tracked]
pub struct TrackedExpression<'db> {
    pub expr: Expression,
    pub span: Span,
}
```

### 4. Accumulators

Accumulators collect side effects (errors, warnings, etc.) during query execution. A single `Diagnostic` type can be used across all compilation phases:

```rust
#[salsa::accumulator]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
    pub severity: DiagnosticSeverity,
    pub phase: CompilationPhase, // Optional: track which phase generated this
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationPhase {
    Parsing,
    HirLowering,
    TypeChecking,
    Optimization,
}
```

### 5. Query Functions

Queries are defined as standalone functions with the `#[salsa::tracked]` attribute:

```rust
// AST level queries
#[salsa::tracked]
pub fn parse_source_file(db: &dyn salsa::Database, source: SourceFile) -> Program<'_> {
    // Implementation
}

#[salsa::tracked]
pub fn diagnostics(db: &dyn salsa::Database, source: SourceFile) -> Vec<Diagnostic> {
    // Collect accumulated diagnostics
    parse_source_file::accumulated::<Diagnostic>(db, source)
}

// HIR level queries
#[salsa::tracked]
pub fn lower_source_to_hir(db: &dyn salsa::Database, source: SourceFile) -> HirProgram<'_> {
    // Implementation
}
```

## Implementation Guide

### 1. Adding New Queries

To add a new query, simply create a tracked function:

```rust
// Define the query as a tracked function
#[salsa::tracked]
pub fn type_check(db: &dyn salsa::Database, program: HirProgram<'_>) -> TypedProgram<'_> {
    // Type checking logic
    let mut type_env = TypeEnvironment::new();
    
    // Type check functions
    for (name, func) in program.functions(db) {
        // Add to accumulator on error
        if let Err(e) = check_function(&mut type_env, func) {
            Diagnostic {
                message: e.to_string(),
                span: func.span(db),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }.accumulate(db);
        }
    }
    
    // Return result
    TypedProgram::new(db, program, type_env)
}
```

### 2. Creating Tracked Types

```rust
// Create tracked types with new method
let program = Program::new(
    db,
    source_file,
    expressions.into_iter().map(|(expr, span)| {
        TrackedExpression::new(db, expr, span)
    }).collect(),
);
```

### 3. Building Dependency Chains

```rust
// Dependencies are automatically created when queries call other queries
#[salsa::tracked]
pub fn compile(db: &dyn salsa::Database, source: SourceFile) -> CompiledProgram {
    // Depends on parse_source_file
    let ast = parse_source_file(db, source);
    
    // Depends on lower_source_to_hir
    let hir = lower_source_to_hir(db, source);
    
    // Depends on type_check
    let typed = type_check(db, hir);
    
    // Generate code
    generate_code(typed)
}
```

### 4. Accessing Accumulated Values

```rust
// To collect accumulated values from a specific query
#[salsa::tracked]
pub fn all_diagnostics(db: &dyn salsa::Database, source: SourceFile) -> Vec<Diagnostic> {
    // Get diagnostics accumulated during parsing
    let parse_diags = parse_source_file::accumulated::<Diagnostic>(db, source);
    
    // Get diagnostics accumulated during HIR lowering
    let hir_diags = lower_source_to_hir::accumulated::<Diagnostic>(db, source);
    
    // Combine and return
    parse_diags.into_iter()
        .chain(hir_diags.into_iter())
        .collect()
}
```

## Practical Examples

### 1. Basic Usage

```rust
use tribute_ast::{TributeDatabaseImpl, parse_source_file, diagnostics};

// Create database
let db = TributeDatabaseImpl::default();

// Create source file
let source = SourceFile::new(
    &db,
    PathBuf::from("example.trb"),
    "(+ 1 2 3)".to_string(),
);

// Parse
let program = parse_source_file(&db, source);

// Check diagnostics
let diagnostics = diagnostics(&db, source);
for diag in diagnostics {
    println!("{}: {}", diag.severity, diag.message);
}
```

### 2. Incremental Compilation

```rust
// Initial parsing
let mut db = TributeDatabaseImpl::default();
let source = SourceFile::new(&db, path, text);
let program1 = parse_source_file(&db, source);

// Modify source
source.set_text(&mut db).to("(+ 1 2 3 4)".to_string());

// Reparse - automatically invalidated and recomputed
let program2 = parse_source_file(&db, source);

// program1 and program2 are different results
```

### 3. Attach Pattern (for testing)

```rust
TributeDatabaseImpl::default().attach(|db| {
    // Use db within this block
    let source = SourceFile::new(db, path, text);
    let program = parse_source_file(db, source);
    
    // Test assertions
    assert_eq!(program.expressions(db).len(), 1);
});
```

## Writing Tests

### 1. Unit Tests

```rust
#[test]
fn test_parse_simple_expression() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceFile::new(
            db,
            PathBuf::from("test.trb"),
            "(+ 1 2)".to_string(),
        );
        
        let program = parse_source_file(db, source);
        let exprs = program.expressions(db);
        
        assert_eq!(exprs.len(), 1);
        match &exprs[0].expr(db) {
            Expression::Call { func, args } => {
                assert_eq!(func.as_str(), "+");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call expression"),
        }
    });
}
```

### 2. Incremental Computation Tests

```rust
#[test]
fn test_incremental_parsing() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceFile::new(db, path, "(+ 1 2)".to_string());
        
        // First parse
        let program1 = parse_source_file(db, source);
        let revision1 = db.salsa_runtime().current_revision();
        
        // Parse again with same content - uses cache
        let program2 = parse_source_file(db, source);
        let revision2 = db.salsa_runtime().current_revision();
        assert_eq!(revision1, revision2); // No revision change
        
        // Modify source
        source.set_text(db).to("(+ 1 2 3)".to_string());
        
        // Reparse - new computation
        let program3 = parse_source_file(db, source);
        let revision3 = db.salsa_runtime().current_revision();
        assert_ne!(revision2, revision3); // Revision changed
    });
}
```

### 3. Diagnostic Tests

```rust
#[test]
fn test_parse_error_diagnostics() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceFile::new(
            db,
            PathBuf::from("error.trb"),
            "(+ 1 2".to_string(), // Missing closing parenthesis
        );
        
        let _ = parse_source_file(db, source);
        let diagnostics = diagnostics(db, source);
        
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, DiagnosticSeverity::Error);
        assert!(diagnostics[0].message.contains("closing"));
    });
}
```

## Future Extensions

### 1. Type System

```rust
// Tracked type for type information
#[salsa::tracked]
pub struct TypedExpression<'db> {
    pub expr: HirExpr<'db>,
    pub ty: Type,
}

// Type checking query
#[salsa::tracked]
fn type_check_expr(
    db: &dyn salsa::Database,
    expr: HirExpr<'_>,
    env: TypeEnvironment<'_>,
) -> TypedExpression<'_> {
    // Implementation
}
```

### 2. Optimization Passes

```rust
// Optimized HIR
#[salsa::tracked]
pub struct OptimizedHir<'db> {
    pub original: HirProgram<'db>,
    #[return_ref]
    pub optimized_functions: BTreeMap<Identifier, HirFunction<'db>>,
}

// Optimization queries
#[salsa::tracked]
fn constant_folding(db: &dyn salsa::Database, hir: HirProgram<'_>) -> OptimizedHir<'_> {
    // Implementation
}

#[salsa::tracked]
fn dead_code_elimination(db: &dyn salsa::Database, hir: OptimizedHir<'_>) -> OptimizedHir<'_> {
    // Implementation
}
```

### 3. Language Server Protocol (LSP)

```rust
// LSP queries
#[salsa::tracked]
fn find_definition(
    db: &dyn salsa::Database,
    file: SourceFile,
    position: Position,
) -> Option<Location> {
    // Implementation
}

#[salsa::tracked]
fn find_references(
    db: &dyn salsa::Database,
    definition: Location,
) -> Vec<Location> {
    // Implementation
}

#[salsa::tracked]
fn hover_info(
    db: &dyn salsa::Database,
    file: SourceFile,
    position: Position,
) -> Option<HoverInfo> {
    // Implementation
}
```

### 4. Parallel Processing

```rust
// Process multiple files in parallel
#[salsa::tracked]
fn compile_workspace(db: &dyn salsa::Database, files: Vec<SourceFile>) -> WorkspaceResult {
    use rayon::prelude::*;
    
    let results: Vec<_> = files
        .par_iter()
        .map(|file| compile(db, *file))
        .collect();
        
    WorkspaceResult::new(db, results)
}
```

## Best Practices

1. **Keep queries pure**: Side effects only through Accumulators
2. **Split into small queries**: Improves reusability and incremental computation efficiency
3. **Use Tracked types**: Store intermediate results as Tracked types
4. **Use attach pattern for tests**: Ensures isolation between tests
5. **Minimize dependencies**: Pass only necessary data to queries
6. **Use `&dyn salsa::Database`**: Always use the generic database type in query signatures

## Debugging Tips

```rust
// Enable Salsa event logging
env_logger::init();
RUST_LOG=salsa=debug cargo test

// Trace query execution
db.salsa_runtime().report_synthetic_reads(true);

// Check dependency graph
let deps = db.salsa_runtime().dependencies();

// Debug a specific query
#[salsa::tracked(recovery_fn = recover_from_parse_error)]
pub fn parse_with_recovery(db: &dyn salsa::Database, source: SourceFile) -> Program<'_> {
    // Implementation
}

fn recover_from_parse_error(
    db: &dyn salsa::Database,
    _cycle: &salsa::Cycle,
    source: SourceFile,
) -> Program<'_> {
    // Return a default/error program
    Program::new(db, source, vec![])
}
```

This guide explains how to effectively use Salsa in the Tribute project. For more information, see the [official Salsa documentation](https://salsa-rs.github.io/salsa/).