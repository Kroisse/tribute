# Code Conventions

## Salsa Database Pattern

All compilation stages use Salsa for incremental compilation:

```rust
#[salsa::tracked]
pub fn stage_resolve<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    // ...
}
```

- Functions marked with `#[salsa::tracked]` are memoized
- Database tracks dependencies automatically
- Use `Accumulator` trait for collecting diagnostics

## Error Handling

Use `derive_more` for error types:

```rust
use derive_more::{Display, Error, From};

#[derive(Debug, Display, Error, From)]
pub enum CompileError {
    #[display("Parse error: {_0}")]
    Parse(ParseError),
    #[display("Type error: {_0}")]
    Type(TypeError),
}
```

## Type System

### Row-Polymorphic Effects

Function types include effect information:

```rust
// Function type: fn(params) ->{effects} return_type
let func_ty = func::Fn::new(db, params, return_ty, effect_row);
```

Effect rows are managed in `crates/tribute-passes/src/typeck/effect_row.rs`.

### Bidirectional Type Checking

Two modes in `crates/tribute-passes/src/typeck/checker.rs`:
- **Infer mode**: Synthesize type from expression
- **Check mode**: Verify expression against expected type

```rust
fn infer_expr(&mut self, expr: ...) -> Type { ... }
fn check_expr(&mut self, expr: ..., expected: Type) { ... }
```

### Type Variables and Unification

- Fresh type variables for unknowns
- Union-find based constraint solver in `typeck/solver.rs`
- Substitution applied after solving

## Name Resolution

Two-phase resolution:

1. **Basic resolution** (`resolve.rs`): Resolves names and paths
   - `src.var` → `func.call` or local reference
   - `src.path` → qualified reference
   - Builds `ModuleEnv` with bindings

2. **Type-directed (TDNR)** (`tdnr.rs`): Resolves UFCS after type inference
   - `expr.method(args)` → `Type::method(expr, args)`
   - Requires inferred type information

## Dialect Operations

### Creating Operations

When creating dialect operations, prefer typed helper functions over `Operation::of_name`:

```rust
// ✅ Preferred: Use typed helper functions
let yield_op = wasm::r#yield(db, location, value);
let call_op = func::call(db, location, callee, args, result_ty);

// ❌ Avoid: Manual operation construction
let yield_op = Operation::of_name(db, location, "wasm.yield")
    .operands(idvec![value])
    .build();
```

### Matching Operations

When matching dialect operations, prefer typed wrappers over manual dialect/name comparison:

```rust
// ✅ Preferred: Use from_operation for type-safe matching
if let Ok(bind_op) = tribute_pat::Bind::from_operation(db, op) {
    let name = bind_op.name(db);
    // ...
}

// ❌ Avoid: Manual dialect and name comparison
let dialect = op.dialect(db);
let op_name = op.name(db);
if dialect == tribute_pat::DIALECT_NAME() && op_name == tribute_pat::BIND() {
    // ...
}
```

Benefits of typed helpers and wrappers:
- Type-safe access to operation attributes and operands
- Compile-time verification of operation structure
- Cleaner, more readable code

## Bindings

Three kinds of bindings in `resolve.rs`:

```rust
pub enum Binding<'db> {
    Function {
        name: Symbol<'db>,
        ty: Type<'db>,
    },
    Constructor {
        type_name: Symbol<'db>,
        ty: Type<'db>,
        tag: Option<Symbol<'db>>,
        params: IdVec<Type<'db>>,
    },
    TypeDef {
        name: Symbol<'db>,
        ty: Type<'db>,
    },
}
```

## Testing

- Use `insta` for snapshot testing
- Run `cargo insta review` when snapshots fail
- Package-specific tests: `cargo test -p <crate-name>`
