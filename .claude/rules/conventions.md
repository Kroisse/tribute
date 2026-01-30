# Code Conventions

## Salsa Database Pattern

All compilation stages use Salsa for incremental compilation:

```rust
#[salsa::tracked]
pub fn insert_boxing<'db>(
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

Effect rows are managed in `crates/tribute-front/src/typeck/effect_row.rs`.

### Bidirectional Type Checking

Two modes in `crates/tribute-front/src/typeck/checker.rs`:
- **Infer mode**: Synthesize type from expression
- **Check mode**: Verify expression against expected type

```rust
fn infer_expr(&mut self, expr: ...) -> Type { ... }
fn check_expr(&mut self, expr: ..., expected: Type) { ... }
```

### Type Variables and Unification

- Fresh type variables for unknowns
- Union-find based constraint solver in `crates/tribute-front/src/typeck/solver.rs`
- Substitution applied after solving

## Name Resolution

Name resolution is performed at the AST level in `tribute-front`.

Two-phase resolution:

1. **Basic resolution** (`crates/tribute-front/src/resolve.rs`): Resolves names and paths
   - Builds `ModuleEnv` with bindings from definitions
   - Resolves variable references to their definitions
   - Handles qualified paths (e.g., `Foo::bar`)

2. **Type-directed (TDNR)** (`crates/tribute-front/src/tdnr.rs`): Resolves UFCS after type inference
   - `expr.method(args)` → `Type::method(expr, args)`
   - Requires inferred type information from typecheck phase

## Dialect Operations

### Creating Operations

Always use typed helper functions to create dialect operations:

```rust
// ✅ Use typed helper functions
let yield_op = wasm::r#yield(db, location, value);
let call_op = func::call(db, location, callee, args, result_ty);

// ❌ Never use manual operation construction
// Operation::of_name has been removed to enforce type safety at compile time
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

Three kinds of bindings in `crates/tribute-front/src/resolve/env.rs`:

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

- Use `cargo nextest run` for running tests (preferred over `cargo test`)
- Use `insta` for snapshot testing
- Run `cargo insta review` when snapshots fail
- Package-specific tests: `cargo nextest run -p <crate-name>`

## Development Workflow

### Starting a New Task

1. **Always start from latest origin/main**:
   ```bash
   git fetch origin
   git checkout -b <branch-name> origin/main
   ```

2. **Branch naming**: Use conventional prefixes
   - `feature/` - New features
   - `fix/` - Bug fixes
   - `refactor/` - Code restructuring
   - `docs/` - Documentation only

### During Development

- Make atomic commits with clear messages (Conventional Commits format)
- Run tests frequently: `cargo nextest run --workspace`
- Use `cargo clippy` and `cargo fmt` before committing

### Completing Work

1. **Create a Pull Request** when work is ready for review
2. **Address review comments** before merging
3. **Squash or rebase** if commit history is messy

### Issue Workflow

- Reference issues in commits: `fix: resolve type inference (#123)`
- Create follow-up issues for out-of-scope discoveries
- Close issues via PR merge, not manually
