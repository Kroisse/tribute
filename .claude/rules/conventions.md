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

## Formatting Utilities

`tribute_core::fmt` provides helpers for diagnostic messages:

```rust
use tribute_core::fmt::{joined, joined_by};

// joined(separator, iterable) -> impl Display
format!("unhandled effects: {}", joined(", ", &effects))

// joined_by(separator, iterable, formatter) -> impl Display
// Custom formatting per item, zero allocation
format!("{}", joined_by(", ", &items, |item, f| write!(f, "#{item}")))
```

## Type System

### Row-Polymorphic Effects

Function types include effect information:

```rust
// Function type: fn(params) ->{effects} return_type
let func_ty = func::Fn::new(db, params, return_ty, effect_row);
```

### Bidirectional Type Checking

Two modes:

- **Infer mode**: Synthesize type from expression
- **Check mode**: Verify expression against expected type

### Name Resolution

Two-phase resolution:

1. **Basic resolution**: Resolves names and paths, builds `ModuleEnv`
2. **Type-directed (TDNR)**: Resolves UFCS after type inference
   (`expr.method(args)` → `Type::method(expr, args)`)

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

When matching dialect operations, prefer typed wrappers over manual
dialect/name comparison:

```rust
// ✅ Preferred: Use from_operation for type-safe matching
if let Ok(call_op) = func::Call::from_operation(db, op) {
    let callee = call_op.callee(db);
    // ...
}

// ❌ Avoid: Manual dialect and name comparison
let dialect = op.dialect(db);
let op_name = op.name(db);
if dialect == func::DIALECT_NAME() && op_name == func::CALL() {
    // ...
}
```

## Rewrite Patterns

### Pattern Interface

Patterns implement `RewritePattern` with `match_and_rewrite`:

- `op`: Original operation (for matching via
  `from_operation`, attribute/region access)
- `rewriter`: `PatternRewriter` for operand access
  and mutations

### Operand Access

Always use `rewriter.operand(i)` for operands —
never `op.operands(db)` (original, possibly stale).

### Mutation Methods

| Method | Description |
| ------ | ----------- |
| `rewriter.replace_op(new_op)` | Replace the current operation |
| `rewriter.insert_op(op)` | Insert before the replacement |
| `rewriter.erase_op(vals)` | Erase, mapping results to given values |
| `rewriter.add_module_op(op)` | Add a top-level operation to the module |

### Return Value

Return `true` if the pattern matched and recorded mutations, `false` otherwise.
