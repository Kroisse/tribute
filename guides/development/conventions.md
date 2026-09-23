# Code Conventions

## Frontend Queries and IR Passes

Use Salsa tracked queries for source-dependent frontend computations and cached
artifact boundaries. Inputs and queries take `&dyn salsa::Database`; diagnostics
use the `Accumulator` trait from inside an active tracked query.

Shared and target lowering mutate an arena `IrContext` in one compilation
session. They use `PassManager`, typed pass targets, and explicit analysis-cache
invalidation. They do not wrap each IR mutation in a Salsa query. See
[the Salsa guide](../salsa.md) and [TrunkIR](ir.md).

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

Prefer `Itertools::format` and `Itertools::format_with` when embedding sequences
in diagnostic messages. They write elements directly into the surrounding
formatter without allocating an intermediate joined string or per-item strings.

```rust
use itertools::Itertools;

let effects = ["State(Int)", "Console"];
let message = format!("unhandled effects: {}", effects.iter().format(", "));
assert_eq!(message, "unhandled effects: State(Int), Console");

let items = [1, 2, 3];
let custom = format!(
    "items: {}",
    items.iter().format_with(", ", |item, f| f(&format_args!("#{item}")))
);
assert_eq!(custom, "items: #1, #2, #3");
```

These are extension methods provided by `itertools::Itertools`. Formatting
adapters are single-use; create a fresh adapter for each formatting operation.
Use them directly instead of adding custom joining wrappers.

When an owned joined `String` is itself the required result, use
`itertools::join` or `Itertools::join`. Avoid collecting a `Vec<String>` only
for joining; keep collections needed for sorting or reuse.

## Type System

### Row-Polymorphic Effects

Typechecked source callable metadata includes its parameter/result types and
effect row. Source-logical IR preserves that signature and the checked calling
convention. Shared CPS legalization alone adds hidden callable parameters.
Do not infer operation kind, ownership, or callable ABI from symbol spelling,
body shape, or an erased pointer type.

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

Use generated typed builders and wrappers with the arena context:

```rust
let value = arith::r#const(ctx, location, i32_ty, Attribute::Int(42));
let result = value.result(ctx);
```

### Matching Operations

When matching dialect operations, prefer typed wrappers over manual
dialect/name comparison:

```rust
if let Ok(call_op) = func::Call::from_op(ctx, op) {
    let callee = call_op.callee(ctx);
    // ...
}

```

## Rewrite Patterns

### Pattern Interface

Patterns implement `RewritePattern` with `match_and_rewrite`:

- `ctx`: Mutable arena context for querying and building IR
- `op`: The matched `OpRef`, inspected through typed `from_op` wrappers
- `rewriter`: `PatternRewriter` that records structural mutations

### Operand Access

Read current operands through typed wrappers or the arena context. The rewriter
records edits for the applicator; preserve SSA use chains and the conversion
target when constructing replacements.

### Mutation Methods

| Method | Description |
| ------ | ----------- |
| `rewriter.replace_op(new_op)` | Replace the current operation |
| `rewriter.insert_op(op)` | Insert before the replacement |
| `rewriter.erase_op(vals)` | Erase, mapping results to given values |
| `rewriter.add_module_op(op)` | Add a top-level operation to the module |

### Return Value

Return `true` if the pattern matched and recorded mutations, `false` otherwise.

Use full conversion where a phase boundary must reject every residual illegal
operation. Bodyless external functions have no body region; inspect optional
regions before using a body accessor.
