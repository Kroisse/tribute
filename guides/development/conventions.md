# Code Conventions

## Frontend Queries and IR Passes

Use Salsa tracked queries for source-dependent frontend computations and cached
artifact boundaries. Inputs and queries take `&dyn salsa::Database`; diagnostics
use the `Accumulator` trait from inside an active tracked query.

Shared and target lowering mutate an arena `IrContext` in one compilation
session. They use `PassManager`, typed pass targets, and explicit analysis-cache
invalidation. They do not wrap each IR mutation in a Salsa query. See
[the Salsa guide](../salsa.md) and [TrunkIR](ir.md).

## Symbols

Use `Symbol::new(&str)` for any runtime name. string_cache selects the static
set, inline storage, or the dynamic set automatically; the constructor does not
require a `'static` string. `Symbol::from(&str)` also accepts borrowed names.

Use `trunk_ir::symbol!("func")` for a literal in trunk-ir's generated symbol set.
It creates the symbol at compile time, including in `const` declarations,
without a runtime lookup. In `match`, compare symbols in a guard instead of
using this macro as a pattern. Short names use inline storage. Unknown
literals and runtime expressions are rejected; use `Symbol::new` for those.
The static set is generated from trunk-ir's source, so adding a literal to a
consumer crate does not register it in that set.

`symbols!` declares named helper functions and continues to support arbitrary
literal names through the runtime constructor.

See the [string_cache documentation](https://docs.rs/string_cache) for the atom
representations and generated literal macros.

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
ability row. Source-logical IR preserves that signature, including the effect
row, and carries no calling convention; the CPS strategy's class specialization
assigns conventions to definitions, convention entries, and indirect transfers.
Shared CPS legalization alone adds hidden callable parameters.
Do not infer operation kind, ownership, or callable ABI from symbol spelling,
body shape, or an erased pointer type.

### Bidirectional Type Checking

Two modes:

- **Infer mode**: Synthesize type from expression
- **Check mode**: Verify expression against expected type

### Name Resolution

Two-phase resolution:

1. **Basic resolution**: Resolves names and paths, builds `PackageEnv`, and
   records on each call the functions its scope names
2. **Type-directed (TDNR)**: Type checking selects one of them by the call's
   types (`expr.method(args)` → `Type::method(expr, args)`). A method call is
   an `ExprKind::Call` whose callee is `ExprKind::Functions`; the selected
   function replaces that callee

## Dialect Operations

### Creating Operations

Use generated builders and wrappers with the arena context. A builder groups
inputs by kind: operands start it, attributes are set by name, and result
types, regions, and successors are each one call:

```rust
let value = arith::Const::operands()
    .value(Attribute::Int(42))
    .results(i32_ty)
    .build(ctx, location);
let result = value.result(ctx);
```

The builder infers result types the declaration fixes:

```rust
// fn cmpi<T: IntegerLike>(
//     predicate: Attr<String>,
//     lhs: Value<T>,
//     rhs: Value<T>,
// ) -> Value<I1> {}
let cmp = arith::Cmpi::operands(lhs, rhs)
    .predicate("slt")
    .build(ctx, location);
```

For a construction that many call sites repeat, define an inherent method on
the wrapper rather than a free constructor function.

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

A pattern that rewrites one operation implements `OpRewritePattern`, naming
the operation's typed wrapper as `type Op`. Its `match_and_rewrite` takes:

- `ctx`: Mutable arena context for querying and building IR
- `op`: The matched operation's typed wrapper; `op.op_ref()` is its `OpRef`
- `rewriter`: `PatternRewriter` that records structural mutations

```rust
impl OpRewritePattern for LowerCall<'_> {
    type Op = func::Call;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        call: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let callee = self.lowered.get(call.callee(ctx))?;
        // ...
        Some(())
    }
}
```

A pattern that matches several operations, a whole dialect, or roots known
only at run time implements `RewritePattern` directly, with an `OpRef` and
an optional `roots`.

`match_and_rewrite` takes `&self`. `PatternApplicator<'p>` holds its patterns
for `'p`, so patterns borrow data the driver computed (`&'p SymbolTable`)
instead of sharing it through `Rc` or `Arc`. A pattern that reports to the
driver writes through a driver-owned `Cell` or `RefCell`; what it writes does
not decide later rewrites, and no pattern caches IR facts across matches.

The applicator offers a pattern only the operations in its roots: `Op` for an
`OpRewritePattern`, `roots` for a `RewritePattern`. A pattern changes attached
IR only through the rewriter: after its first sweep, the applicator revisits
only the operations the recorded changes touch.

A pass that converts types builds its type converter in `run` and passes it to
`PatternApplicator`. `PassManager` holds no converter.

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

Return `Some(())` if the pattern matched and recorded mutations, `None`
otherwise. Each check that rejects the operation is a `?`: on an `Option`,
or on a `Result` after `.ok()`.

Use full conversion where a phase boundary must reject every residual illegal
operation. Bodyless external functions have no body region; inspect optional
regions before using a body accessor.
