# TrunkIR System

TrunkIR is Tribute's multi-level dialect IR, inspired by MLIR's dialect concept.

## Core Structures

Located in `crates/trunk-ir/src/context.rs` and `crates/trunk-ir/src/refs.rs`:

- **`IrContext`** - Arena-based IR context holding all operations, blocks,
  regions, and types
- **`OpRef`** - Reference to an operation in the arena
- **`ValueRef`** - Reference to an SSA value
- **`ValueDef`** - Either operation result or block argument
- **`BlockRef`** - Reference to a basic block
- **`RegionRef`** - Reference to a control flow region
- **`TypeRef`** - Reference to an interned type
- **`Attribute`** - Value attributes (bool, int, float, string, type, symbol, span)
- **`Symbol`** - Interned identifier (4 bytes, O(1) comparison)
  - Can hold simple names or qualified paths (e.g., `"map"` or `"std::List::map"`)
  - Path operations available via `ModulePathExt` trait in `tribute-ir`

## Dialects

Dialects are split across two crates:

- **trunk-ir** (`crates/trunk-ir/src/dialect/`): Target-independent dialects
- **tribute-ir** (`crates/tribute-ir/src/dialect/`): Tribute-specific
  high-level dialects

### Infrastructure (trunk-ir)

| Dialect | File | Purpose |
| ------- | ---- | ------- |
| `core` | `core.rs` | Core types: i32, f64, nil, tuple, string, ptr, array, ref_ |

### High-level Tribute (tribute-ir)

| Dialect | File | Purpose |
| ------- | ---- | ------- |
| `ability` | `ability.rs` | Evidence-based handler dispatch: evidence_lookup, evidence_extend, handler_table, handler_entry |
| `adt` | `adt.rs` | ADT ops: struct_new, variant_new, array_get, field_get |
| `closure` | `closure.rs` | Closures and captures |
| `tribute_rt` | `tribute_rt.rs` | Runtime boxing/unboxing (box_int, unbox_int, etc.) and RC ops (retain, release) |

### Mid-level (trunk-ir)

| Dialect | File | Purpose |
| ------- | ---- | ------- |
| `func` | `func.rs` | Function ops: func, call, call_indirect, return, constant |
| `cont` | `cont.rs` | Continuation-based control flow |
| `scf` | `scf.rs` | Structured control flow: if, while, for |
| `arith` | `arith.rs` | Arithmetic: add, mul, div, cmp, etc. |
| `mem` | `mem.rs` | Memory operations |

### Low-level (trunk-ir)

| Dialect | File | Purpose |
| ------- | ---- | ------- |
| `wasm` | `wasm.rs` | WebAssembly target ops |

## `#[dialect]` Macro

Operations are defined using the `#[dialect]` attribute macro:

```rust
#[trunk_ir::dialect]
mod func {
    // Function definition with body region
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)] {}
    }

    // Function call with callee attribute
    #[attr(callee: Symbol)]
    fn call(#[rest] args: ()) -> result {}

    // Return with optional value
    fn r#return(#[rest] values: ()) {}
}
```

**Annotations**:

- `#[attr(...)]` - Attributes (metadata stored on operation)
- `#[region(...)]` - Regions (nested control flow)
- `#[rest]` - Variadic operands
- `-> result` - Operation produces a result value

## Type System

Types are defined per-dialect and interned:

```rust
// Creating types via typed constructors
let nil_ty = core::nil(ctx).as_type_ref();
let func_ty = core::func(ctx, return_ty, params, effect).as_type_ref();

// Type has: dialect, name, params, attrs
let data = ctx.types.get(nil_ty);
data.dialect  // Symbol("core")
data.name     // Symbol("nil")
data.params   // []
```

## Working with IR

```rust
// Create an IR context
let mut ctx = IrContext::new();

// Create operations via typed constructors
let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
let call = func::call(&mut ctx, loc, [c.result(&ctx)], i32_ty, Symbol::new("add"));

// Match operations via typed wrappers
if let Ok(func) = func::Func::from_op(&ctx, op) {
    let name = func.sym_name(&ctx);
    let body = func.body(&ctx);
}

// Check operation type
if func::Call::matches(&ctx, op) { ... }
```
