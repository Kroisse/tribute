# TrunkIR System

TrunkIR is Tribute's multi-level dialect IR, inspired by MLIR's dialect concept.

## Core Structures

Located in `crates/trunk-ir/src/ir.rs`:

- **`Operation`** - Generic IR operation with operands, results, regions, successors, and attributes
- **`Value`** - SSA value with definition site and index
- **`ValueDef`** - Either operation result or block argument
- **`Block`** - Basic block with arguments and operations
- **`Region`** - Control flow region (list of blocks)
- **`Attribute`** - Value attributes (bool, int, float, string, type, symbol, span)
- **`Symbol`** - Interned identifier (cheap comparison)
- **`Type`** - Dialect-parametric type (dialect.name + params + attrs)

## Dialects

All dialects are in `crates/trunk-ir/src/dialect/`:

### Infrastructure
| Dialect | File | Purpose |
|---------|------|---------|
| `core` | `core.rs` | Core types: i32, f64, nil, tuple, string, ptr, array, ref_ |
| `ty` | `ty.rs` | Type definition operations |

### High-level (Source-like)
| Dialect | File | Purpose |
|---------|------|---------|
| `src` | `src.rs` | Unresolved AST ops: var, call, path, binop, lambda, tuple, const |
| `ability` | `ability.rs` | Ability (algebraic effects) operations |
| `adt` | `adt.rs` | ADT ops: struct_new, variant_new, array_get, field_get |

### Mid-level
| Dialect | File | Purpose |
|---------|------|---------|
| `func` | `func.rs` | Function ops: func, call, call_indirect, return, constant |
| `case` | `case.rs` | Pattern matching: case, case_region |
| `pat` | `pat.rs` | Pattern operations |
| `cont` | `cont.rs` | Continuation-based control flow |
| `closure` | `closure.rs` | Closures and captures |
| `scf` | `scf.rs` | Structured control flow: if, while, for |
| `arith` | `arith.rs` | Arithmetic: add, mul, div, cmp, etc. |
| `mem` | `mem.rs` | Memory operations |
| `list` | `list.rs` | List operations |

### Low-level (Target-specific)
| Dialect | File | Purpose |
|---------|------|---------|
| `wasm` | `wasm.rs` | WebAssembly target ops |

## dialect! Macro

Operations are defined using the `dialect!` macro in `crates/trunk-ir/src/ops.rs`:

```rust
dialect! {
    mod func {
        // Function definition with body region
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() { #[region(body)] {} };

        // Function call with callee attribute
        #[attr(callee: SymbolRef)]
        fn call(#[rest] args) -> result;

        // Return with optional value
        fn r#return(value);
    }
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
// Creating types
let i64_ty = core::I64::new(db);
let func_ty = func::Fn::new(db, param_types, return_type, effect);

// Type has: dialect, name, params, attrs
type.dialect()  // "core"
type.name()     // "i64"
type.params()   // []
```

## Working with IR

```rust
// Create a module
let mut module = Module::new(db);

// Add an operation
let op = Operation::new(db, "func.func", ...);
module.add_op(op);

// Traverse operations and convert to typed wrappers
for op in module.ops() {
    // Use DialectOp::from_operation to get typed wrapper
    if let Ok(func) = func::Func::from_operation(db, op) {
        // Work with typed func operation
        let name = func.sym_name(db);
        let body = func.body(db);
    }
}
```
