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
- **`Symbol`** - Interned identifier (4 bytes, O(1) comparison)
  - Can hold simple names or qualified paths (e.g., `"map"` or `"std::List::map"`)
  - Path operations available via `ModulePathExt` trait in `tribute-ir`
- **`Type`** - Dialect-parametric type (dialect.name + params + attrs)

## Dialects

Dialects are split across two crates:
- **trunk-ir** (`crates/trunk-ir/src/dialect/`): Target-independent dialects
- **tribute-ir** (`crates/tribute-ir/src/dialect/`): Tribute-specific high-level dialects

### Infrastructure (trunk-ir)
| Dialect | File | Purpose |
|---------|------|---------|
| `core` | `core.rs` | Core types: i32, f64, nil, tuple, string, ptr, array, ref_ |

### High-level Tribute (tribute-ir)
| Dialect | File | Purpose |
|---------|------|---------|
| `ability` | `ability.rs` | Evidence-based handler dispatch: evidence_lookup, evidence_extend, marker_prompt |
| `adt` | `adt.rs` | ADT ops: struct_new, variant_new, array_get, field_get |
| `closure` | `closure.rs` | Closures and captures |

### Mid-level (trunk-ir)
| Dialect | File | Purpose |
|---------|------|---------|
| `func` | `func.rs` | Function ops: func, call, call_indirect, return, constant |
| `cont` | `cont.rs` | Continuation-based control flow |
| `scf` | `scf.rs` | Structured control flow: if, while, for |
| `arith` | `arith.rs` | Arithmetic: add, mul, div, cmp, etc. |
| `mem` | `mem.rs` | Memory operations |

### Low-level (trunk-ir)
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
