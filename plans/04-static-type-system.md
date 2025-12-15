# Static Type System Plan

**Status**: ❌ **DEPRECATED** - Superseded by new language design
**Superseded By**: [../new-plans/tribute-type-inference.md](../new-plans/tribute-type-inference.md), [../new-plans/tribute-types.md](../new-plans/tribute-types.md)
**Deprecation Date**: 2024-12-15
**Reason**: New design includes row polymorphic effect typing and ability (algebraic effects) system integration

> ⚠️ **This plan has been deprecated in favor of the new comprehensive language design.**
> See `new-plans/` directory for the updated type system design that includes:
> - Row polymorphic effect types (`fn(a) ->{E} b`)
> - Ability (algebraic effects) system
> - struct/enum type declarations
> - UFCS (Uniform Function Call Syntax)
> - Type inference with effect inference

---

## Original Overview (DEPRECATED)
Add a gradual static type system to Tribute using bidirectional type checking, allowing optional type annotations while maintaining compatibility with untyped code.

## Priority: Medium (3/4)
Important for language maturity. Using bidirectional type checking makes implementation more feasible.

## Architecture: Multi-Level IR with Typed IR

**AST → HIR → Typed IR → Cranelift IR**

1. **AST**: Tree-sitter parsing + type annotation parsing
2. **HIR**: Semantic analysis + scope resolution (existing)
3. **Typed IR**: Type inference + type checking (new stage)
4. **Cranelift IR**: Type-informed code generation

### New Crates:
- `tribute-types`: Type definitions and type checker
- `tribute-typed-ir`: Typed intermediate representation

## Type System Design

### Core Types

#### Phase 1: Basic Types (MVP)
```tribute
// Primitive types
Int, Float, String, Bool, Unit

// Function types
(Int, Int) -> Int
```

#### Phase 2: Advanced Types (Future)
```tribute
// Generic types (standard library)
List<T>, Option<T>, Result<T, E>

// User-defined types (requires ADT system)
struct Point { x: Int, y: Int }
enum Color { Red, Green, Blue }
```

### Type Annotations

#### Phase 1: Basic Annotations
```tribute
// Variable annotations
let x: Int = 42
let name: String = "Tribute"

// Function annotations
fn add(a: Int, b: Int) -> Int {
  a + b
}
```

#### Phase 2: Generic Annotations (Future)
```tribute
// Generic functions
fn identity<T>(x: T) -> T {
  x
}
```

## Implementation Steps

### Phase 1: Infrastructure (Days 1-3)
1. **Create `tribute-types` crate**
   ```rust
   // Phase 1: Basic types only
   pub enum Type {
       Int, Float, String, Bool, Unit,
       Function { params: Vec<Type>, ret: Box<Type> },
       Unknown, // For inference
       Any,     // For gradual typing
   }
   
   // Phase 2: Add generics later
   // Generic(String, Vec<Type>), // Generic<T, U, ...>
   ```

2. **Create `tribute-typed-ir` crate**
   ```rust
   #[salsa::tracked]
   pub struct TypedProgram<'db> {
       pub functions: BTreeMap<Identifier, TypedFunction<'db>>,
       pub type_errors: Vec<TypeError>,
   }
   
   #[salsa::tracked]
   pub struct TypedExpr<'db> {
       pub expr: Expr,
       pub inferred_type: Type,
       pub span: Span,
   }
   ```

3. **Extend Tree-sitter grammar** for type annotations
   ```tribute
   fn add(a: Int, b: Int) -> Int { a + b }
   let x: Int = 42
   ```

### Phase 2: Bidirectional Type Checker (Days 4-6)
1. **HIR → Typed IR transformation**
   ```rust
   #[salsa::tracked]
   fn type_check_hir_program(db: &dyn Db, hir: HirProgram) -> TypedProgram;
   
   #[salsa::tracked]
   fn type_check_hir_function(db: &dyn Db, func: HirFunction) -> TypedFunction;
   ```

2. **Bidirectional Algorithm**
   ```rust
   impl TypeChecker {
       fn check_expr(&mut self, hir_expr: HirExpr, expected: Type) -> TypedExpr;
       fn infer_expr(&mut self, hir_expr: HirExpr) -> (TypedExpr, Type);
   }
   ```

3. **Type Environment Management**
   ```rust
   struct TypeEnv {
       vars: HashMap<Identifier, Type>,
       functions: HashMap<Identifier, FunctionType>,
   }
   ```

### Phase 3: Pipeline Integration (Days 7-8)
1. **New Compilation Pipeline**
   ```rust
   // main tribute crate
   pub fn compile_with_types(source: &str) -> Result<TypedProgram, CompileError> {
       let ast = parse_str(db, "input", source)?;
       let hir = lower_ast_to_hir(db, ast)?;
       let typed_ir = type_check_hir_program(db, hir)?; // new stage
       Ok(typed_ir)
   }
   ```

2. **Cranelift Integration**
   ```rust
   // tribute-cranelift crate
   fn compile_typed_ir(typed_ir: TypedProgram) -> CompiledModule {
       // Type-informed optimized code generation
   }
   ```

### Phase 4: Error Reporting (Days 9-10)
1. **Rich Type Errors**
   ```rust
   #[derive(Debug, Clone)]
   pub struct TypeError {
       pub kind: TypeErrorKind,
       pub span: Span,
       pub suggestion: Option<String>,
   }
   ```

2. **Integration with Diagnostics**
   ```rust
   impl From<TypeError> for Diagnostic {
       fn from(err: TypeError) -> Self {
           // Integration with existing diagnostic system
       }
   }
   ```

3. **Error Examples**
   ```
   error: type mismatch
     --> example.trb:3:5
     3 | let x: Int = "hello"
       |     ^^^^^ expected Int, found String
   ```

### Phase 5: Gradual Typing (Days 11-12)
1. **Any Type Support**
   - Untyped code defaults to `Any` type
   - Explicit casts at typed/untyped boundaries
   - No runtime overhead for fully typed code

2. **Backward Compatibility**
   - Existing `.trb` files work without modification
   - Type annotations are optional
   - Gradual migration path

### Phase 6: Advanced Features (Future)
1. **Generic Type System**
   - Type parameters and constraints
   - Generic functions and data structures
   - Standard library types (List, Option, Result)

2. **Algebraic Data Types (ADTs)**
   - User-defined structs and enums
   - Pattern matching exhaustiveness
   - Constructor functions

3. **Trait/Interface System**
   - Type classes and implementations
   - Generic constraints

## Design Decisions

### Gradual Typing
- Untyped code defaults to `Any` type
- Explicit casts at typed/untyped boundaries
- No runtime overhead for fully typed code

### Type Inference Algorithm
- **Bidirectional Type Checking** (instead of Hindley-Milner)
- **Check Mode**: Verify expressions against expected types
- **Infer Mode**: Infer types from expressions
- Local type inference (no global inference)
- Explicit annotations for function signatures
- Infer variable types from initialization

### Error Recovery
- Continue checking despite type errors
- Provide helpful error messages
- Suggest type annotations

## Bidirectional Type Checking Algorithm

### Core Functions:
```rust
fn check_expr(expr: HirExpr, expected: Type) -> Result<TypedExpr, TypeError>;
fn infer_expr(expr: HirExpr) -> Result<(TypedExpr, Type), TypeError>;
```

### Algorithm Benefits:
- **Simpler Implementation**: Easier than Hindley-Milner
- **Better Error Messages**: Clear type mismatch reporting
- **Predictable**: Type inference results are intuitive
- **Gradual**: Easy to integrate with existing untyped code

### Example:
```rust
impl TypeChecker {
    fn check_expr(&mut self, expr: HirExpr, expected: Type) -> Result<TypedExpr, TypeError> {
        match expr.expr {
            Expr::Number(n) => {
                self.check_subtype(Type::Int, expected)?;
                Ok(TypedExpr::new(expr.expr, Type::Int, expr.span))
            }
            Expr::Call { func, args } => {
                let (func_typed, func_type) = self.infer_expr(*func)?;
                match func_type {
                    Type::Function { params, ret } => {
                        // Check arguments against parameters
                        let mut typed_args = Vec::new();
                        for (arg, param_type) in args.iter().zip(params.iter()) {
                            let typed_arg = self.check_expr(arg.clone(), param_type.clone())?;
                            typed_args.push(typed_arg);
                        }
                        self.check_subtype(*ret, expected)?;
                        Ok(TypedExpr::new(
                            Expr::Call { func: Box::new(func_typed), args: typed_args },
                            *ret,
                            expr.span
                        ))
                    }
                    _ => Err(TypeError::NotCallable { span: expr.span })
                }
            }
            _ => {
                let (typed_expr, inferred_type) = self.infer_expr(expr)?;
                self.check_subtype(inferred_type, expected)?;
                Ok(typed_expr)
            }
        }
    }
}
```

## Technical Challenges
- Balancing inference power vs simplicity
- Error message quality
- Performance of type checking
- Integration with existing codebase
- Salsa integration for incremental type checking

## Dependencies
- Syntax modernization (✅ completed)
- HIR stability (✅ stable)
- New crates: `tribute-types`, `tribute-typed-ir`

## Success Criteria
- Type check standard library
- Catch common type errors
- No false positives
- Reasonable compile-time performance
- Clear migration path for untyped code